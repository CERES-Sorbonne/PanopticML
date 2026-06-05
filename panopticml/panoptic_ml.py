import base64
import os
import pickle
from collections import defaultdict
from io import BytesIO
from typing import TYPE_CHECKING

import pacmap
import umap


def check_huggingface_connection():
    import socket
    try:
        socket.create_connection(("huggingface.co", 443), timeout=2)
        return True
    except (OSError, socket.timeout):
        os.environ['HF_HUB_OFFLINE'] = '1'
        return False

check_huggingface_connection()

from enum import Enum

import msgspec
import numpy as np
import requests
from PIL import Image
from pydantic import BaseModel
from sklearn.manifold import TSNE

from panoptic.core.plugin.plugin import APlugin
from panoptic.models.action_models import (
    ActionContext, ActionResult, Group, InputFile, Notif, NotifType,
    OwnVectorType, PropertyId, Score, ScoreList,
)
from panoptic.core.databases.media.models import Map, Vector, VectorType
from panoptic.models.models import Instance

from .compute import make_clusters
from .compute.clustering import cluster_by_text
from .compute.faiss_tree import FaissTreeManager
from .compute.transformer import TransformerManager, type_to_class_mapping, extract_model_type
from .compute_vector_task import ComputeVectorsTask
from .utils import is_image_url, ClusterByTagsEnum, process_tags, normalize_positions


def group_by_sha1(instances: list[Instance]) -> dict:
    result: dict[str, list] = {}
    for instance in instances:
        if instance.sha1 not in result:
            result[instance.sha1] = []
        result[instance.sha1].append(instance)
    return result


class PluginParams(BaseModel):
    compute_on_import: bool = True
    save_text_searches: bool = False


class ModelEnum(Enum):
    clip = "openai/clip-vit-base-patch32"
    mobilenet = "google/mobilenet_v2_1.0_224"
    siglip = "google/siglip2-so400m-patch16-naflex"
    dinov = "facebook/dinov2-base"


def vector_name(vec_type: VectorType) -> str:
    res = f"{vec_type.id}: {vec_type.source}"
    if vec_type.params:
        for k in vec_type.params:
            res += f'_{k}_{vec_type.params[k]}'
    return res


class PanopticML(APlugin):
    """
    Default Machine Learning plugin for Panoptic.
    Uses CLIP to generate vectors and FAISS for clustering / similarity functions.
    """

    def __init__(self, name: str, project, plugin_path: str):
        self.params = PluginParams()
        super().__init__(name=name, project=project, plugin_path=plugin_path)
        self.project.on_instance_import(self._on_import)
        self.project.on_folder_delete(self._on_folder_delete)
        self.add_action_easy(self.create_default_vector_type, ['vector_type'])
        self.add_action_easy(self.create_custom_vector_type, ['vector_type'])
        self.add_action_easy(self.compute_vectors, ['vector'])
        self.add_action_easy(self.find_images, ['similar'])
        self.add_action_easy(self.find_images_from_file, ['execute'])
        self.add_action_easy(self.compute_clusters, ['group'])
        self.add_action_easy(self.cluster_by_tags, ['group'])
        self.add_action_easy(self.find_duplicates, ['group'])
        self.add_action_easy(self.search_by_text, ['execute', 'text_search'])
        self.add_action_easy(self.pacmap, ['map', 'execute'])
        self.add_action_easy(self.umap, ['map', 'execute'])
        self.add_action_easy(self.tsne, ['map', 'execute'])

        self.trees = FaissTreeManager(self)
        self.transformers = TransformerManager()
        self.text_vectors: defaultdict = defaultdict(dict)

    def _start(self) -> None:
        for t in self.vector_types:
            self.trees.get(t)
            self.transformers.get(t)  # pre-warm: load model weights at startup

        if len(self.vector_types) == 0:
            vt = self.project.upsert_vector_type(
                VectorType(id=-1, source=self.name,
                           params={"model": ModelEnum.clip.value, "greyscale": False})
            )
            self.vector_types.append(vt)

    # ------------------------------------------------------------------
    # Vector type creation
    # ------------------------------------------------------------------

    def create_default_vector_type(self, ctx: ActionContext, model: ModelEnum = ModelEnum.clip, greyscale: bool = False) -> ActionResult:
        """Create a vector type using a predefined model.
        @model: the embedding model to use
        @greyscale: convert images to greyscale before embedding
        """
        vt = VectorType(id=-1, source=self.name, params={"model": model.value, "greyscale": greyscale})
        res = self.project.upsert_vector_type(vt)
        return ActionResult(value=res)

    def create_custom_vector_type(self, ctx: ActionContext, model: str = '', greyscale: bool = False) -> ActionResult:
        """Create a vector type using a custom HuggingFace model name.
        @model: HuggingFace model identifier (e.g. openai/clip-vit-base-patch32)
        @greyscale: convert images to greyscale before embedding
        """
        vt = VectorType(id=-1, source=self.name, params={"model": model, "greyscale": greyscale})
        res = self.project.upsert_vector_type(vt)
        return ActionResult(value=res)

    # ------------------------------------------------------------------
    # Vector computation
    # ------------------------------------------------------------------

    def compute_vectors(self, context: ActionContext, vec_type: OwnVectorType) -> ActionResult:
        """Compute image embedding vectors for selected images.
        @vec_type: the vector space to compute into
        """
        instances = self._get_instances(context)
        self._enqueue_vectors_task(instances, vec_type)
        return ActionResult(notifs=[Notif(
            type=NotifType.INFO,
            name="ComputeVector",
            message=f"Started computing vectors {vector_name(vec_type)} for {len(instances)} images",
        )])

    def _enqueue_vectors_task(self, instances: list, vec_type: VectorType) -> None:
        if not instances:
            return
        task = ComputeVectorsTask(self, vec_type, instances)
        self.project.add_task(task)

    def rebuild_index(self, vec_type: VectorType) -> None:
        self.trees.rebuild_tree(vec_type)

    # ------------------------------------------------------------------
    # Import / delete event hooks
    # ------------------------------------------------------------------

    def _on_import(self, instances: list) -> None:
        if not self.params.compute_on_import:
            return
        for vt in self.vector_types:
            self._enqueue_vectors_task(instances, vt)

    def _on_folder_delete(self, folders: list) -> None:
        for vt in self.vector_types:
            self.trees.rebuild_tree(vt)

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def compute_clusters(self, context: ActionContext, vec_type: OwnVectorType,
                         nb_clusters: int = 10) -> ActionResult:
        """Compute image clusters with Faiss K-means.
        @vec_type: vector space to cluster in
        @nb_clusters: requested number of clusters
        """
        instances = self._get_instances(context)
        sha1_to_instance = group_by_sha1(instances)
        sha1s = list(sha1_to_instance.keys())
        if not sha1s:
            return ActionResult(notifs=[Notif(NotifType.ERROR, name="NoData", message="No instances found")])

        vectors = self.project.get_vectors(type_id=vec_type.id, sha1s=sha1s)
        if not vectors:
            return ActionResult(notifs=[Notif(
                NotifType.ERROR,
                name="NoData",
                message=f"No vectors ({vector_name(vec_type)}) found. Compute vectors first.",
            )])

        clusters, distances = make_clusters(vectors, method="kmeans", nb_clusters=nb_clusters)

        groups = []
        for i, (cluster, distance) in enumerate(zip(clusters, distances)):
            group = Group(score=Score(value=float(distance), min=0, max=100, max_is_best=False))
            group.sha1s = sorted(cluster)
            group.name = f"Cluster {i}"
            groups.append(group)

        return ActionResult(groups=groups)

    def find_duplicates(self, context: ActionContext, vec_type: OwnVectorType,
                        min_similarity: float = 0.95) -> ActionResult:
        """Find near-duplicate images by similarity threshold.
        @vec_type: vector space to compare in
        @min_similarity: minimum cosine similarity to consider images duplicates
        """
        instances = self._get_instances(context)
        sha1_to_instance = group_by_sha1(instances)
        sha1s = list(sha1_to_instance.keys())
        if not sha1s:
            return ActionResult()

        pano_vectors = self.project.get_vectors(type_id=vec_type.id, sha1s=sha1s)
        tree = self.trees.get(vec_type)
        groups = self._compute_duplicate_groups(tree, pano_vectors, min_similarity)
        return ActionResult(groups=groups)

    @staticmethod
    def _compute_duplicate_groups(tree, pano_vectors, min_similarity):
        vectors_sha1 = [(i.data, i.sha1) for i in pano_vectors]
        sha1_set = {sha1 for _, sha1 in vectors_sha1}
        already_in_clusters = set()
        groups = []
        for vector, sha1 in vectors_sha1:
            if sha1 in already_in_clusters:
                continue
            res = tree.query([vector], 150)
            filtered = [r for r in res if r['dist'] >= min_similarity and r['sha1'] in sha1_set]
            res_sha1s = [r['sha1'] for r in filtered]
            res_scores = [r['dist'] for r in filtered]
            if len(res_sha1s) <= 1:
                continue
            already_in_clusters.update(res_sha1s)
            groups.append(Group(sha1s=res_sha1s, scores=ScoreList(min=0, max=1, max_is_best=True, values=res_scores)))
        return groups

    # ------------------------------------------------------------------
    # Similarity search
    # ------------------------------------------------------------------

    def find_images(self, context: ActionContext, vec_type: OwnVectorType,
                    max_results: int = 200) -> ActionResult:
        return self.find_images_from_file(context, vec_type, image_file=None,
                                          max_results=max_results)

    def find_images_from_file(self, context: ActionContext, vec_type: OwnVectorType,
                    image_file: InputFile = None, max_results: int = 200) -> ActionResult:
        """Find similar images using cosine similarity.
        @vec_type: vector space to search in
        @image_file: optional uploaded image to search by (defaults to selected images)
        @max_results: maximum number of similar images to return (top-k neighbours)
        """
        ignore_sha1s: set = set()

        if image_file is None:
            instances = self._get_instances(context)
            sha1s = [i.sha1 for i in instances]
            ignore_sha1s = set(sha1s)
            raw_vectors = self.project.get_vectors(type_id=vec_type.id, sha1s=sha1s)
            vector_datas = [x.data for x in raw_vectors]
        else:
            im = Image.open(BytesIO(base64.b64decode(image_file)))
            if im.mode in ("RGBA", "P"):
                im = im.convert("RGB")
            transformer = self.transformers.get(vec_type)
            vector_datas = [transformer.to_vector(im)]

        if not vector_datas:
            return ActionResult(notifs=[Notif(
                NotifType.ERROR,
                name="NoData",
                message=f"No vectors ({vector_name(vec_type)}) found. Compute vectors first.",
            )])

        tree = self.trees.get(vec_type)
        if not tree:
            return ActionResult(notifs=[Notif(
                NotifType.ERROR, name="NoFaissTree",
                message=f"No Faiss tree for {vector_name(vec_type)}",
            )])

        # cap the number of neighbours: querying the full tree (k defaults to
        # 999999) returns every image in the dataset, which overwhelms the
        # frontend on large collections (e.g. 500k instances).
        # ask for a few extra to compensate for ignored (queried) sha1s.
        k = max(1, max_results) + len(ignore_sha1s)
        res = tree.query(vector_datas, k=k)
        index = {r['sha1']: r['dist'] for r in res if r['sha1'] not in ignore_sha1s}
        index = dict(list(index.items())[:max(1, max_results)])
        res_sha1s = list(index.keys())
        scores = ScoreList(min=0, max=1, values=[index[s] for s in res_sha1s],
                           max_is_best=True, description="Cosine similarity (1 = identical)")
        return ActionResult(groups=[Group(sha1s=res_sha1s, scores=scores)])

    def search_by_text(self, context: ActionContext, vec_type: OwnVectorType,
                       text: str = '', min_similarity: float = 0.5) -> ActionResult:
        """Search images by text similarity.
        @vec_type: vector space to search in
        @text: search query or image URL
        @min_similarity: minimum similarity threshold (0–1)
        """
        if not text:
            return ActionResult(notifs=[Notif(
                NotifType.ERROR, name="EmptySearchText",
                message="Please provide a non-empty search text",
            )])

        context_instances = self._get_instances(context)
        context_sha1s = {i.sha1 for i in context_instances}

        tree = self.trees.get(vec_type)
        if not tree:
            return ActionResult(notifs=[Notif(
                NotifType.ERROR, name="NoFaissTree",
                message=f"No Faiss tree for {vector_name(vec_type)}",
            )])

        self._load_text_vectors(vec_type)
        text_vectors = []

        if text in self.text_vectors[vec_type]:
            resulting_images = tree.query([self.text_vectors[vec_type][text]])
        else:
            transformer = self.transformers.get(vec_type)
            try:
                if is_image_url(text):
                    im = Image.open(requests.get(text, stream=True).raw)
                    vec = transformer.to_vector(im)
                    resulting_images = tree.query([vec])
                else:
                    resulting_images, text_vectors = tree.query_texts([text], transformer, return_vec=True)
            except ValueError as e:
                return ActionResult(notifs=[Notif(NotifType.ERROR, name="TextSimilarityError", message=str(e))])

        filtered = [inst for inst in resulting_images if inst['sha1'] in context_sha1s]
        index = {r['sha1']: r['dist'] for r in filtered}
        res_sha1s = np.asarray(list(index.keys()))
        res_scores = np.asarray([index[sha1] for sha1 in res_sha1s])

        max_text_sim = type_to_class_mapping[extract_model_type(vec_type)].max_text_sim
        remapped = np.around(np.interp(res_scores, [0, max_text_sim], [0, 1]), decimals=2)
        mask = remapped >= min_similarity
        final_sha1s = res_sha1s[mask].tolist()
        final_scores = remapped[mask].tolist()

        if self.params.save_text_searches and len(text_vectors) > 0:
            self._save_text_vectors([text], text_vectors, vec_type)

        group = Group(sha1s=final_sha1s,
                      scores=ScoreList(min=0, max=1, values=final_scores,
                                       description="Text-image similarity remapped to [0,1]"))
        group.name = f"Text Search: {text}"
        return ActionResult(groups=[group])

    def cluster_by_tags(self, context: ActionContext, tags: PropertyId,
                        vec_type: OwnVectorType, min_similarity: float = 0.5,
                        parent_tags: ClusterByTagsEnum = ClusterByTagsEnum.use,
                        multiple: bool = False, prefix: str = '') -> ActionResult:
        """Cluster images guided by a Tag/MultiTag property.
        @tags: the tag or multi-tag property to guide clustering
        @vec_type: vector space to use
        @min_similarity: minimum text-image similarity to include an image
        @parent_tags: how to handle parent tag names
        @multiple: allow an image to appear in multiple clusters
        @prefix: prefix to prepend to each tag before embedding
        """
        props = self.project.get_properties(id=[int(tags)])
        if not props:
            return ActionResult(notifs=[Notif(
                NotifType.ERROR, name="PropertyNotFound",
                message=f"Property {tags} not found",
            )])
        tag_prop = props[0]
        if tag_prop.dtype not in ('tag', 'multi_tags'):
            return ActionResult(notifs=[Notif(
                NotifType.ERROR, name="WrongPropertyType",
                message=f"Property <{tag_prop.name}> is not a Tag or MultiTags property",
            )])

        instances = self._get_instances(context)
        sha1_to_instance = group_by_sha1(instances)
        sha1s = list(sha1_to_instance.keys())
        if not sha1s:
            return ActionResult()

        all_tags = self.project.get_tags(list_id=tag_prop.tag_list_id)
        tags_text = process_tags(all_tags, parent_tags=parent_tags)
        if prefix:
            tags_text = [prefix + t for t in tags_text]

        texts_to_transform = []
        text_vectors = []
        transformer = None
        self._load_text_vectors(vec_type)

        for text in tags_text:
            if text in self.text_vectors[vec_type]:
                text_vectors.append(self.text_vectors[vec_type][text])
            else:
                texts_to_transform.append(text)

        if texts_to_transform:
            transformer = self.transformers.get(vec_type)
            transformed = transformer.get_text_vectors(texts_to_transform)
            text_vectors = [*transformed, *text_vectors]
            self._save_text_vectors(texts_to_transform, transformed, vec_type)

        pano_vectors = self.project.get_vectors(type_id=vec_type.id, sha1s=sha1s)
        if not pano_vectors:
            return ActionResult(notifs=[Notif(
                NotifType.ERROR, name="NoData",
                message=f"No vectors ({vector_name(vec_type)}) found. Compute vectors first.",
            )])

        max_text_sim = transformer.max_text_sim if transformer else 0.2
        groups = cluster_by_text(pano_vectors, text_vectors, tags_text, min_similarity, max_text_sim, multiple)
        return ActionResult(groups=groups)

    # ------------------------------------------------------------------
    # Dimensionality reduction / maps
    # ------------------------------------------------------------------

    def pacmap(self, ctx: ActionContext, vec_type: OwnVectorType, map_name: str = "") -> ActionResult:
        """Compute a PaCMAP 2D map for the selected images.
        @vec_type: vector space to reduce
        @map_name: name for the saved map (auto-generated if empty)
        """
        instances = self._get_instances(ctx)
        sha1s = list({i.sha1 for i in instances})
        vectors = self.project.get_vectors(vec_type.id, sha1s=sha1s)
        print(len(vectors))
        if len(vectors) < 2:
            return ActionResult(notifs=[Notif(NotifType.ERROR, name="NoData",
                message=f"Need at least 2 vectors to compute a map (got {len(vectors)}).")])
        points = self.project.run_in_executor(self._get_pacmap_coordinates, vectors)
        return self._save_map(points, vec_type, map_name or f"pacmap: {vec_type.params['model']}")

    def tsne(self, ctx: ActionContext, vec_type: OwnVectorType, map_name: str = "") -> ActionResult:
        """Compute a t-SNE 2D map for the selected images.
        @vec_type: vector space to reduce
        @map_name: name for the saved map (auto-generated if empty)
        """
        instances = self._get_instances(ctx)
        sha1s = list({i.sha1 for i in instances})
        vectors = self.project.get_vectors(vec_type.id, sha1s=sha1s)
        if len(vectors) < 2:
            return ActionResult(notifs=[Notif(NotifType.ERROR, name="NoData",
                message=f"Need at least 2 vectors to compute a map (got {len(vectors)}).")])
        points = self.project.run_in_executor(self._get_tsne_coordinates, vectors)
        return self._save_map(points, vec_type, map_name or f"tsne: {vec_type.params['model']}")

    def umap(self, ctx: ActionContext, vec_type: OwnVectorType, map_name: str = "") -> ActionResult:
        """Compute a UMAP 2D map for the selected images.
        @vec_type: vector space to reduce
        @map_name: name for the saved map (auto-generated if empty)
        """
        instances = self._get_instances(ctx)
        sha1s = list({i.sha1 for i in instances})
        vectors = self.project.get_vectors(vec_type.id, sha1s=sha1s)
        if len(vectors) < 2:
            return ActionResult(notifs=[Notif(NotifType.ERROR, name="NoData",
                message=f"Need at least 2 vectors to compute a map (got {len(vectors)}).")])
        points = self.project.run_in_executor(self._get_umap_coordinates, vectors)
        return self._save_map(points, vec_type, map_name or f"umap: {vec_type.params['model']}")

    def _save_map(self, points: dict, vec_type: VectorType, name: str) -> ActionResult:
        flat: list = []
        for sha1, (x, y) in points.items():
            flat += [sha1, x, y]
        flat = normalize_positions(flat, 100)
        point_map = self.project.upsert_map(Map(
            id=-1, source=self.name, name=name,
            key='sha1', count=len(points), data=flat,
        ))
        return ActionResult(value={f.name: getattr(point_map, f.name) for f in msgspec.structs.fields(point_map)})

    @staticmethod
    def _get_pacmap_coordinates(vectors: list[Vector]) -> dict:
        if len(vectors) < 2:
            return {}
        data = np.asarray([v.data for v in vectors])
        embedding = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0)
        result = embedding.fit_transform(data, init="pca")
        return {vectors[i].sha1: result[i].tolist() for i in range(result.shape[0])}

    @staticmethod
    def _get_tsne_coordinates(vectors: list[Vector]) -> dict:
        if len(vectors) < 2:
            return {}
        data = np.asarray([v.data for v in vectors])
        result = TSNE(n_components=2, perplexity=30, random_state=None).fit_transform(data)
        return {vectors[i].sha1: result[i].tolist() for i in range(result.shape[0])}

    @staticmethod
    def _get_umap_coordinates(vectors: list[Vector]) -> dict:
        if len(vectors) < 2:
            return {}
        data = np.asarray([v.data for v in vectors])
        result = umap.UMAP(n_components=2, random_state=None).fit_transform(data)
        return {vectors[i].sha1: result[i].tolist() for i in range(result.shape[0])}

    # ------------------------------------------------------------------
    # Text vector cache
    # ------------------------------------------------------------------

    def _save_text_vectors(self, texts, text_vectors: list, vec_type: VectorType) -> None:
        for text, vec in zip(texts, text_vectors):
            self.text_vectors[vec_type][text] = vec
        with open(self.data_path / (str(vec_type) + '_text_vectors.pkl'), 'wb') as f:
            pickle.dump(self.text_vectors[vec_type], f)

    def _load_text_vectors(self, vec_type: VectorType) -> None:
        if self.text_vectors[vec_type]:
            return
        path = self.data_path / (str(vec_type) + '_text_vectors.pkl')
        if path.exists():
            with open(path, 'rb') as f:
                self.text_vectors[vec_type] = pickle.load(f)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_instances(self, context: ActionContext) -> list:
        if context.instance_ids:
            return self.project.get_instances(id=context.instance_ids)
        return self.project.get_instances()

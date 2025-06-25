from typing import Dict

import numpy as np
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

from panoptic.core.plugin.plugin import APlugin
from panoptic.core.plugin.plugin_project_interface import PluginProjectInterface
from panoptic.models import Instance, ActionContext, PropertyId, PropertyType
from panoptic.models.results import Group, ActionResult, Notif, NotifType, NotifFunction, ScoreList, Score
from panoptic.utils import group_by_sha1

from .compute import make_clusters
from .compute.faiss_tree import load_faiss_tree, create_faiss_tree, FaissTree
from .compute.similarity import get_text_vectors
from .compute.transformers import get_transformer, TransformerName
from .compute_vector_task import ComputeVectorTask
from .models import VectorType


class PluginParams(BaseModel):
    """
    @greyscale: if this is checked, vectors can be recomputed but this time images will be converted to greyscale before
    @model: the name of the transformer to user, values: clip | mobilenet, default to clip
    """
    greyscale: bool = False
    model: str = "clip"


class PanopticML(APlugin):
    """
    Default Machine Learning plugin for Panoptic
    Uses CLIP to generate vectors and FAISS for clustering / similarity functions
    """

    def __init__(self, project: PluginProjectInterface, plugin_path: str, name: str):
        super().__init__(name=name, project=project, plugin_path=plugin_path)
        self.params: PluginParams = PluginParams()

        self.project.on_instance_import(self.compute_image_vectors_on_import)
        self.project.on_folder_delete(self.rebuild_trees)
        self.add_action_easy(self.find_images, ['similar'])
        self.add_action_easy(self.compute_clusters, ['group'])
        self.add_action_easy(self.cluster_by_tags, ['group'])
        self.add_action_easy(self.find_duplicates, ['group'])
        self.add_action_easy(self.search_by_text, ['execute'])
        self._comp_vec_desc = self.add_action_easy(self.compute_vectors, ['execute'])
        self._comp_all_vec_desc = self.add_action_easy(self.compute_all_vectors, ['execute'])

        self.trees: Dict[VectorType, FaissTree] = {}
        self._transformer = None

    @property
    def transformer(self):
        if self._transformer is None:
            self._transformer = self._load_transformer()
        return self._transformer

    async def start(self):
        await super().start()

        [await self._get_tree(t) for t in VectorType]

    def _get_vector_func_notifs(self, vec_type: VectorType):
        res = [
            NotifFunction(self._comp_vec_desc.id,
                          ActionContext(ui_inputs={"vec_type": vec_type.value}),
                          message=f"Compute all vectors of type {vec_type.value}"),
            NotifFunction(self._comp_all_vec_desc.id,
                          ActionContext(),
                          message="Compute vectors off all types")
        ]
        return res

    async def compute_vectors(self, context: ActionContext, vec_type: VectorType):
        """
        Compute image vectors of selected vector type
        """

        instances = await self.project.get_instances(ids=context.instance_ids)
        for i in instances:
            await self._compute_image_vector(i, vec_type)

        notif = Notif(type=NotifType.INFO,
                      name="ComputeVector",
                      message=f"Successfuly started compute of vectors of type {vec_type.value}")
        return ActionResult(notifs=[notif])

    async def compute_image_vectors_on_import(self, instance: Instance):
        await self._compute_image_vector(instance, VectorType.clip)
        if self.params.greyscale:
            await self._compute_image_vector(instance, VectorType.clip_grey)

    async def compute_all_vectors(self, context: ActionContext):
        """
        Compute image vectors of all supported types
        """
        res = [await self.compute_vectors(context, t) for t in VectorType]
        return ActionResult(notifs=[n for r in res for n in r.notifs])

    async def _compute_image_vector(self, instance: Instance, vector: VectorType):
        task = ComputeVectorTask(self, self.name, vector, instance, self.data_path)
        self.project.add_task(task)

    async def compute_clusters(self, context: ActionContext, vec_type: VectorType = VectorType.rgb,
                               nb_clusters: int = 10, label_clusters: bool = False):
        """
        Computes images clusters with Faiss Kmeans
        @nb_clusters: requested number of clusters
        """
        instances = await self.project.get_instances(context.instance_ids)
        sha1_to_instance = group_by_sha1(instances)
        sha1_to_ahash = {i.sha1: i.ahash for i in instances}
        sha1s = list(sha1_to_instance.keys())

        if not sha1s:
            empty_notif = Notif(NotifType.ERROR, name="NoData", message="No instance found")
            return ActionResult(notifs=[empty_notif])

        vectors = await self.project.get_vectors(source=self.name, vector_type=vec_type.value, sha1s=sha1s)

        if not vectors:
            empty_notif = Notif(NotifType.ERROR,
                                name="NoData",
                                message=f"""For the clustering function image vectors are needed.
                                        No such vectors ({vec_type.value}) could be found. 
                                        Compute the vectors and try again.) """,
                                functions=self._get_vector_func_notifs(vec_type))
            return ActionResult(notifs=[empty_notif])
        clusters, distances = make_clusters(vectors, method="kmeans", nb_clusters=nb_clusters)
        groups = []
        groups_images = []
        labels = []
        i = 0
        # TODO: put back mistral when it's working properly
        if label_clusters:
            from mistral_test import create_labels_from_group, generate_group_image
        for cluster, distance in zip(clusters, distances):
            group = Group(score=Score(min=0, max=100, max_is_best=False, value=distance))
            if label_clusters:
                images = [sha1_to_instance[sha1][0].url for sha1 in cluster[:20]]
                groups_images.append(generate_group_image(images, i))
                i += 1
            group.sha1s = sorted(cluster, key=lambda sha1: sha1_to_ahash[sha1])
            groups.append(group)
        if len(groups_images) > 0:
            labels = create_labels_from_group(groups_images)
        for i, g in enumerate(groups):
            g.name = f"Cluster {i}" if not len(labels) > 0 else "-".join(labels[i])

        return ActionResult(groups=groups)

    async def find_images(self, context: ActionContext, vec_type: VectorType = VectorType.rgb):
        """
        Find Similar images using Cosine distances.
        dist: 0 -> images are considered highly dissimilar
        dist: 1 -> images are considered identical
        See: https://en.wikipedia.org/wiki/Cosine_similarity for more.
        """
        instances = await self.project.get_instances(context.instance_ids)
        sha1s = [i.sha1 for i in instances]
        ignore_sha1s = set(sha1s)
        vectors = await self.project.get_vectors(source=self.name, vector_type=vec_type.value, sha1s=sha1s)

        if not vectors:
            return ActionResult(notifs=[Notif(
                NotifType.ERROR,
                name="NoData",
                message=f"""For the similarity function image vectors are needed.
                            No such vectors ({vec_type.value}) could be found. 
                            Compute the vectors and try again.) """,
                functions=self._get_vector_func_notifs(vec_type))])

        vector_datas = [x.data for x in vectors]

        tree = await self._get_tree(vec_type)
        if not tree:
            notif = Notif(type=NotifType.ERROR, name="NoFaissTree",
                          message=f"No Faiss tree could be loaded for vec_type {vec_type.value}")
            return ActionResult(notifs=[notif])

        res = tree.query(vector_datas)
        index = {r['sha1']: r['dist'] for r in res if r['sha1'] not in ignore_sha1s}

        res_sha1s = list(index.keys())
        res_scores = ScoreList(min=0, max=1, values=[index[sha1] for sha1 in res_sha1s],
                               max_is_best=True,
                               description="Similarity between 0 and 1. 1 is best")

        res = Group(sha1s=res_sha1s, scores=res_scores)
        return ActionResult(groups=[res])

    async def search_by_text(self, context: ActionContext, vec_type: VectorType = VectorType.rgb, text: str = '', min_similarity: float = 0.5):
        """Search image using text similarity"""
        if text == '':
            notif = Notif(type=NotifType.ERROR, name="EmptySearchText",
                          message="Please give a valid and not empty text search argument")
            return ActionResult(notifs=[notif])

        context_instances = await self.project.get_instances(context.instance_ids)
        context_sha1s = [i.sha1 for i in context_instances]

        tree = await self._get_tree(vec_type)
        if not tree:
            notif = Notif(type=NotifType.ERROR, name="NoFaissTree",
                          message=f"No Faiss tree could be loaded for vec_type {vec_type.value}")
            return ActionResult(notifs=[notif])

        try:
            resulting_images = tree.query_texts([text], self.transformer)
        except ValueError as e:
            return ActionResult(notifs=[Notif(type=NotifType.ERROR, name="TextSimilarityError", message=str(e))])


        # filter out images if they are not in the current context
        filtered_instances = [inst for inst in resulting_images if inst['sha1'] in context_sha1s]

        index = {r['sha1']: r['dist'] for r in filtered_instances}
        res_sha1s = np.asarray(list(index.keys()))
        res_scores = np.asarray([index[sha1] for sha1 in res_sha1s])

        # remap score since text to image similary tends to be between 0.1 and 0.4 and filter by similarity
        remaped_scores = np.around(np.interp(res_scores, [0, 0.375], [0, 1]), decimals=2)
        final_scores = remaped_scores[remaped_scores >= min_similarity].tolist()
        final_sha1s = res_sha1s[remaped_scores >= min_similarity].tolist()

        scores = ScoreList(min=0, max=1, values=final_scores, description="Similarity between image and text never give less than 0.1 and more than 0.4, hence here the values, remapped between 0 and 1")
        res = Group(sha1s=final_sha1s, scores=scores)
        res.name = "Text Search: " + text
        return ActionResult(groups=[res])

    async def cluster_by_tags(self, context: ActionContext, tags: PropertyId, vec_type: VectorType = VectorType.rgb):
        """Cluster images using a Tag/MultiTag property to guide the result"""
        props = await self.project.get_properties(ids=[tags])
        tag_prop = props[0]

        if tag_prop.type != PropertyType.tag and tag_prop.type != PropertyType.multi_tags:
            notif = Notif(type=NotifType.ERROR,
                          name="WrongPropertyType",
                          message=f"""Property: <{tag_prop.name}> is not of type Tag or MultiTags. This function only
                                  accepts tag types properties. Please choose another property""")
            return ActionResult(notifs=[notif])

        instances = await self.project.get_instances(context.instance_ids)
        sha1_to_instance = group_by_sha1(instances)
        sha1s = list(sha1_to_instance.keys())
        if not sha1s:
            return None
        # TODO: get tags text from the PropertyId
        tags_text = [t.value for t in await self.project.get_tags(property_ids=[tags])]
        text_vectors = get_text_vectors(tags_text, self.transformer)
        pano_vectors = await self.project.get_vectors(source=self.name, vector_type=vec_type.value, sha1s=sha1s)

        if not pano_vectors:
            return ActionResult(notifs=[Notif(
                NotifType.ERROR,
                name="NoData",
                message=f"""The Cluster_By_Tags function needs image vectors.
                            No such vectors ({vec_type.value}) could be found. 
                            Compute the vectors and try again.) """,
                functions=self._get_vector_func_notifs(vec_type))])

        vectors, sha1s = zip(*[(i.data, i.sha1) for i in pano_vectors])
        sha1s_array = np.asarray(sha1s)
        text_vectors_reshaped = np.squeeze(text_vectors, axis=1)

        images_vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        text_vectors_norm = text_vectors_reshaped / np.linalg.norm(text_vectors_reshaped, axis=1, keepdims=True)

        matrix = cosine_similarity(images_vectors_norm, text_vectors_norm)
        closest_text_indices = np.argmax(matrix, axis=1)
        similarities = np.max(matrix, axis=1)

        clusters = []
        distances = []

        for text_index in list(set(closest_text_indices)):
            cluster = sha1s_array[closest_text_indices == text_index]
            cluster_sim = similarities[closest_text_indices == text_index]
            distance = (1 - np.mean(cluster_sim)) * 100
            sorting_index = cluster_sim.argsort()
            sorted_cluster = cluster[sorting_index[::-1]]
            clusters.append(sorted_cluster)
            distances.append(distance)

        groups = []
        for cluster, distance in zip(clusters, distances):
            group = Group(score=distance)
            group.sha1s = list(cluster)
            groups.append(group)
        for i, g in enumerate(groups):
            g.name = f"Cluster {tags_text[i]}"

        return ActionResult(groups=groups)

    async def find_duplicates(self, context: ActionContext, min_similarity: float = 0.95):
        """
        Create clusters with at least `min_similarity` between the images of the cluster
        @min_similarity: the minimal similarity value between images of the cluster
        """
        # on récupère les vecteurs
        # pour chaque vecteur on récupère ses plus similaires (150 pour test) puis on filtre tout ce qui est < min_similarity
        # on marque tous les images dans le cluster pour ne pas les requêter à nouveau
        instances = await self.project.get_instances(context.instance_ids)
        sha1_to_instance = group_by_sha1(instances)
        sha1s = list(sha1_to_instance.keys())
        if not sha1s:
            return None
        # TODO: get tags text from the PropertyId
        pano_vectors = await self.project.get_vectors(source=self.name, vector_type='clip', sha1s=sha1s)
        vectors, sha1s = zip(*[(i.data, i.sha1) for i in pano_vectors])
        already_in_clusters = set()
        groups = []
        for vector, sha1 in zip(vectors, sha1s):
            if sha1 in already_in_clusters:
                continue
            tree = await self._get_tree(VectorType.rgb)
            res = tree.query([vector.data], 150)
            filtered = [r for r in res if r['dist'] >= min_similarity and r['sha1'] in sha1s]
            res_sha1s = [r['sha1'] for r in filtered]
            res_scores = [r['dist'] for r in filtered]
            score_list = ScoreList(min=0, max=1, max_is_best=True, values=res_scores)
            if len(res_sha1s) == 1:
                continue
            already_in_clusters.update(res_sha1s)
            groups.append(Group(sha1s=res_sha1s, scores=score_list))
        return ActionResult(groups=groups)

    async def rebuild_trees(self, deleted):
        for type_ in VectorType:
            await self._update_tree(type_)

    async def _get_tree(self, vec_type: VectorType):
        tree = self.trees.get(vec_type)
        if tree:
            return tree
        tree = load_faiss_tree(self, vec_type)
        if tree:
            self.trees[vec_type] = tree
            return tree
        tree = await create_faiss_tree(self, vec_type)
        if tree:
            self.trees[vec_type] = tree
            return tree

    async def _update_tree(self, vec_type: VectorType):
        tree = await create_faiss_tree(self, vec_type)
        self.trees[vec_type] = tree
        print(f"updated {vec_type.value} faiss tree")
        return tree

    def _load_transformer(self):
        return get_transformer(TransformerName[self.params.model])

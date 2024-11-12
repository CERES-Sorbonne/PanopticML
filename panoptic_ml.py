import os.path

from panoptic.core.plugin.plugin import APlugin
from panoptic.models import Instance, ActionContext
from panoptic.models.results import Group, ActionResult
from panoptic.utils import group_by_sha1
from panoptic.core.plugin.plugin_project_interface import PluginProjectInterface

from .compute.similarity import get_similar_images_from_text
from .compute import reload_tree, get_similar_images, make_clusters
from .compute_vector_task import ComputeVectorTask


class PanopticML(APlugin):
    """
    Default Machine Learning plugin for Panoptic
    Uses CLIP to generate vectors and FAISS for clustering / similarity functions
    """

    def __init__(self, project: PluginProjectInterface, plugin_path: str, name: str):
        super().__init__(name=name, project=project, plugin_path=plugin_path)
        reload_tree(self.data_path)

        self.project.on_instance_import(self.compute_image_vector)
        self.add_action_easy(self.find_images, ['similar'])
        self.add_action_easy(self.compute_clusters, ['group'])
        # self.add_action_easy(self.search_by_text, ['execute'])

    async def start(self):
        await super().start()
        vectors = await self.project.get_vectors(self.name, 'clip')

        # TODO: handle this properly with an import hook
        if not os.path.exists(os.path.join(self.data_path, 'tree_faiss.pkl')) and len(vectors) > 0:
            from .compute import compute_faiss_index
            await compute_faiss_index(self.data_path, self.project, self.name, 'clip')
            reload_tree(self.data_path)

    async def compute_image_vector(self, instance: Instance):
        task = ComputeVectorTask(self.project, self.name, 'clip', instance, self.data_path)
        self.project.add_task(task)

    async def compute_clusters(self, context: ActionContext, nb_clusters: int = 10):
        """
        Computes images clusters with Faiss Kmeans
        @nb_clusters: requested number of clusters
        """
        instances = await self.project.get_instances(context.instance_ids)
        sha1_to_instance = group_by_sha1(instances)
        sha1_to_ahash = {i.sha1: i.ahash for i in instances}
        sha1s = list(sha1_to_instance.keys())
        if not sha1s:
            return None

        vectors = await self.project.get_vectors(source=self.name, vector_type='clip', sha1s=sha1s)
        clusters, distances = make_clusters(vectors, method="kmeans", nb_clusters=nb_clusters)
        groups = []
        for cluster, distance in zip(clusters, distances):
            group = Group(score=distance)
            group.sha1s = sorted(cluster, key=lambda sha1: sha1_to_ahash[sha1])
            groups.append(group)
        for i, g in enumerate(groups):
            g.name = f"Cluster {i}"

        return ActionResult(groups=groups)

    async def find_images(self, context: ActionContext):
        instances = await self.project.get_instances(context.instance_ids)
        sha1s = [i.sha1 for i in instances]
        ignore_sha1s = set(sha1s)
        vectors = await self.project.get_vectors(source=self.name, vector_type='clip', sha1s=sha1s)
        vector_datas = [x.data for x in vectors]
        res = get_similar_images(vector_datas)
        index = {r['sha1']: r['dist'] for r in res if r['sha1'] not in ignore_sha1s}

        res_sha1s = list(index.keys())
        res_scores = [index[sha1] for sha1 in res_sha1s]

        res = Group(sha1s=res_sha1s, scores=res_scores)
        return ActionResult(instances=res)

    async def search_by_text(self, context: ActionContext, text: str):
        instances = get_similar_images_from_text(text)
        index = {r['sha1']: r['dist'] for r in instances}
        res_sha1s = list(index.keys())
        res_scores = [index[sha1] for sha1 in res_sha1s]
        res = Group(sha1s=res_sha1s, scores=res_scores)
        res.name = "Text Search: " + text
        return ActionResult(instances=res)

    # async def cluster_by_tags(self, context: ActionContext, tags: PropertyId):
    #     instances = await self.project.get_instances(context.instance_ids)
    #     sha1_to_instance = group_by_sha1(instances)
    #     sha1s = list(sha1_to_instance.keys())
    #     if not sha1s:
    #         return None
    #
    #     text_vectors = get_text_vectors(tags)
    #     pano_vectors = await self.project.get_vectors(source=self.name, vector_type='clip', sha1s=sha1s)
    #     vectors, sha1s = zip(*[(i.data, i.sha1) for i in pano_vectors])
    #     sha1s_array = np.asarray(sha1s)
    #     text_vectors_reshaped = np.squeeze(text_vectors, axis=1)
    #
    #     images_vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    #     text_vectors_norm = text_vectors_reshaped / np.linalg.norm(text_vectors_reshaped, axis=1, keepdims=True)
    #
    #     matrix = cosine_similarity(images_vectors_norm, text_vectors_norm)
    #     closest_text_indices = np.argmax(matrix, axis=1)
    #     closest_text_probs = np.max(matrix, axis=1)
    #
    #     clusters = []
    #     distances = []
    #
    #     for text_index in list(set(closest_text_indices)):
    #         cluster = sha1s_array[closest_text_indices == text_index]
    #         distance = 100 - np.mean(closest_text_probs[closest_text_indices == text_index]) * 100
    #         clusters.append(cluster)
    #         distances.append(distances)



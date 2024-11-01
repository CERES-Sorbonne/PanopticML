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

    def __init__(self, project: PluginProjectInterface, plugin_path: str):
        super().__init__(name='PanopticML', project=project, plugin_path=plugin_path)
        reload_tree(self.project.base_path)

        self.project.on_instance_import(self.compute_image_vector)
        self.add_action_easy(self.find_images, ['similar'])
        self.add_action_easy(self.compute_clusters, ['group'])
        # self.add_action_easy(self.search_by_text, ['execute'])

    async def start(self):
        await super().start()
        vectors = await self.project.get_vectors(self.name, 'clip')

        # TODO: handle this properly with an import hook
        if not os.path.exists(os.path.join(self.project.base_path, 'tree_faiss.pkl')) and len(vectors) > 0:
            from .compute import compute_faiss_index
            await compute_faiss_index(self.project.base_path, self.project, self.name, 'clip')
            reload_tree(self.project.base_path)

    async def compute_image_vector(self, instance: Instance):
        task = ComputeVectorTask(self.project, self.name, 'clip', instance)
        self.project.add_task(task)

    async def compute_clusters(self, context: ActionContext, nb_clusters: int = 10):
        """
        Computes images clusters with Faiss Kmeans
        @nb_clusters: requested number of clusters
        """
        instances = await self.project.get_instances(context.instance_ids)
        sha1_to_instance = group_by_sha1(instances)
        sha1s = list(sha1_to_instance.keys())
        if not sha1s:
            return None

        vectors = await self.project.get_vectors(source=self.name, vector_type='clip', sha1s=sha1s)
        clusters, distances = make_clusters(vectors, method="kmeans", nb_clusters=nb_clusters)
        groups = []
        # TODO: simplify once plugins can return sha1 instead of ids
        for cluster, distance in zip(clusters, distances):
            group = Group(score=distance)
            instances = []
            for sha1 in cluster:
                instances.extend(sha1_to_instance[sha1])
            # sort instances inside cluster by average_hash
            group.ids = [i.id for i in sorted(instances, key=lambda inst: inst.ahash)]
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
        return ActionResult(groups=[res])
import os
import pickle

import faiss
import numpy as np

from panoptic.models import VectorType
from panoptic.core.plugin.plugin import APlugin
from .transformer import Transformer


class FaissTree:
    def __init__(self, index: faiss.IndexFlatIP, labels: list[str]):
        self.index = index
        self.labels = labels

    def query(self, vectors: list[np.ndarray], k=999999):
        vector_center = np.mean(vectors, axis=0)

        norm = np.linalg.norm(vector_center)
        if norm > 0:
            vector_center = vector_center / norm

        vector = np.asarray([vector_center])

        real_k = min(k, len(self.labels))
        vector = vector.reshape(1, -1)
        dist, ind = self.index.search(vector, real_k)
        indices = [x for x in ind[0]]
        distances = [x for x in dist[0]]  # avoid some strange overflow behavior
        return [{'sha1': self.labels[i], 'dist': float('%.2f' % (distances[index]))} for index, i in
                enumerate(indices)]

    def query_texts(self, texts: list[str], transformer: Transformer):
        text_vectors = transformer.get_text_vectors(texts)
        return self.query(text_vectors)


def gen_tree_file_name(type_id: int):
    return f"faiss_tree_vec_id_{type_id}.pkl"


async def create_faiss_tree(plugin: APlugin, type_id: int):
    project = plugin.project
    name = gen_tree_file_name(type_id)
    vectors = await project.get_vectors(type_id)

    if vectors is None or len(vectors) == 0:
        return
    vec_data, sha1_list = zip(*[(i.data, i.sha1) for i in vectors])
    vec_np_arr = np.asarray(vec_data)
    faiss.normalize_L2(vec_np_arr)

    # create the faiss index based on this post: https://anttihavanko.medium.com/building-image-search-with-openai-clip-5a1deaa7a6e2
    vector_size = vec_np_arr.shape[1]
    index = faiss.IndexFlatIP(vector_size)
    # faiss.ParameterSpace().set_index_parameter(index, 'nprobe', 100)
    index.add(np.asarray(vec_np_arr))

    tree = FaissTree(index, sha1_list)

    with open(os.path.join(plugin.data_path, name), 'wb') as f:
        pickle.dump(tree, f)

    return FaissTree(index=index, labels=sha1_list)


def load_faiss_tree(plugin: APlugin, vec_type: int) -> FaissTree | None:
    name = gen_tree_file_name(vec_type)
    path = os.path.join(plugin.data_path, name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except ModuleNotFoundError:
        return None


class FaissTreeManager:
    def __init__(self, plugin: APlugin):
        self.trees: dict[int, FaissTree] = {}
        self.plugin = plugin

    async def get(self, vec_type: VectorType):
        type_id = vec_type.id

        if self.trees.get(type_id):
            return self.trees[type_id]

        tree = load_faiss_tree(self.plugin, type_id)
        if tree:
            self.trees[type_id] = tree
            return tree
        tree = await create_faiss_tree(self.plugin, type_id)
        if tree:
            self.trees[type_id] = tree
            return tree

    async def rebuild_tree(self, vec_type: VectorType):
        type_id = vec_type.id
        tree = await create_faiss_tree(self.plugin, type_id)
        self.trees[type_id] = tree
        print(f"updated vec [{type_id}] faiss tree")
        return tree

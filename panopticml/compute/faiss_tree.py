import os
import pickle

import faiss
import numpy as np

from panoptic.core.databases.media.models import VectorType
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
        vector = np.asarray([vector_center]).reshape(1, -1)
        real_k = min(k, len(self.labels))
        dist, ind = self.index.search(vector, real_k)
        return [{'sha1': self.labels[i], 'dist': float('%.2f' % dist[0][idx])}
                for idx, i in enumerate(ind[0])]

    def query_texts(self, texts: list[str], transformer: Transformer, return_vec=False):
        text_vectors = transformer.get_text_vectors(texts)
        images = self.query(text_vectors)
        if return_vec:
            return images, text_vectors
        return images


def _tree_file_name(type_id: int) -> str:
    return f"faiss_tree_vec_id_{type_id}.pkl"


def create_faiss_tree(plugin, type_id: int):
    vectors = plugin.project.get_vectors(type_id)
    if not vectors:
        return None

    vec_data, sha1_list = zip(*[(v.data, v.sha1) for v in vectors])
    vec_np = np.asarray(vec_data)
    faiss.normalize_L2(vec_np)

    index = faiss.IndexFlatIP(vec_np.shape[1])
    index.add(vec_np)
    tree = FaissTree(index, sha1_list)

    with open(os.path.join(plugin.data_path, _tree_file_name(type_id)), 'wb') as f:
        pickle.dump(tree, f)

    return tree


def load_faiss_tree(plugin, type_id: int) -> FaissTree | None:
    path = os.path.join(plugin.data_path, _tree_file_name(type_id))
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except (ModuleNotFoundError, Exception):
        return None


class FaissTreeManager:
    def __init__(self, plugin):
        self.trees: dict[int, FaissTree] = {}
        self.plugin = plugin

    def get(self, vec_type: VectorType) -> FaissTree | None:
        type_id = vec_type.id
        if self.trees.get(type_id):
            return self.trees[type_id]
        tree = load_faiss_tree(self.plugin, type_id)
        if tree:
            self.trees[type_id] = tree
            return tree
        tree = create_faiss_tree(self.plugin, type_id)
        if tree:
            self.trees[type_id] = tree
        return tree

    def rebuild_tree(self, vec_type: VectorType) -> FaissTree | None:
        type_id = vec_type.id
        tree = create_faiss_tree(self.plugin, type_id)
        self.trees[type_id] = tree
        print(f"updated vec [{type_id}] faiss tree")
        return tree

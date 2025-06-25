import os
import pickle

import faiss
import numpy as np

from compute.similarity import get_text_vectors
from models import VectorType
from panoptic.core.plugin.plugin import APlugin


class FaissTree:
    def __init__(self, index: faiss.IndexFlatIP, labels: list[str]):
        self.index = index
        self.labels = labels

    def query(self, vectors: list[np.ndarray], k=999999):
        vector_center = np.mean(vectors, axis=0)
        vector = np.asarray([vector_center])

        faiss.normalize_L2(vector)
        real_k = min(k, len(self.labels))
        vector = vector.reshape(1, -1)
        dist, ind = self.index.search(vector, real_k)
        indices = [x for x in ind[0]]
        distances = [x for x in dist[0]]  # avoid some strange overflow behavior
        return [{'sha1': self.labels[i], 'dist': float('%.2f' % (distances[index]))} for index, i in
                enumerate(indices)]

    def query_texts(self, texts: list[str], transformer):
        text_vectors = get_text_vectors(texts, transformer)
        return self.query(text_vectors[0])


def gen_tree_file_name(vec_type: VectorType):
    return f"{vec_type.value}_faiss_tree.pkl"


async def create_faiss_tree(plugin: APlugin, vec_type: VectorType):
    project = plugin.project
    name = gen_tree_file_name(vec_type)
    vectors = await project.get_vectors(plugin.name, vector_type=vec_type.value)

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


def load_faiss_tree(plugin: APlugin, vec_type: VectorType) -> FaissTree | None:
    name = gen_tree_file_name(vec_type)
    path = os.path.join(plugin.data_path, name)
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

import faiss
import numpy as np
from sklearn.cluster import HDBSCAN
import torch

from panoptic.models import Vector
from panoptic.models.results import ScoreList, Score, Group
from ..utils import similarity_matrix


def make_clusters(vectors: list[Vector], **kwargs) -> (list[list[str]], list[int]):
    res_clusters = []
    res_distances = []
    vectors, sha1 = zip(*[(i.data, i.sha1) for i in vectors])
    sha1 = np.asarray(sha1)
    clusters: np.ndarray

    clusters, distances = _make_clusters_faiss(vectors, **kwargs)

    for cluster in list(set(clusters)):
        sha1_cluster = sha1[clusters == cluster]
        current_cluster_distances = distances[clusters == cluster]
        if distances is not None:
            res_distances.append(np.mean(current_cluster_distances))
        res_clusters.append(list(sha1_cluster))
    # sort clusters by distances
    sorted_clusters = [cluster for _, cluster in sorted(zip(res_distances, res_clusters))]
    res_distances_scaled = [i * 100 for i in res_distances]
    return sorted_clusters, sorted(res_distances_scaled)


def _make_clusters_faiss(vectors, nb_clusters=6, **kwargs) -> (np.ndarray, np.ndarray):
    def _make_single_kmean(vectors, nb_clusters):
        kmean = faiss.Kmeans(vectors.shape[1], nb_clusters, niter=20, verbose=False)
        kmean.train(vectors)
        return kmean.index.search(vectors, 1)

    vectors = np.asarray(vectors)
    if nb_clusters == -1:
        clusterer = HDBSCAN(min_cluster_size=5)
        clusterer.fit(vectors)
        indices = clusterer.labels_
        probabilities = clusterer.probabilities_
        distances = np.zeros_like(probabilities, dtype=np.float32)
        unique_clusters = np.unique(indices)
        # compute distances just like the one returned by kmeans to have consistent metrics
        for cluster_id in unique_clusters:
            if cluster_id == -1:
                distances[indices == -1] = 100.0
                continue
            cluster_mask = (indices == cluster_id)
            cluster_vectors = vectors[cluster_mask]
            cluster_probabilities = probabilities[cluster_mask]
            center_local_index = np.argmax(cluster_probabilities)
            center_vector = cluster_vectors[center_local_index].reshape(1, -1)
            dists = faiss.pairwise_distances(center_vector, cluster_vectors)[0]
            distances[cluster_mask] = dists
    else:
        distances, indices = _make_single_kmean(vectors, nb_clusters)
    return indices.flatten(), distances.flatten()


def cluster_by_text(image_vectors: list[Vector], text_vectors: list[np.array], text_labels: list[str],
                    min_similarity: float, max_text_sim: float, multiple: bool) -> list[Group]:
    vectors, sha1s = zip(*[(i.data, i.sha1) for i in image_vectors])
    sha1s_array = np.asarray(sha1s)

    if not multiple:
        similarities, closest_text_indices = similarity_matrix(vectors, text_vectors, multiple=False)
        similarities = torch.from_numpy(
            np.around(np.interp(similarities.numpy(), [0, max_text_sim], [0, 1]), decimals=2)
        )
        mask = similarities >= min_similarity
        filtered_similarities = similarities[mask]
        filtered_sha1s = sha1s_array[mask]
        filtered_indices = closest_text_indices[mask]

        clusters = []
        distances = []
        clusters_text = []
        cluster_sims = []

        for text_index in filtered_indices.unique():
            clusters_text.append(text_labels[text_index])
            cluster = filtered_sha1s[filtered_indices == text_index]

            cluster_sim = filtered_similarities[filtered_indices == text_index]
            sorted_sim = cluster_sim.sort(descending=True).values
            cluster_sims.append([round(x, 2) for x in sorted_sim.tolist()])

            sorting_index = cluster_sim.argsort(descending=True)
            sorted_cluster = cluster[sorting_index]
            if type(sorted_cluster) is not np.ndarray:
                sorted_cluster = np.array([sorted_cluster])
            clusters.append(sorted_cluster)

            distance = float((1 - torch.mean(cluster_sim)) * 100)
            distances.append(distance)

        groups = []
        for cluster, distance, name, cluster_sim in zip(clusters, distances, clusters_text, cluster_sims):
            similarities = ScoreList(min=0, max=1, max_is_best=True, values=cluster_sim)
            group = Group(score=Score(distance, min=0, max=200, max_is_best=False,
                                      description="Mean distance between the images of this cluster and the queried text"),
                          scores=similarities)
            group.sha1s = list(cluster)
            group.name = f"Cluster {name}"
            groups.append(group)

        return groups

    else:
        similarity_matrix_full = similarity_matrix(vectors, text_vectors, multiple=True)
        similarity_matrix_remapped = torch.from_numpy(
            np.around(np.interp(similarity_matrix_full.numpy(), [0, max_text_sim], [0, 1]), decimals=2)
        )

        groups = []
        for text_idx, text_label in enumerate(text_labels):
            text_similarities = similarity_matrix_remapped[:, text_idx]
            mask = text_similarities >= min_similarity

            cluster_sha1s = sha1s_array[mask]
            cluster_sims = text_similarities[mask]

            if len(cluster_sha1s) > 0:
                sorted_sim = cluster_sims.sort(descending=True).values
                sorting_index = cluster_sims.argsort(descending=True)
                sorted_cluster = np.atleast_1d(cluster_sha1s[sorting_index])

                distance = float((1 - torch.mean(cluster_sims)) * 100)

                similarities = ScoreList(min=0, max=1, max_is_best=True,
                                         values=[round(x, 2) for x in sorted_sim.tolist()])
                group = Group(score=Score(distance, min=0, max=200, max_is_best=False,
                                          description="Mean distance between the images of this cluster and the queried text"),
                              scores=similarities)
                group.sha1s = list(sorted_cluster)
                group.name = f"Cluster {text_label}"
                groups.append(group)

        return groups


def custom_range(min_i, max_i, steps, increments):
    """
    Generate a range of values from min_i to max_i with a variable increment for each step
    :param min_i: first value
    :param max_i: last value
    :param steps: values for which increment should change
    :param increments: increments values
    :return:
    """
    i = min_i
    current_step = 0
    current_incr = 1
    while i < max_i:
        yield i
        if i >= steps[current_step] and current_step < len(steps) - 1:
            current_incr = increments[current_step]
            current_step += 1
        i += current_incr

import io
from enum import Enum

import numpy as np
import torch
import re
from PIL import Image

from panoptic.models import Tag


def preprocess_image(image_data: bytes, params: dict):
    image = Image.open(io.BytesIO(image_data))
    if params.get('greyscale'):
        image = image.convert('L').convert('RGB')
    else:
        image = image.convert('RGB')
    return image

def cosine_similarity(embedding1: np.array, embedding2: np.array):
    """
    Calcule la similarité cosinus entre deux embeddings normalisés

    Args:
        embedding1: premier embedding (normalisé)
        embedding2: deuxième embedding (normalisé)

    Returns:
        float: similarité cosinus (entre -1 et 1)
    """
    # Pour des embeddings normalisés, similarité cosinus = produit scalaire
    embedding1_torch = torch.from_numpy(embedding1).squeeze()
    embedding2_torch = torch.from_numpy(embedding2).squeeze()

    return torch.dot(embedding1_torch, embedding2_torch).item()


def similarity_matrix(vectors1: list[np.array], vectors2: list[np.array], multiple = False) \
        -> tuple[torch.Tensor, torch.Tensor] | tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Create a similarity matrix from two list of normalized np vectors
    Returns two tensors, one for best scores and one for best_indices of vectors1 into vectors2
    i.e for each vector1 we'll know the closest vector of vectors2 and the similarity between them
    remap_similarity: float : not all models return the same scale of text / image similarity so we remap it with this parameter
    multiple: if true will return all the indices that are above the threshold
    threshold: float : if defined will only return indices above it
    """
    astorch1 = torch.from_numpy(np.vstack(vectors1))
    astorch2 = torch.from_numpy(np.vstack(vectors2))
    matrix = torch.mm(astorch1, astorch2.T)

    if not multiple:
        best_scores, best_indices = torch.max(matrix, dim=1)
        return best_scores, best_indices
    else:
        return matrix
        # scores_list = []
        # indices_list = []
        #
        # for i in range(matrix.shape[0]):
        #     row = matrix[i]
        #
        #     remapped_scores = torch.from_numpy(
        #         np.around(np.interp(row.numpy(), [0, remap_similarity], [0, 1]), decimals=2)
        #     )
        #     # Apply mask BEFORE remapping
        #     mask = remapped_scores >= threshold
        #     matching_scores = remapped_scores[mask]
        #     matching_indices = torch.where(mask)[0]
        #
        #     scores_list.append(matching_scores)
        #     indices_list.append(matching_indices)
        #
        # return scores_list, indices_list

def resolve_device():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    # TODO: when silicon bugs are working again put it back
    # elif torch.backends.mps.is_available():
    #     device = 'mps'
    return device

def is_image_url(url):
    pattern = re.compile(
        r'''(?xi)
        \b
        https?://                       # protocole http ou https
        [\w.-]+(?:\.[\w.-]+)+           # domaine
        (?:/[^\s?#]*)*                  # chemin éventuel
        /?                              # éventuellement un / final
        [^\s?#]*                        # éventuellement un nom de fichier
        \.(?:jpg|jpeg|png|gif|webp|svg|bmp|tiff|ico)  # extension image
        (?:\?[^\s#]*)?                  # paramètres après ?
        \b
        '''
    )
    return re.match(pattern, url)


class ClusterByTagsEnum(Enum):
    use = "use"
    combine = "combine"
    ignore = "ignore"

def process_tags(tags: list[Tag], parent_tags: ClusterByTagsEnum):
    match parent_tags:
        case ClusterByTagsEnum.use:
            return [t.value for t in tags]
        case ClusterByTagsEnum.combine:
            tree = build_tag_hierarchy(tags)
            return [tree[t.id]['full_name'] for t in tags if tree[t.id]['is_leaf']]
        case ClusterByTagsEnum.ignore:
            tree = build_tag_hierarchy(tags)
            return [t.value for t in tags if tree[t.id]['is_leaf']]


def build_tag_hierarchy(tags):
    """
    Make a tag tree to find if a tag is a leaf, and if so the name taking account of it's ancestors
    Returns:
        dict: {tag_id: {'full_name': str, 'is_leaf': bool, 'name': str}}
    """
    tag_map = {tag.id: tag for tag in tags}

    tags_with_children = set()
    for tag in tags:
        if tag.parents:
            for parent_id in tag.parents:
                tags_with_children.add(parent_id)

    def get_all_ancestors_ordered(tag_id, visited=None):
        if visited is None:
            visited = set()

        if tag_id in visited:
            return []

        visited.add(tag_id)
        tag = tag_map.get(tag_id)

        if not tag or not tag.parents:
            return []

        all_ancestors = []
        for parent_id in tag.parents:
            parent_ancestors = get_all_ancestors_ordered(parent_id, visited.copy())
            all_ancestors.extend(parent_ancestors)
            if parent_id not in all_ancestors:
                all_ancestors.append(parent_id)

        seen = set()
        unique_ancestors = []
        for ancestor_id in all_ancestors:
            if ancestor_id not in seen:
                seen.add(ancestor_id)
                unique_ancestors.append(ancestor_id)

        return unique_ancestors

    hierarchy = {}

    for tag in tags:
        ancestors = get_all_ancestors_ordered(tag.id)

        # Build full name of tag with ancestors name
        ancestor_values = [tag_map[ancestor_id].value for ancestor_id in ancestors]
        full_name_parts = ancestor_values + [tag.value]
        full_name = " ".join(full_name_parts)
        is_leaf = tag.id not in tags_with_children

        hierarchy[tag.id] = {
            'full_name': full_name,
            'is_leaf': is_leaf,
            'name': tag.value
        }

    return hierarchy
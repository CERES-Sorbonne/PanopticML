import io
import numpy as np
import torch
import re
from PIL import Image


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


def similarity_matrix(vectors1: list[np.array], vectors2: list[np.array]) -> (torch.Tensor, torch.Tensor):
    """
    Create a similarity matrix from two list of normalized np vectors
    Returns two tensors, one for best scores and one for best_indices of vectors1 into vectors2
    i.e for each vector1 we'll know the closest vector of vectors2 and the similarity between them
    """
    astorch1 = torch.from_numpy(np.vstack(vectors1))
    astorch2 = torch.from_numpy(np.vstack(vectors2))
    matrix = torch.mm(astorch1, astorch2.T)
    best_scores, best_indices = torch.max(matrix, dim=1)
    return best_scores, best_indices

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
import os
import pathlib
from itertools import product

import faiss
import pytest
import numpy as np

from panopticml.compute.faiss_tree import FaissTree
from panopticml.compute.transformer import get_transformer, Transformer
from panopticml.panoptic_ml import ModelEnum
from panopticml.utils import preprocess_image, cosine_similarity

transformers_to_test = [transformer.value for transformer in ModelEnum]

def create_faiss_tree(vectors, images):
    vectors = np.asarray(vectors)
    faiss.normalize_L2(vectors)

    vector_size = vectors.shape[1]
    index = faiss.IndexFlatIP(vector_size)
    index.add(np.asarray(vectors))

    tree = FaissTree(index, images)
    return tree

def get_images():
    res_dir = pathlib.Path(__file__).parent / 'resources'
    return [f for f in res_dir.iterdir() if f.suffix in ['.jpg', '.jpeg', '.png', '.gif'] and f.name != 'cropped_chat.png']


def generate_vectors(transformer: Transformer, images=None):
    vectors = []
    images = get_images() if not images else images
    for img_path in images:
        with open(img_path, mode='rb') as f:
            image_data = preprocess_image(f.read(), {'greyscale': False})
        vectors.append(transformer.to_vector(image_data))
    return vectors, images

@pytest.fixture(scope='session')
def all_models():
    models = {}
    for model_name in transformers_to_test:
        print('preloading ' + model_name.name)
        models[model_name] = get_transformer(model_name)
    return models

@pytest.mark.parametrize("model_name, vector_type", list(product(transformers_to_test, [{'greyscale': False}, {'greyscale': True}])))
def test_image_to_vector(model_name, vector_type, all_models):
    """Test tous les transformers disponibles"""
    for img_path in get_images():
        with open(img_path, mode='rb') as f:
            image_data = f.read()
        test_image = preprocess_image(image_data, vector_type)

        print(f"\n=== Testing {model_name.value.upper()} with image {img_path} ===")

        transformer = all_models[model_name]
        print(f"Transformer {model_name} initialisé avec succès")

        # Tester la conversion d'image en vecteur
        print("Testing image to vector conversion...")
        image_vector = transformer.to_vector(test_image)

        # Vérifications
        assert isinstance(image_vector, np.ndarray), f"Le résultat doit être un numpy array pour {model_name}"
        assert image_vector.size > 0, f"Le vecteur ne doit pas être vide pour {model_name}"
        print(f"Image convertie en vecteur de taille: {image_vector.shape}")



@pytest.mark.parametrize("model_name", transformers_to_test)
def test_text_to_vector(model_name, all_models):
    transformer = all_models[model_name]
    test_text = "This is some random text depicting an image"

    if transformer.can_handle_text:
        print("Testing text to vector conversion...")
        text_vector = transformer.to_text_vector(test_text)

        # Vérifications
        assert isinstance(text_vector,
                          np.ndarray), f"Le résultat texte doit être un numpy array pour {model_name}"
        assert text_vector.size > 0, f"Le vecteur texte ne doit pas être vide pour {model_name}"
        print(f"✓ Texte converti en vecteur de taille: {text_vector.shape}")
    else:
        print("✗ Ce transformer ne supporte pas la conversion de texte")


@pytest.fixture(scope='module')
def mobileclip_transformer():
    """Load only the MobileCLIP transformer (avoids preloading every model)."""
    return get_transformer(ModelEnum.mobileclip_s2.value)


def test_mobileclip_vectors_are_meaningful(mobileclip_transformer):
    """
    Sanity check targeting the 'clusters make no sense' symptom: MobileCLIP image
    vectors must be L2-normalized, of consistent dimension, NOT collapsed (distinct
    images must not be near-identical), and discriminative enough that a cropped cat
    retrieves the cat.
    """
    transformer = mobileclip_transformer
    image_vectors, images = generate_vectors(transformer)

    dims = {v.shape for v in image_vectors}
    assert len(dims) == 1, f"Inconsistent vector dimensions across images: {dims}"

    for v, img in zip(image_vectors, images):
        norm = float(np.linalg.norm(v))
        assert np.isclose(norm, 1.0, atol=1e-3), f"Vector for {img.name} not L2-normalized (norm={norm:.4f})"

    # If embeddings are collapsed, every pair is ~identical and clustering is garbage.
    sims = []
    for i in range(len(image_vectors)):
        for j in range(i + 1, len(image_vectors)):
            sims.append(float(cosine_similarity(image_vectors[i], image_vectors[j])))
    print("MobileCLIP pairwise image cosine sims:", [round(s, 3) for s in sims])
    assert max(sims) < 0.99, f"Distinct images are near-identical (collapsed embeddings): max sim={max(sims):.4f}"

    # Image-image retrieval: a cropped cat must match the full cat image.
    tree = create_faiss_tree(image_vectors, images)
    test_image = pathlib.Path(__file__).parent / 'resources' / 'cropped_chat.png'
    test_vectors, _ = generate_vectors(transformer, [test_image])
    best_result = os.path.basename(tree.query([test_vectors[0]])[0]['sha1'])
    assert best_result == "chat.png", f"cropped_chat should retrieve chat.png, got {best_result}"


def test_mobileclip_text_image_similarity(mobileclip_transformer):
    """MobileCLIP text->image retrieval must point each prompt at the right image."""
    transformer = mobileclip_transformer
    assert transformer.can_handle_text
    texts = ['A jumping spider', 'A bird', 'A happy dog', 'A small grey cat']
    expected = ['spider.jpg', 'bird.gif', 'dog.jpg', 'chat.png']
    image_vectors, images = generate_vectors(transformer)

    for text, expected_image in zip(texts, expected):
        text_vector = transformer.to_text_vector(text)
        sims = sorted(
            ((float(cosine_similarity(text_vector, iv)), img.name) for iv, img in zip(image_vectors, images)),
            reverse=True,
        )
        print(f"\nText '{text}' -> {sims[:3]}")
        assert sims[0][1] == expected_image, f"'{text}': expected {expected_image}, got {sims[0][1]}"


@pytest.mark.parametrize("model_name", transformers_to_test)
def test_index_creation(model_name, all_models):
    transformer = all_models[model_name]
    vectors, images = generate_vectors(transformer)
    create_faiss_tree(vectors, images)

@pytest.mark.parametrize("model_name", transformers_to_test)
def test_image_image_similarity(model_name, all_models):
    """
    This test shoud check if an image is similar to itself when querying the faiss index
    """
    transformer = all_models[model_name]
    image_vectors, images = generate_vectors(transformer)
    tree = create_faiss_tree(image_vectors, images)
    test_image = pathlib.Path(__file__).parent / 'resources' / 'cropped_chat.png'
    test_vectors, _ = generate_vectors(transformer, [test_image])
    result_images = tree.query([test_vectors[0]])
    best_result = os.path.basename(result_images[0]['sha1'])
    assert best_result == "chat.png"

@pytest.mark.parametrize("model_name", transformers_to_test)
def test_text_image_similarity(model_name, all_models):
    transformer = all_models[model_name]
    texts = ['A jumping spider', 'A bird', 'A happy dog', 'An arachnoid robot', 'A small grey cat']
    expected_results = ['spider.jpg', 'bird.gif', 'dog.jpg', 'spider.jpg', 'chat.png']
    image_vectors, images = generate_vectors(transformer)
    if not transformer.can_handle_text:
        return
    texts_vectors = [transformer.to_text_vector(text) for text in texts]

    # Pour chaque requête texte
    for i, (text, text_vector, expected_image) in enumerate(zip(texts, texts_vectors, expected_results)):
        similarities = []

        # Calculer la similarité avec chaque image
        for j, (image_vector, image_name) in enumerate(zip(image_vectors, images)):
            # Assurez-vous que les vecteurs ont les bonnes dimensions
            text_vec = text_vector.flatten() if text_vector.ndim > 1 else text_vector
            img_vec = image_vector.flatten() if image_vector.ndim > 1 else image_vector

            # Calcul de la similarité cosinus
            cosine_sim = cosine_similarity(text_vec, img_vec)
            similarities.append((cosine_sim, image_name, j))

        # Trouver l'image avec la plus haute similarité
        similarities.sort(key=lambda x: x[0], reverse=True)
        best_match_name = similarities[0][1]
        best_similarity = similarities[0][0]

        # Debug info
        print(f"\nTexte: '{text}'")
        print(f"Attendu: {expected_image}")
        print(f"Trouvé: {best_match_name}")
        print(f"Similarité: {best_similarity:.4f}")
        print("Top 3 similarités:")
        for sim, name, idx in similarities[:3]:
            print(f"  {name}: {sim:.4f}")

        # Vérification
        assert best_match_name.name == expected_image, (
            f"Pour le texte '{text}', attendu '{expected_image}' "
            f"mais trouvé '{best_match_name}' (similarité: {best_similarity:.4f})"
        )



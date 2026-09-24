import gc
import os
import pathlib

import pytest
import numpy as np
import torch  # noqa: F401  before faiss, see panopticml/compute/__init__.py
import faiss

from panopticml.compute.faiss_tree import FaissTree
from panopticml.compute.transformer import get_transformer, Transformer
from panopticml.compute_vector_task import _preprocess_worker
from panopticml.panoptic_ml import ModelEnum
from panopticml.utils import preprocess_image, cosine_similarity


def _selected_models() -> list[str]:
    """Models to test, from PANOPTICML_TEST_MODELS: comma-separated ModelEnum names
    (e.g. "clip,siglip,dinov3"). Unset, empty or "all": every model."""
    selection = os.environ.get('PANOPTICML_TEST_MODELS', '').strip()
    if not selection or selection == 'all':
        return [model.value for model in ModelEnum]
    names = [name.strip() for name in selection.split(',') if name.strip()]
    unknown = [name for name in names if name not in ModelEnum.__members__]
    if unknown:
        raise ValueError(f"PANOPTICML_TEST_MODELS: unknown models {unknown}, "
                         f"expected among {list(ModelEnum.__members__)}")
    return [ModelEnum[name].value for name in names]


transformers_to_test = _selected_models()

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


def generate_fast_vectors(transformer: Transformer, images=None):
    """Vectors through the production path: ComputeVectorsTask decodes + resizes with
    _preprocess_worker, then embeds with forward_from_arrays (no HF processor)."""
    images = get_images() if not images else images
    arrays = []
    for img_path in images:
        _, array = _preprocess_worker((img_path.name, img_path.read_bytes(), transformer.preprocess_size, False))
        arrays.append(array)
    return list(transformer.forward_from_arrays(arrays)), images

@pytest.fixture(scope='session', params=transformers_to_test)
def model_name(request):
    return request.param


@pytest.fixture(scope='session')
def transformer(model_name):
    """One model loaded at a time: pytest groups the tests by model and drops the previous
    one before loading the next (all of them together don't fit in a CI runner's RAM)."""
    yield get_transformer(model_name)
    gc.collect()


@pytest.mark.parametrize("vector_type", [{'greyscale': False}, {'greyscale': True}])
def test_image_to_vector(model_name, vector_type, transformer):
    """Test tous les transformers disponibles"""
    for img_path in get_images():
        with open(img_path, mode='rb') as f:
            image_data = f.read()
        test_image = preprocess_image(image_data, vector_type)

        print(f"\n=== Testing {model_name.upper()} with image {img_path} ===")


        # Tester la conversion d'image en vecteur
        print("Testing image to vector conversion...")
        image_vector = transformer.to_vector(test_image)

        # Vérifications
        assert isinstance(image_vector, np.ndarray), f"Le résultat doit être un numpy array pour {model_name}"
        assert image_vector.size > 0, f"Le vecteur ne doit pas être vide pour {model_name}"
        assert np.isfinite(image_vector).all(), f"Le vecteur contient des NaN/inf pour {model_name}"
        print(f"Image convertie en vecteur de taille: {image_vector.shape}")



def test_text_to_vector(model_name, transformer):
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
    if ModelEnum.mobileclip_s2.value not in transformers_to_test:
        pytest.skip("mobileclip_s2 not in PANOPTICML_TEST_MODELS")
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


def test_index_creation(model_name, transformer):
    vectors, images = generate_vectors(transformer)
    create_faiss_tree(vectors, images)

def test_image_image_similarity(model_name, transformer):
    """
    This test shoud check if an image is similar to itself when querying the faiss index
    """
    image_vectors, images = generate_vectors(transformer)
    tree = create_faiss_tree(image_vectors, images)
    test_image = pathlib.Path(__file__).parent / 'resources' / 'cropped_chat.png'
    test_vectors, _ = generate_vectors(transformer, [test_image])
    result_images = tree.query([test_vectors[0]])
    best_result = os.path.basename(result_images[0]['sha1'])
    assert best_result == "chat.png"

def test_text_image_similarity(model_name, transformer):
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


def test_fast_path_matches_processor(model_name, transformer):
    """
    forward_from_arrays is what actually computes the stored vectors: it must give the same
    dimension as the processor path (text / image queries use that one) and close vectors.
    """
    slow_vectors, images = generate_vectors(transformer)
    fast_vectors, _ = generate_fast_vectors(transformer, images)

    for slow, fast, img in zip(slow_vectors, fast_vectors, images):
        slow, fast = slow.flatten(), fast.flatten()
        assert np.isfinite(fast).all(), f"Fast path vector for {img.name} contains NaN/inf"
        assert fast.shape == slow.shape, f"{img.name}: fast path {fast.shape} vs processor {slow.shape}"
        sim = float(np.dot(slow, fast) / (np.linalg.norm(slow) * np.linalg.norm(fast)))
        # the fast path squashes to a square instead of resize + center crop
        assert sim > 0.75, f"{img.name}: fast path vector too far from processor vector (cos={sim:.3f})"


def test_fast_path_image_image_similarity(model_name, transformer):
    image_vectors, images = generate_fast_vectors(transformer)
    tree = create_faiss_tree(image_vectors, images)
    test_image = pathlib.Path(__file__).parent / 'resources' / 'cropped_chat.png'
    test_vectors, _ = generate_vectors(transformer, [test_image])
    best_result = os.path.basename(tree.query([test_vectors[0]])[0]['sha1'])
    assert best_result == "chat.png"


def test_fast_path_text_image_similarity(model_name, transformer):
    """Text search runs against stored vectors, i.e. fast path ones."""
    if not transformer.can_handle_text:
        pytest.skip(f"{model_name} does not handle text")
    texts = ['A jumping spider', 'A bird', 'A happy dog', 'A small grey cat']
    expected = ['spider.jpg', 'bird.gif', 'dog.jpg', 'chat.png']
    image_vectors, images = generate_fast_vectors(transformer)

    for text, expected_image in zip(texts, expected):
        text_vector = transformer.to_text_vector(text)
        sims = sorted(
            ((float(cosine_similarity(text_vector, iv)), img.name) for iv, img in zip(image_vectors, images)),
            reverse=True,
        )
        assert sims[0][1] == expected_image, f"'{text}': expected {expected_image}, got {sims[0][1]}"

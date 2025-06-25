import os
from itertools import product

import faiss
import pytest
import numpy as np

from compute.faiss_tree import FaissTree
from compute.transformers import get_transformer, TransformerName, Transformer
from models import VectorType
from utils import preprocess_image

def create_faiss_tree(vectors, images):
    vectors = np.asarray(vectors)
    faiss.normalize_L2(vectors)

    vector_size = vectors.shape[1]
    index = faiss.IndexFlatIP(vector_size)
    index.add(np.asarray(vectors))

    tree = FaissTree(index, images)
    return tree

def get_images():
    return [os.path.join('./resources', f) for f in os.listdir('./resources') if f.split('.')[-1] in ['jpg', 'jpeg', 'png', 'gif']]

def generate_vectors(transformer: Transformer):
    vectors = []
    images = get_images()
    for img_path in images:
        with open(img_path, mode='rb') as f:
            image_data = preprocess_image(f.read())
        vectors.append(transformer.to_vector(image_data))
    return vectors, images

@pytest.fixture(scope='session')
def all_models():
    models = {}
    for model_name in TransformerName:
        print('preloading ' + model_name.name)
        models[model_name] = get_transformer(model_name)
    return models

@pytest.mark.parametrize("model_name, vector_type", list(product(TransformerName, VectorType)))
def test_image_to_vector(model_name, vector_type, all_models):
    """Test tous les transformers disponibles"""
    for img_path in get_images():
        with open(img_path, mode='rb') as f:
            image_data = f.read()
        test_image = preprocess_image(image_data, vector_type)

        print(f"\n=== Testing {model_name.value.upper()} with image {img_path} ===")

        transformer = all_models[model_name]
        print(f"✓ Transformer {model_name} initialisé avec succès")

        # Tester la conversion d'image en vecteur
        print("Testing image to vector conversion...")
        image_vector = transformer.to_vector(test_image)

        # Vérifications
        assert isinstance(image_vector, np.ndarray), f"Le résultat doit être un numpy array pour {model_name}"
        assert image_vector.size > 0, f"Le vecteur ne doit pas être vide pour {model_name}"
        print(f"✓ Image convertie en vecteur de taille: {image_vector.shape}")



@pytest.mark.parametrize("model_name", TransformerName)
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


@pytest.mark.parametrize("model_name", TransformerName)
def test_index_creation(model_name, all_models):
    transformer = all_models[model_name]
    vectors, images = generate_vectors(transformer)
    create_faiss_tree(vectors)

@pytest.mark.parametrize("model_name", TransformerName)
def test_text_image_similarity(model_name, all_models):
    transformer = all_models[model_name]
    texts = ['A jumping spider', 'engraving', 'statue of a face', 'An arachnoid robot']
    expected_results = ['spider.jpg', 'img1.png', 'img2.jpg', 'spider.jpg']
    image_vectors, images = generate_vectors(transformer)
    if not transformer.can_handle_text:
        return
    tree = create_faiss_tree(image_vectors, images)
    for index, text in enumerate(texts):
        result_images = tree.query_texts([text], transformer)
        best_result = os.path.basename(result_images[0]['sha1'])
        print(f"Best image for text: {text} is {best_result}")
        assert best_result == expected_results[index]


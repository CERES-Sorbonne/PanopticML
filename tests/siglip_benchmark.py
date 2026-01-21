import io
import sys
from pathlib import Path

import torch
import torchvision.io
from PIL import Image
from tqdm import tqdm
import numpy as np



class Transformer(object):
    def __init__(self):
        import torch
        from transformers import logging
        logging.set_verbosity_error()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.processor = None
        self.model = None

    @property
    def can_handle_text(self):
        return False

class AutoTransformer(Transformer):
    def __init__(self, hugging_face_model=None):
        super().__init__()
        from transformers import AutoModel, AutoProcessor
        if hugging_face_model:
            self.model = AutoModel.from_pretrained(hugging_face_model).to(self.device)
            self.processor = AutoProcessor.from_pretrained(hugging_face_model)
            self.name = hugging_face_model

    @property
    def can_handle_text(self):
        return True

    def to_vector(self, image: Image) -> np.ndarray:
        inputs = self.processor(images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            vision_outputs = self.model.vision_model(**inputs)
            image_embeds = vision_outputs[1]  # pooled output

            # Applique la projection si elle existe
            if hasattr(self.model, 'visual_projection'):
                image_embeds = self.model.visual_projection(image_embeds)

            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        return image_embeds.cpu().numpy().flatten()

    def to_text_vector(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], return_tensors="pt", padding="max_length", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            text_outputs = self.model.text_model(**inputs)
            text_embeds = text_outputs[1]  # pooled output

            # Applique la projection si elle existe
            if hasattr(self.model, 'text_projection'):
                text_embeds = self.model.text_projection(text_embeds)

            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds.cpu().numpy().flatten()

class SIGLIPTransformer(AutoTransformer):
    def __init__(self, model_name):
        model_name = model_name

        super().__init__(model_name)
        self.name = "SIGLIP"

    @property
    def can_handle_text(self):
        return True

    def to_text_vector_siglip_large(self, text: str) -> np.ndarray:
        # Pour les modèles non-naflex, force le forward complet
        # Crée une image dummy minimale
        dummy_img = Image.new('RGB', (16, 16))  # Plus petit possible

        inputs = self.processor(text=[text], images=[dummy_img], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            text_embeds = outputs.text_embeds
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds.cpu().numpy().flatten()


def get_images(folder, nb=None):
    images = [f for f in folder.iterdir() if
            f.suffix in ['.jpg', '.jpeg', '.png', '.gif'] and f.name != 'cropped_chat.png']
    if nb is not None:
        images = images[:nb]
    return images


def generate_vectors(transformer: Transformer, images=None):
    vectors = []
    images = get_images() if not images else images

    for img_path in tqdm(images):
        image = torchvision.io.read_image(img_path)
        vectors.append(transformer.to_vector(image))
    return vectors, images

def generate_vectors_batch(transformer: Transformer, images=None, batch_size=16):
    image_paths = []
    vectors = []

    for i in tqdm(range(0, len(images), batch_size), desc="Processing batches"):
        batch_paths = images[i:i + batch_size]
        batch_images = []

        # Charger le batch
        for img_path in batch_paths:
            batch_images.append(torchvision.io.read_image(img_path))

        # Traitement batch
        inputs = transformer.processor(images=batch_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(transformer.device) for k, v in inputs.items()}

        with torch.no_grad():
            features = transformer.model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)

        batch_vectors = features.cpu().numpy()
        vectors.extend(batch_vectors)
        image_paths.extend([str(p) for p in batch_paths])

    return vectors, image_paths

def text_image_sim():
    folder = r"./resources"
    siglip = SIGLIPTransformer()
    images = get_images(Path(folder))
    pil_images = [Image.open(i).convert('RGB') for i in images]
    vectors = [siglip.to_vector(i) for i in pil_images]
    texts = ["a cat", "a dog", "a spider"]
    text_vectors = [siglip.to_text_vector_siglip_large(t) for t in texts]
    sim_match = np.dot(vectors[1], text_vectors[0])
    sim_nomatch = np.dot(vectors[1], text_vectors[1])
    print(f"Similarité image-texte correspondant: {sim_match}")
    print(f"Similarité image-texte non-correspondant: {sim_nomatch}")
    print(f"Match > NoMatch ? {sim_match > sim_nomatch}")

def test_debug():
    models_to_test = [
        "google/siglip2-large-patch16-512",  # Celui qui pose problème
        "google/siglip-large-patch16-384",  # Version SigLIP v1
        "google/siglip2-so400m-patch14-384",
        "google/siglip2-so400m-patch16-naflex",
        "google/siglip2-giant-opt-patch16-384"
    ]
    for model_name in models_to_test:
        siglip = SIGLIPTransformer(model_name)
        print(f"===== USING MODEL {model_name} =====")
        image1 = Image.open("./resources/chat.png").convert('RGB')
        image2 = Image.open("./resources/cropped_chat.png").convert('RGB') # Un autre chat
        image3 = Image.open("./resources/spider.jpg").convert('RGB') # Quelque chose de différent

        img1_vec = siglip.to_vector(image1)
        img2_vec = siglip.to_vector(image2)
        img3_vec = siglip.to_vector(image3)

        print("=== SIMILARITÉ IMAGE-IMAGE ===")
        print(f"Chat1 vs Chat2: {np.dot(img1_vec, img2_vec)}")
        print(f"Chat1 vs araignée: {np.dot(img1_vec, img3_vec)}")

        # Test 2 : Similarité TEXTE-TEXTE
        text1 = "a photo of a cat"
        text2 = "a picture of a feline"
        text3 = "a spider looking at the camera"

        txt1_vec = siglip.to_text_vector(text1)
        txt2_vec = siglip.to_text_vector(text2)
        txt3_vec = siglip.to_text_vector(text3)

        print("\n=== SIMILARITÉ TEXTE-TEXTE ===")
        print(f"Cat vs Feline: {np.dot(txt1_vec, txt2_vec)}")
        print(f"Cat vs Spider: {np.dot(txt1_vec, txt3_vec)}")

        # Test 3 : TEXTE-IMAGE (on sait que c'est cassé)
        print("\n=== SIMILARITÉ TEXTE-IMAGE ===")
        print(f"Text 'cat' vs Image chat: {np.dot(txt1_vec, img1_vec)}")
        print(f"Text 'cat' vs Image spider: {np.dot(txt1_vec, img3_vec)}")
        print(f"Text 'spider' vs Image spider: {np.dot(txt3_vec, img3_vec)}")
        print(f"Text 'spider' vs Image chat: {np.dot(txt3_vec, img1_vec)}")

if __name__ == "__main__":
    # vectors, images = generate_vectors(siglip, get_images(Path(folder), 200))
    # vectors, images = generate_vectors_batch(siglip, get_images(Path(folder), 200), 16)
    # text_image_sim()
    test_debug()
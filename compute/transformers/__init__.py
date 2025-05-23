from PIL import Image
import numpy as np

def get_transformer(model="mobilenet"):
    if model == "mobilenet":
        return GoogleTransformer()
    elif model == "clip":
        return CLIPTransformer()
    else:
        return GoogleTransformer()

class Transformer(object):
    def __init__(self):
        import torch
        from transformers import logging
        logging.set_verbosity_error()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def can_handle_text(self):
        return False

class GoogleTransformer(Transformer):
    def __init__(self):
        super().__init__()
        from transformers import MobileNetV2Model, AutoImageProcessor
        self.model = MobileNetV2Model.from_pretrained("google/mobilenet_v2_1.0_224")
        self.processor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")

    @property
    def can_handle_text(self):
        return False

    def to_vector(self, image: Image) -> np.ndarray:
        input1 = self.processor(images=image, return_tensors="pt")
        output1 = self.model(**input1)
        pooled_output1 = output1[1].detach().numpy()
        vector = pooled_output1.flatten()
        return vector

class CLIPTransformer(Transformer):
    def __init__(self):
        super().__init__()
        from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    @property
    def can_handle_text(self):
        return True

    def to_vector(self, image: Image) -> np.ndarray:
        image = self.processor(
            text=None,
            images=image,
            return_tensors="pt"
        )["pixel_values"].to(self.device)  # Transférer sur le bon appareil
        embedding = self.model.get_image_features(image)
        # Convertir les embeddings en tableau numpy
        embedding_as_np = embedding.cpu().detach().numpy()
        return embedding_as_np[0]

    def to_text_vector(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text=text, return_tensors="pt").to(self.device)  # Transférer sur le bon appareil
        text_embeddings = self.model.get_text_features(**inputs)
        # Convertir les embeddings en tableau numpy
        embedding_as_np = text_embeddings.cpu().detach().numpy()
        return embedding_as_np.reshape(1, -1)



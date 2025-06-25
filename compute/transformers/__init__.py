from enum import Enum

from PIL import Image
import numpy as np

class TransformerName(Enum):
    mobilenet = "mobilenet"
    clip = "clip"
    siglip = "siglip"

def get_transformer(model: TransformerName=TransformerName.clip):
    match model:
        case TransformerName.mobilenet:
            return GoogleTransformer()
        case TransformerName.clip:
            return CLIPTransformer()
        case TransformerName.siglip:
            return SIGLIPTransformer()

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
        # from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
        from transformers import AutoModel, AutoTokenizer, AutoProcessor
        ckpt = "openai/clip-vit-base-patch32"
        self.model = AutoModel.from_pretrained(ckpt).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
        self.processor = AutoProcessor.from_pretrained(ckpt)


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


class SIGLIPTransformer(Transformer):
    def __init__(self):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer, AutoProcessor
        ckpt = "google/siglip2-so400m-patch16-naflex"
        self.model = AutoModel.from_pretrained(ckpt).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
        self.processor = AutoProcessor.from_pretrained(ckpt)

    @property
    def can_handle_text(self):
        return True

    def to_vector(self, image: Image) -> np.ndarray:
        inputs = (self.processor(images=[image], return_tensors="pt")
                  .to(self.device))
        image_embeddings = self.model.get_image_features(**inputs)
        embedding_as_np = image_embeddings.cpu().detach().numpy()
        return embedding_as_np[0]


    def to_text_vector(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text=text, return_tensors="pt").to(self.device)
        text_embeddings = self.model.get_text_features(**inputs)
        # Convertir les embeddings en tableau numpy
        embedding_as_np = text_embeddings.cpu().detach().numpy()
        return embedding_as_np.reshape(1, -1)
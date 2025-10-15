from enum import Enum

import torch
from PIL import Image
import numpy as np

from ...utils import resolve_device


class TransformerName(Enum):
    mobilenet = "mobilenet"
    clip = "clip"
    siglip = "siglip"
    dinov2 = "dinov2"
    auto = "auto"

def get_transformer(model: TransformerName=TransformerName.clip, hugging_face_model=None):
    match model:
        case TransformerName.mobilenet:
            return GoogleTransformer()
        case TransformerName.clip:
            return CLIPTransformer()
        case TransformerName.siglip:
            return SIGLIPTransformer()
        case TransformerName.dinov2:
            return Dinov2Transformer()
        case TransformerName.auto:
            return AutoTransformer(hugging_face_model)

class Transformer(object):
    def __init__(self):
        import torch
        from transformers import logging
        logging.set_verbosity_error()
        self.device = resolve_device()
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
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().flatten()

    def to_text_vector(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().flatten()

class GoogleTransformer(Transformer):
    def __init__(self):
        super().__init__()
        from transformers import MobileNetV2Model, AutoImageProcessor
        self.model = MobileNetV2Model.from_pretrained("google/mobilenet_v2_1.0_224")
        self.processor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")
        self.name = "MobileNetV2"

    @property
    def can_handle_text(self):
        return False

    def to_vector(self, image: Image) -> np.ndarray:
        input1 = self.processor(images=image, return_tensors="pt")
        output1 = self.model(**input1)
        pooled_output1 = output1[1].detach().numpy()
        vector = pooled_output1.flatten()
        return vector

class CLIPTransformer(AutoTransformer):
    def __init__(self):
        model_name = "openai/clip-vit-base-patch32"
        super().__init__(model_name)
        self.name = "CLIP"


    @property
    def can_handle_text(self):
        return True

class SIGLIPTransformer(AutoTransformer):
    def __init__(self):
        model_name = "google/siglip2-so400m-patch16-naflex"
        super().__init__(model_name)
        self.name = "SIGLIP"

    @property
    def can_handle_text(self):
        return True


class Dinov2Transformer(Transformer):
    def __init__(self):
        super().__init__()
        from transformers import AutoModel, AutoImageProcessor
        ckpt = "facebook/dinov2-base"
        self.model = AutoModel.from_pretrained(ckpt).to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(ckpt, use_fast=True)
        self.name = "Dinov2"

    @property
    def can_handle_text(self):
        return False

    def to_vector(self, image: Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
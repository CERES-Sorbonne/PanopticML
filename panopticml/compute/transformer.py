import asyncio
from enum import Enum
import torch
from PIL import Image
import numpy as np
from transformers import AutoConfig

from panoptic.models import VectorType
from ..utils import resolve_device


def get_model_type(huggingface_model: str):
    return AutoConfig.from_pretrained(huggingface_model).model_type


def get_transformer(huggingface_model=None):
    model_type = get_model_type(huggingface_model)
    if model_type in type_to_class_mapping:
        return type_to_class_mapping[model_type](huggingface_model)
    return AutoTransformer(huggingface_model)

class Transformer(object):
    def __init__(self, huggingface_model: str):
        from transformers import logging
        logging.set_verbosity_error()
        self.device = resolve_device()
        self.tokenizer = None
        self.processor = None
        self.model = None
        self.can_handle_text = False
        self.name = huggingface_model

    @property
    def max_text_sim(self):
        if not self.can_handle_text:
            raise ValueError("Cannot get max text sim for this model since it cannot handle text")
        return self._max_text_sim

    @max_text_sim.setter
    def max_text_sim(self, max_text_sim):
        if max_text_sim < 1:
            self._max_text_sim = max_text_sim

    def to_vector(self, image: Image):
        pass

    def to_text_vector(self, text: str):
        pass

    def get_text_vectors(self, texts: [str]):
        vectors = []
        if self.can_handle_text:
            for text in texts:
                vectors.append(self.to_text_vector(text))
        else:
            raise ValueError(f"The selected transformer {self.name} does not support text vectors.")
        return np.asarray(vectors)


class AutoTransformer(Transformer):
    def __init__(self, huggingface_model):
        super().__init__(huggingface_model)
        from transformers import AutoModel, AutoProcessor
        self.model = AutoModel.from_pretrained(huggingface_model).to(self.device)
        self.processor = AutoProcessor.from_pretrained(huggingface_model)
        self.can_handle_text = True

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


class MobileNetTransformer(Transformer):
    def __init__(self, huggingface_model: str):
        super().__init__(huggingface_model)
        from transformers import MobileNetV2Model, AutoImageProcessor
        self.model = MobileNetV2Model.from_pretrained(huggingface_model)
        self.processor = AutoImageProcessor.from_pretrained(huggingface_model)

    def to_vector(self, image: Image) -> np.ndarray:
        input1 = self.processor(images=image, return_tensors="pt")
        output1 = self.model(**input1)
        pooled_output1 = output1[1].detach().numpy()
        vector = pooled_output1.flatten()
        return vector


class CLIPTransformer(AutoTransformer):
    def __init__(self, huggingface_model):
        super().__init__(huggingface_model)
        self.max_text_sim = 0.375


class SIGLIPTransformer(AutoTransformer):
    def __init__(self, huggingface_model):
        super().__init__(huggingface_model)
        self.max_text_sim = 0.20


class Dinov2Transformer(Transformer):
    def __init__(self, huggingface_model: str):
        super().__init__(huggingface_model)
        from transformers import AutoModel, AutoImageProcessor
        self.model = AutoModel.from_pretrained(huggingface_model).to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(huggingface_model, use_fast=True)
        self.can_handle_text = False

    def to_vector(self, image: Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]

        # Normalisation L2
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding


type_to_class_mapping = {
    "mobilenet_v2": MobileNetTransformer,
    "dinov2": Dinov2Transformer,
    "siglip2": SIGLIPTransformer,
    "clip": CLIPTransformer
}


GET_LOCK = asyncio.Lock()

class TransformerManager:
    def __init__(self):
        self.transformers: dict[int, Transformer] = {}

    def get(self, vec_type: VectorType):
        if self.transformers.get(vec_type.id):
            return self.transformers[vec_type.id]
        self.transformers[vec_type.id] = get_transformer(vec_type.params["model"])
        return self.transformers[vec_type.id]

    async def async_get(self, project, vec_type: VectorType):
        async with GET_LOCK:
            if self.transformers.get(vec_type.id):
                return self.transformers[vec_type.id]
            self.transformers[vec_type.id] = await project.run_async(get_transformer, vec_type.params["model"])
            return self.transformers[vec_type.id]

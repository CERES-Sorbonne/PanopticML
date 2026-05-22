import threading

import torch
from PIL import Image
import numpy as np
from transformers import AutoConfig

from panoptic.core.databases.media.models import VectorType
from ..utils import resolve_device


def _unwrap_model_output(features) -> torch.Tensor:
    if isinstance(features, torch.Tensor):
        return features
    if hasattr(features, 'image_embeds'):
        return features.image_embeds
    if hasattr(features, 'pooler_output') and features.pooler_output is not None:
        return features.pooler_output
    if hasattr(features, 'last_hidden_state'):
        return features.last_hidden_state[:, 0]
    raise ValueError(f"Cannot extract embedding tensor from {type(features)}")


def _get_input_size(processor) -> int:
    """Read the expected square input size from a HuggingFace image processor."""
    ip = processor.image_processor
    for attr in ('crop_size', 'size'):
        cfg = getattr(ip, attr, None)
        if isinstance(cfg, dict):
            for key in ('height', 'shortest_edge', 'width'):
                if key in cfg:
                    return int(cfg[key])
        if isinstance(cfg, int):
            return cfg
    return 224


def get_model_type(huggingface_model: str) -> str:
    return AutoConfig.from_pretrained(huggingface_model).model_type


def extract_model_type(vec_type: VectorType) -> str:
    full_name = vec_type.params['model']
    return full_name.split('/')[-1].split('-')[0]


def get_transformer(huggingface_model: str) -> 'Transformer':
    model_type = get_model_type(huggingface_model)
    if model_type in type_to_class_mapping:
        return type_to_class_mapping[model_type](huggingface_model)
    return AutoTransformer(huggingface_model)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Transformer:
    def __init__(self, huggingface_model: str):
        from transformers import logging
        logging.set_verbosity_error()
        self.device = resolve_device()
        self.processor = None
        self.model = None
        self.can_handle_text = False
        self.name = huggingface_model
        self.preprocess_size: int = 224  # overridden by subclasses

    @property
    def max_text_sim(self) -> float:
        if not self.can_handle_text:
            raise ValueError(f"Model {self.name} does not support text similarity")
        return self._max_text_sim

    @max_text_sim.setter
    def max_text_sim(self, value: float):
        if value < 1:
            self._max_text_sim = value

    def to_vector(self, image: Image.Image) -> np.ndarray:
        return self.to_vectors_batch([image])[0]

    def to_vectors_batch(self, images: list[Image.Image]) -> np.ndarray:
        raise NotImplementedError

    def forward_from_arrays(self, arrays: list[np.ndarray]) -> np.ndarray:
        """Forward pass from pre-resized uint8 (H×W×3) numpy arrays.

        Default: convert back to PIL and go through the normal path.
        Subclasses override this to do GPU normalization directly.
        """
        from PIL import Image as _Image
        images = [_Image.fromarray(a) for a in arrays]
        return self.to_vectors_batch(images)

    def get_text_vectors(self, texts: list[str]) -> np.ndarray:
        if not self.can_handle_text:
            raise ValueError(f"Model {self.name} does not support text vectors")
        return np.asarray([self.to_text_vector(t) for t in texts])

    def to_text_vector(self, text: str) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# AutoTransformer  (CLIP, SigLIP, …)
# ---------------------------------------------------------------------------

class AutoTransformer(Transformer):
    max_text_sim = 0.20

    def __init__(self, huggingface_model: str):
        super().__init__(huggingface_model)
        import logging
        from transformers import AutoModel, AutoProcessor
        logger = logging.getLogger('PanopticML')

        dtype = torch.float16 if self.device == 'cuda' else torch.float32
        self.model = AutoModel.from_pretrained(
            huggingface_model,
            torch_dtype=dtype,
            device_map=self.device,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(huggingface_model)
        self.can_handle_text = True

        # Preprocessing params — used by forward_from_arrays
        self.preprocess_size = _get_input_size(self.processor)
        ip = self.processor.image_processor
        self._norm_mean = torch.tensor(ip.image_mean, dtype=torch.float32,
                                       device=self.device).view(1, 3, 1, 1)
        self._norm_std  = torch.tensor(ip.image_std,  dtype=torch.float32,
                                       device=self.device).view(1, 3, 1, 1)

        logger.info(f"PanopticML: loaded {huggingface_model!r} "
                    f"on {self.device} input={self.preprocess_size}px")

    def _to_device(self, inputs: dict) -> dict:
        return {k: v.to(self.device) for k, v in inputs.items()}

    # --- PIL path (similarity search, text search, single-image ops) -------

    def to_vectors_batch(self, images: list[Image.Image]) -> np.ndarray:
        inputs = self._to_device(self.processor(images=images, return_tensors="pt"))
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
            features = _unwrap_model_output(features)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().float().numpy()

    def to_text_vector(self, text: str) -> np.ndarray:
        inputs = self._to_device(self.processor(text=[text], return_tensors="pt"))
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
            features = _unwrap_model_output(features)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().float().numpy().flatten()

    # --- Fast path: pre-resized arrays, GPU normalization ------------------

    def forward_from_arrays(self, arrays: list[np.ndarray]) -> np.ndarray:
        # arrays: list of uint8 H×W×3 already resized to preprocess_size
        batch = (torch.from_numpy(np.stack(arrays))
                 .permute(0, 3, 1, 2)   # N×3×H×W
                 .float().div_(255)
                 .to(self.device))
        batch = (batch - self._norm_mean) / self._norm_std
        with torch.no_grad():
            features = self.model.get_image_features(pixel_values=batch)
            features = _unwrap_model_output(features)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().float().numpy()


# ---------------------------------------------------------------------------
# MobileNet
# ---------------------------------------------------------------------------

class MobileNetTransformer(Transformer):
    def __init__(self, huggingface_model: str):
        super().__init__(huggingface_model)
        from transformers import MobileNetV2Model, AutoImageProcessor
        dtype = torch.float16 if self.device == 'cuda' else torch.float32
        self.model = MobileNetV2Model.from_pretrained(
            huggingface_model, torch_dtype=dtype, device_map=self.device
        )
        self.model.eval()
        self.processor = AutoImageProcessor.from_pretrained(huggingface_model)
        self.preprocess_size = _get_input_size(self.processor)
        ip = self.processor
        self._norm_mean = torch.tensor(ip.image_mean, dtype=torch.float32,
                                       device=self.device).view(1, 3, 1, 1)
        self._norm_std  = torch.tensor(ip.image_std,  dtype=torch.float32,
                                       device=self.device).view(1, 3, 1, 1)

    def to_vectors_batch(self, images: list[Image.Image]) -> np.ndarray:
        inputs = {k: v.to(self.device)
                  for k, v in self.processor(images=images, return_tensors="pt").items()}
        with torch.no_grad():
            output = self.model(**inputs)
        return output[1].detach().cpu().float().numpy()

    def forward_from_arrays(self, arrays: list[np.ndarray]) -> np.ndarray:
        batch = (torch.from_numpy(np.stack(arrays))
                 .permute(0, 3, 1, 2).float().div_(255).to(self.device))
        batch = (batch - self._norm_mean) / self._norm_std
        with torch.no_grad():
            output = self.model(pixel_values=batch)
        return output[1].detach().cpu().float().numpy()


# ---------------------------------------------------------------------------
# CLIP / SigLIP  (aliases of AutoTransformer with different max_text_sim)
# ---------------------------------------------------------------------------

class CLIPTransformer(AutoTransformer):
    max_text_sim = 0.375

    def __init__(self, huggingface_model: str):
        super().__init__(huggingface_model)


class SIGLIPTransformer(AutoTransformer):
    max_text_sim = 0.20

    def __init__(self, huggingface_model: str):
        super().__init__(huggingface_model)


# ---------------------------------------------------------------------------
# DINOv2
# ---------------------------------------------------------------------------

class Dinov2Transformer(Transformer):
    def __init__(self, huggingface_model: str):
        super().__init__(huggingface_model)
        from transformers import AutoModel, AutoImageProcessor
        dtype = torch.float16 if self.device == 'cuda' else torch.float32
        self.model = AutoModel.from_pretrained(
            huggingface_model, torch_dtype=dtype, device_map=self.device
        )
        self.model.eval()
        self.processor = AutoImageProcessor.from_pretrained(huggingface_model, use_fast=True)
        self.can_handle_text = False
        self.preprocess_size = _get_input_size(self.processor)
        ip = self.processor
        self._norm_mean = torch.tensor(ip.image_mean, dtype=torch.float32,
                                       device=self.device).view(1, 3, 1, 1)
        self._norm_std  = torch.tensor(ip.image_std,  dtype=torch.float32,
                                       device=self.device).view(1, 3, 1, 1)

    def to_vectors_batch(self, images: list[Image.Image]) -> np.ndarray:
        inputs = {k: v.to(self.device)
                  for k, v in self.processor(images=images, return_tensors="pt").items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().float().numpy()
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    def forward_from_arrays(self, arrays: list[np.ndarray]) -> np.ndarray:
        batch = (torch.from_numpy(np.stack(arrays))
                 .permute(0, 3, 1, 2).float().div_(255).to(self.device))
        batch = (batch - self._norm_mean) / self._norm_std
        with torch.no_grad():
            outputs = self.model(pixel_values=batch)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().float().numpy()
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms


# ---------------------------------------------------------------------------

type_to_class_mapping = {
    "mobilenet_v2": MobileNetTransformer,
    "dinov2":       Dinov2Transformer,
    "siglip2":      SIGLIPTransformer,
    "siglip":       SIGLIPTransformer,
    "clip":         CLIPTransformer,
}


class TransformerManager:
    def __init__(self):
        self.transformers: dict[int, Transformer] = {}
        self._lock = threading.Lock()

    def get(self, vec_type: VectorType) -> Transformer:
        type_id = vec_type.id
        if self.transformers.get(type_id):
            return self.transformers[type_id]
        with self._lock:
            if self.transformers.get(type_id):
                return self.transformers[type_id]
            self.transformers[type_id] = get_transformer(vec_type.params["model"])
            return self.transformers[type_id]

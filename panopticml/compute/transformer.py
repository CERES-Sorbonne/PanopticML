import asyncio
from enum import Enum
import torch
from PIL import Image
import numpy as np
from transformers import AutoConfig, BitsAndBytesConfig

from panoptic.models import VectorType
from ..utils import resolve_device


def get_model_type(huggingface_model: str):
    try:
        config = AutoConfig.from_pretrained(huggingface_model)
    except (ValueError, OSError):
        # Some models (e.g. NVIDIA RADIO) ship a custom config that requires
        # trust_remote_code to be loaded.
        config = AutoConfig.from_pretrained(huggingface_model, trust_remote_code=True)
    model_type = getattr(config, "model_type", "") or ""
    if not model_type:
        # Custom configs sometimes leave model_type empty (e.g. RADIO). Fall back
        # to the config class name: RADIOConfig -> "radio".
        model_type = type(config).__name__.lower().removesuffix("config")
    return model_type

def extract_model_type(vec_type: VectorType):
    full_name = vec_type.params['model']
    full_model = full_name.split('/')[-1]
    model = full_model.split('-')[0]
    return model.lower()

def get_transformer(huggingface_model=None):
    model_type = get_model_type(huggingface_model)
    if model_type in type_to_class_mapping:
        return type_to_class_mapping[model_type](huggingface_model)
    try:
        model = AutoTransformer(huggingface_model)
        return model
    except torch.OutOfMemoryError:
        for model in type_to_class_mapping:
            del type_to_class_mapping[model]
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
    max_text_sim = 0.20

    def __init__(self, huggingface_model):
        super().__init__(huggingface_model)
        from transformers import AutoModel, AutoProcessor
        torch_type = torch.float32
        if torch.cuda.is_available():
            torch_type = torch.float16
        # Chargement en float16 sans quantification
        self.model = AutoModel.from_pretrained(
            huggingface_model,
            torch_dtype=torch_type,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(huggingface_model, trust_remote_code=True)
        self.can_handle_text = True

    def to_vector(self, image: Image) -> np.ndarray:
        inputs = self.processor(images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().float().numpy().flatten()

    def to_text_vector(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().float().numpy().flatten()

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
    max_test_sim = 0.375

    def __init__(self, huggingface_model):
        super().__init__(huggingface_model)



class SIGLIPTransformer(AutoTransformer):
    max_text_sim = 0.20

    def __init__(self, huggingface_model):
        super().__init__(huggingface_model)


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


class Dinov3Transformer(Transformer):
    """
    Meta DINOv3 vision transformer (self-supervised foundation model).
    See: https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m
    """
    def __init__(self, huggingface_model: str):
        super().__init__(huggingface_model)
        from transformers import AutoModel, AutoImageProcessor
        self.model = AutoModel.from_pretrained(huggingface_model).to(self.device)
        self.model.eval()
        self.processor = AutoImageProcessor.from_pretrained(huggingface_model, use_fast=True)
        self.can_handle_text = False
        # Number of leading non-patch tokens (CLS + register tokens) to skip when
        # falling back to patch mean-pooling.
        num_register = getattr(self.model.config, "num_register_tokens", 0) or 0
        self._num_prefix_tokens = 1 + num_register

    def to_vector(self, image: Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        pooler_output = getattr(outputs, "pooler_output", None)
        if pooler_output is not None:
            embedding = pooler_output.cpu().numpy()[0]
        else:
            patch_tokens = outputs.last_hidden_state[:, self._num_prefix_tokens:, :]
            embedding = patch_tokens.mean(dim=1).cpu().numpy()[0]

        # Normalisation L2
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding


def _repair_remote_code_modules(huggingface_model: str):
    """
    Work around a transformers bug with NVIDIA RADIO models: `radio_model.py`
    imports `dual_hybrid_vit` via `from . import dual_hybrid_vit`, a form the
    relative-import scanner does not detect, so transformers never copies that
    file into its dynamic `transformers_modules` directory and loading fails
    with a missing-file error. Here we copy every `*.py` of the repo from the
    hub snapshot into the dynamic module directory so all submodules resolve.
    """
    import os
    import shutil
    from pathlib import Path
    from huggingface_hub import snapshot_download
    from transformers.dynamic_module_utils import _sanitize_module_name
    from transformers.utils import HF_MODULES_CACHE

    snapshot = Path(snapshot_download(huggingface_model, allow_patterns=["*.py"]))
    submodule = os.path.sep.join(_sanitize_module_name(p) for p in huggingface_model.split("/"))
    base_dir = Path(HF_MODULES_CACHE) / "transformers_modules" / submodule
    if not base_dir.exists():
        return
    # The dynamic code is executed from a per-commit subfolder; copy any missing
    # `*.py` of the repo into each one.
    for module_dir in (d for d in base_dir.iterdir() if d.is_dir()):
        for py_file in snapshot.glob("*.py"):
            target = module_dir / py_file.name
            if not target.exists():
                shutil.copy(py_file, target)


class RadioTransformer(Transformer):
    """
    NVIDIA C-RADIO transformer (agglomerative vision foundation model).
    See: https://huggingface.co/nvidia/C-RADIOv4-H
    """
    def __init__(self, huggingface_model: str):
        super().__init__(huggingface_model)
        from transformers import AutoModel, CLIPImageProcessor
        self.processor = CLIPImageProcessor.from_pretrained(huggingface_model)
        # Proactively ensure all remote-code submodules are present (transformers
        # fails to copy dual_hybrid_vit.py for RADIO models).
        _repair_remote_code_modules(huggingface_model)
        try:
            self.model = AutoModel.from_pretrained(
                huggingface_model,
                trust_remote_code=True
            )
        except (FileNotFoundError, ModuleNotFoundError):
            # First-ever load: the dynamic module dir only exists after the failed
            # attempt above. Repair it and retry.
            _repair_remote_code_modules(huggingface_model)
            self.model = AutoModel.from_pretrained(
                huggingface_model,
                trust_remote_code=True
            )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.can_handle_text = False

    def to_vector(self, image: Image) -> np.ndarray:
        pixel_values = self.processor(
            images=image, return_tensors="pt", do_resize=True
        ).pixel_values.to(self.device)

        # RADIO requires the input resolution to be a multiple of its patch step;
        # the variable-resolution processor does not guarantee this, so snap to the
        # nearest supported resolution before the forward pass.
        height, width = pixel_values.shape[-2:]
        nearest = self.model.get_nearest_supported_resolution(height, width)
        if (nearest.height, nearest.width) != (height, width):
            pixel_values = torch.nn.functional.interpolate(
                pixel_values, size=(nearest.height, nearest.width),
                mode="bilinear", align_corners=False
            )

        with torch.no_grad():
            summary, _features = self.model(pixel_values)

        embedding = summary.cpu().float().numpy().flatten()

        # Normalisation L2
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding


type_to_class_mapping = {
    "mobilenet_v2": MobileNetTransformer,
    "dinov2": Dinov2Transformer,
    "dinov3": Dinov3Transformer,
    "siglip2": SIGLIPTransformer,
    "siglip": SIGLIPTransformer,
    "clip": CLIPTransformer,
    "radio": RadioTransformer
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

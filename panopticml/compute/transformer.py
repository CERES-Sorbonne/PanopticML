import gc
import math
import threading

import torch
from PIL import Image
import numpy as np

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
    """Read the expected square input size from a HuggingFace image processor.

    Accepts either an AutoProcessor (which wraps an image_processor) or an
    AutoImageProcessor (which is one).
    """
    ip = getattr(processor, 'image_processor', processor)
    for attr in ('crop_size', 'size'):
        cfg = getattr(ip, attr, None)
        if cfg is None:
            continue
        if isinstance(cfg, int):
            return cfg
        # a dict on transformers 4, a SizeDict object on transformers 5
        for key in ('height', 'shortest_edge', 'width'):
            value = cfg.get(key) if isinstance(cfg, dict) else getattr(cfg, key, None)
            if value:
                return int(value)
    return 224


def from_pretrained(loader, huggingface_model: str, **kwargs):
    """Load a config/model/processor, preferring the local HuggingFace cache.

    ``from_pretrained`` revalidates against huggingface.co on every call, which costs
    1-4s per artefact even when the files are already on disk — several seconds of
    plugin startup for nothing. Try the cache first and only hit the network when the
    model really isn't there yet (first use of a model).
    """
    try:
        return loader.from_pretrained(huggingface_model, local_files_only=True, **kwargs)
    except Exception:
        return loader.from_pretrained(huggingface_model, **kwargs)


def get_model_type(huggingface_model: str) -> str:
    from transformers import AutoConfig
    try:
        # explicit False: never let transformers prompt on stdin for remote code
        config = from_pretrained(AutoConfig, huggingface_model, trust_remote_code=False)
    except (ValueError, OSError):
        # Some models (e.g. NVIDIA RADIO) ship a custom config that needs trust_remote_code.
        config = from_pretrained(AutoConfig, huggingface_model, trust_remote_code=True)
    model_type = getattr(config, 'model_type', '') or ''
    if not model_type:
        # Custom configs can leave model_type empty: RADIOConfig -> "radio".
        model_type = type(config).__name__.lower().removesuffix('config')
    return model_type


def get_transformer(huggingface_model: str) -> 'Transformer':
    # Apple MobileCLIP repos ship a raw open_clip checkpoint and no transformers config,
    # so AutoConfig can't detect them: route them by name first.
    if 'mobileclip' in huggingface_model.lower():
        return MobileClipTransformer(huggingface_model)
    model_type = get_model_type(huggingface_model)
    if model_type in type_to_class_mapping:
        return type_to_class_mapping[model_type](huggingface_model)
    return AutoTransformer(huggingface_model)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Transformer:
    # reduced precision used on CUDA; models whose activations overflow fp16 use bfloat16
    cuda_dtype = torch.float16

    def __init__(self, huggingface_model: str):
        from transformers import logging
        logging.set_verbosity_error()
        self.device = resolve_device()
        self.dtype = self._resolve_dtype()
        self.processor = None
        self.model = None
        self.can_handle_text = False
        self.name = huggingface_model
        self.preprocess_size: int = 224  # overridden by subclasses

    def _resolve_dtype(self) -> torch.dtype:
        if self.device != 'cuda':
            return torch.float32
        if self.cuda_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            return torch.float32  # pre-Ampere GPUs: fp16 would overflow, stay in full precision
        return self.cuda_dtype

    @property
    def max_text_sim(self) -> float:
        if not self.can_handle_text:
            raise ValueError(f"Model {self.name} does not support text similarity")
        return self._max_text_sim

    @max_text_sim.setter
    def max_text_sim(self, value: float):
        if value < 1:
            self._max_text_sim = value

    def _to_device(self, inputs) -> dict:
        """Move processor outputs to the device. Float tensors are cast to the model
        dtype (fp16 on CUDA); integer tensors (input_ids, masks) keep their type."""
        return {
            k: v.to(self.device, self.dtype) if v.is_floating_point() else v.to(self.device)
            for k, v in inputs.items()
        }

    def _normalize_batch(self, arrays: list[np.ndarray]) -> torch.Tensor:
        """uint8 H×W×3 arrays -> normalized N×3×H×W tensor in the model dtype."""
        batch = (torch.from_numpy(np.stack(arrays))
                 .permute(0, 3, 1, 2).float().div_(255).to(self.device))
        batch = (batch - self._norm_mean) / self._norm_std
        return batch.to(self.dtype)

    def to_vector(self, image: Image.Image) -> np.ndarray:
        return self.to_vectors_batch([image])[0]

    def to_vectors_batch(self, images) -> np.ndarray:
        raise NotImplementedError

    def forward_from_arrays(self, arrays: list[np.ndarray]) -> np.ndarray:
        """Forward pass from pre-resized uint8 (H×W×3) arrays. Subclasses override
        with a GPU fast path; the default rebuilds PIL images for the normal path."""
        return self.to_vectors_batch([Image.fromarray(a) for a in arrays])

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

        # trust_remote_code: custom vector types can name any HuggingFace model
        self.model = from_pretrained(
            AutoModel, huggingface_model, torch_dtype=self.dtype, device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()
        self.processor = from_pretrained(AutoProcessor, huggingface_model, trust_remote_code=True)
        self.can_handle_text = True

        ip = self.processor.image_processor
        # NaFlex models (e.g. siglip2 …-naflex) patchify inside the processor and take a
        # patch *sequence*, not a fixed H×W tensor — they can't use the GPU fast path and
        # always go through the processor.
        self.patch_sequence = hasattr(ip, 'max_num_patches')
        if self.patch_sequence:
            self.preprocess_size = math.isqrt(ip.max_num_patches) * ip.patch_size
        else:
            self.preprocess_size = _get_input_size(self.processor)
            self._norm_mean = torch.tensor(ip.image_mean, dtype=torch.float32,
                                           device=self.device).view(1, 3, 1, 1)
            self._norm_std = torch.tensor(ip.image_std, dtype=torch.float32,
                                          device=self.device).view(1, 3, 1, 1)

        logger.info(f"PanopticML: loaded {huggingface_model!r} on {self.device} "
                    f"input={self.preprocess_size}px"
                    f"{' (patch sequence)' if self.patch_sequence else ''}")

    def to_vectors_batch(self, images) -> np.ndarray:
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

    def forward_from_arrays(self, arrays: list[np.ndarray]) -> np.ndarray:
        if self.patch_sequence:
            return self.to_vectors_batch(arrays)  # processor handles patchification
        with torch.no_grad():
            features = self.model.get_image_features(pixel_values=self._normalize_batch(arrays))
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
        self.model = from_pretrained(
            MobileNetV2Model, huggingface_model, torch_dtype=self.dtype, device_map=self.device
        )
        self.model.eval()
        self.processor = from_pretrained(AutoImageProcessor, huggingface_model)
        self.preprocess_size = _get_input_size(self.processor)
        self._norm_mean = torch.tensor(self.processor.image_mean, dtype=torch.float32,
                                       device=self.device).view(1, 3, 1, 1)
        self._norm_std = torch.tensor(self.processor.image_std, dtype=torch.float32,
                                      device=self.device).view(1, 3, 1, 1)

    def to_vectors_batch(self, images) -> np.ndarray:
        inputs = self._to_device(self.processor(images=images, return_tensors="pt"))
        with torch.no_grad():
            output = self.model(**inputs)
        return output[1].detach().cpu().float().numpy()

    def forward_from_arrays(self, arrays: list[np.ndarray]) -> np.ndarray:
        with torch.no_grad():
            output = self.model(pixel_values=self._normalize_batch(arrays))
        return output[1].detach().cpu().float().numpy()


# ---------------------------------------------------------------------------
# CLIP / SigLIP  (aliases of AutoTransformer with different max_text_sim)
# ---------------------------------------------------------------------------

class CLIPTransformer(AutoTransformer):
    max_text_sim = 0.375


class SIGLIPTransformer(AutoTransformer):
    max_text_sim = 0.20


# ---------------------------------------------------------------------------
# DINOv2
# ---------------------------------------------------------------------------

class Dinov2Transformer(Transformer):
    def __init__(self, huggingface_model: str):
        super().__init__(huggingface_model)
        from transformers import AutoModel, AutoImageProcessor
        self.model = from_pretrained(
            AutoModel, huggingface_model, torch_dtype=self.dtype, device_map=self.device
        )
        self.model.eval()
        self.processor = from_pretrained(AutoImageProcessor, huggingface_model, use_fast=True)
        self.can_handle_text = False
        self.preprocess_size = _get_input_size(self.processor)
        self._norm_mean = torch.tensor(self.processor.image_mean, dtype=torch.float32,
                                       device=self.device).view(1, 3, 1, 1)
        self._norm_std = torch.tensor(self.processor.image_std, dtype=torch.float32,
                                      device=self.device).view(1, 3, 1, 1)

    def _embed(self, outputs) -> np.ndarray:
        return _l2_normalize(outputs.last_hidden_state.mean(dim=1))

    def to_vectors_batch(self, images) -> np.ndarray:
        inputs = self._to_device(self.processor(images=images, return_tensors="pt"))
        with torch.no_grad():
            outputs = self.model(**inputs)
        return self._embed(outputs)

    def forward_from_arrays(self, arrays: list[np.ndarray]) -> np.ndarray:
        with torch.no_grad():
            outputs = self.model(pixel_values=self._normalize_batch(arrays))
        return self._embed(outputs)


def _l2_normalize(embeddings: torch.Tensor) -> np.ndarray:
    embeddings = embeddings.cpu().float().numpy()
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


# ---------------------------------------------------------------------------
# DINOv3
# ---------------------------------------------------------------------------

class Dinov3Transformer(Dinov2Transformer):
    """Meta DINOv3 vision transformer (needs transformers >= 4.56).
    See: https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m
    """
    # DINOv3 activations overflow fp16: every vector comes out NaN
    cuda_dtype = torch.bfloat16

    def __init__(self, huggingface_model: str):
        super().__init__(huggingface_model)
        # CLS + register tokens, skipped when falling back to patch mean-pooling
        num_register = getattr(self.model.config, 'num_register_tokens', 0) or 0
        self._num_prefix_tokens = 1 + num_register

    def _embed(self, outputs) -> np.ndarray:
        pooled = getattr(outputs, 'pooler_output', None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, self._num_prefix_tokens:].mean(dim=1)
        return _l2_normalize(pooled)


# ---------------------------------------------------------------------------
# NVIDIA C-RADIO
# ---------------------------------------------------------------------------

def _repair_remote_code_modules(huggingface_model: str) -> None:
    """Work around a transformers bug with NVIDIA RADIO models: `radio_model.py` imports
    `dual_hybrid_vit` with `from . import dual_hybrid_vit`, a form the relative-import
    scanner misses, so transformers never copies that file into its dynamic
    `transformers_modules` directory and loading fails with a missing file. Copy every
    `*.py` of the repo from the hub snapshot into that directory.

    Best effort: recent transformers load RADIO without it.
    """
    import os
    import shutil
    from pathlib import Path
    try:
        from huggingface_hub import snapshot_download
        from transformers.dynamic_module_utils import _sanitize_module_name
        from transformers.utils import HF_MODULES_CACHE
        try:
            snapshot = snapshot_download(huggingface_model, allow_patterns=["*.py"], local_files_only=True)
        except Exception:
            snapshot = snapshot_download(huggingface_model, allow_patterns=["*.py"])
        submodule = os.path.sep.join(_sanitize_module_name(p) for p in huggingface_model.split("/"))
        base_dir = Path(HF_MODULES_CACHE) / "transformers_modules" / submodule
        if not base_dir.exists():
            return
        # the dynamic code runs from a per-commit subfolder: fill in each one
        for module_dir in (d for d in base_dir.iterdir() if d.is_dir()):
            for py_file in Path(snapshot).glob("*.py"):
                target = module_dir / py_file.name
                if not target.exists():
                    shutil.copy(py_file, target)
    except Exception:
        pass


class RadioTransformer(Transformer):
    """NVIDIA C-RADIO agglomerative vision model (remote code, needs timm and einops).
    Vectors are the model's summary token.
    See: https://huggingface.co/nvidia/C-RADIOv4-H
    """
    def __init__(self, huggingface_model: str):
        super().__init__(huggingface_model)
        from transformers import AutoModel, CLIPImageProcessor
        self.dtype = torch.float32  # remote code: kept in full precision
        self.processor = from_pretrained(CLIPImageProcessor, huggingface_model)
        _repair_remote_code_modules(huggingface_model)
        try:
            self.model = from_pretrained(AutoModel, huggingface_model, trust_remote_code=True)
        except (FileNotFoundError, ModuleNotFoundError):
            # First-ever load: the dynamic module dir only exists after the failed
            # attempt above. Repair it and retry.
            _repair_remote_code_modules(huggingface_model)
            self.model = from_pretrained(AutoModel, huggingface_model, trust_remote_code=True)
        self.model = self.model.to(self.device).eval()
        self.can_handle_text = False

        # RADIO needs a resolution that is a multiple of its patch step
        size = _get_input_size(self.processor)
        self.preprocess_size = self.model.get_nearest_supported_resolution(size, size).height
        # The model normalizes its input itself: the processor only rescales to [0, 1]
        # (do_normalize is off), so the fast path must not normalize either.
        normalize = getattr(self.processor, 'do_normalize', True)
        mean = self.processor.image_mean if normalize else (0.0, 0.0, 0.0)
        std = self.processor.image_std if normalize else (1.0, 1.0, 1.0)
        self._norm_mean = torch.tensor(mean, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self._norm_std = torch.tensor(std, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    def _embed(self, pixel_values: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            summary, _features = self.model(pixel_values)
        return _l2_normalize(summary)

    def to_vectors_batch(self, images) -> np.ndarray:
        # Variable resolution (aspect ratio kept): one image at a time, each snapped to
        # the nearest resolution the model supports.
        vectors = []
        for image in images:
            pixel_values = self.processor(
                images=image, return_tensors="pt", do_resize=True
            ).pixel_values.to(self.device)
            height, width = pixel_values.shape[-2:]
            nearest = self.model.get_nearest_supported_resolution(height, width)
            if (nearest.height, nearest.width) != (height, width):
                pixel_values = torch.nn.functional.interpolate(
                    pixel_values, size=(nearest.height, nearest.width),
                    mode="bilinear", align_corners=False
                )
            vectors.append(self._embed(pixel_values)[0])
        return np.stack(vectors)

    def forward_from_arrays(self, arrays: list[np.ndarray]) -> np.ndarray:
        return self._embed(self._normalize_batch(arrays))


# ---------------------------------------------------------------------------
# Apple MobileCLIP
# ---------------------------------------------------------------------------

class MobileClipTransformer(Transformer):
    """Apple MobileCLIP / MobileCLIP2 vision-language models, loaded through open_clip.
    See: https://huggingface.co/apple/MobileCLIP2-S2
    """
    max_text_sim = 0.375

    # open_clip pretrained tags per MobileCLIP architecture
    _PRETRAINED_TAGS = {
        "MobileCLIP2-S0": "dfndr2b",
        "MobileCLIP2-S2": "dfndr2b",
        "MobileCLIP2-S3": "dfndr2b",
        "MobileCLIP2-S4": "dfndr2b",
        "MobileCLIP2-B": "dfndr2b",
        "MobileCLIP2-L-14": "dfndr2b",
        "MobileCLIP-S1": "datacompdr",
        "MobileCLIP-S2": "datacompdr",
        "MobileCLIP-B": "datacompdr",
    }

    def __init__(self, huggingface_model: str):
        super().__init__(huggingface_model)
        import open_clip
        self.dtype = torch.float32  # open_clip weights stay in full precision

        arch = huggingface_model.split('/')[-1]
        tag = self._PRETRAINED_TAGS.get(arch)
        if tag is None:
            tags = open_clip.list_pretrained_tags_by_model(arch)
            if not tags:
                raise ValueError(f"No open_clip pretrained weights available for {arch}")
            tag = tags[0]

        self.model, _, self.processor = open_clip.create_model_and_transforms(arch, pretrained=tag)
        # eval() is required: MobileCLIP relies on batchnorm layers.
        self.model = self.model.to(self.device).eval()

        # Fuse MobileCLIP's train-time parallel conv+BN branches into single convs:
        # identical embeddings, ~1.3x faster inference on CPU. One-time cost at load.
        try:
            from timm.utils import reparameterize_model
            self.model = reparameterize_model(self.model)
        except Exception:
            # older timm without the util: the un-fused model is still correct
            pass

        self.tokenizer = open_clip.get_tokenizer(arch)
        self.can_handle_text = True

        cfg = open_clip.get_model_preprocess_cfg(self.model)
        size = cfg['size']
        self.preprocess_size = size if isinstance(size, int) else size[0]
        self._norm_mean = torch.tensor(cfg['mean'], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self._norm_std = torch.tensor(cfg['std'], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    def _encode_image(self, batch: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            features = self.model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().float().numpy()

    def to_vectors_batch(self, images) -> np.ndarray:
        batch = torch.stack([self.processor(image.convert("RGB")) for image in images])
        return self._encode_image(batch.to(self.device))

    def forward_from_arrays(self, arrays: list[np.ndarray]) -> np.ndarray:
        return self._encode_image(self._normalize_batch(arrays))

    def to_text_vector(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().float().numpy().flatten()


# ---------------------------------------------------------------------------

type_to_class_mapping = {
    "mobilenet_v2": MobileNetTransformer,
    "dinov2":       Dinov2Transformer,
    "dinov3":       Dinov3Transformer,
    "dinov3_vit":   Dinov3Transformer,  # model_type of the transformers DINOv3 ViT configs
    "siglip2":      SIGLIPTransformer,
    "siglip":       SIGLIPTransformer,
    "clip":         CLIPTransformer,
    "radio":        RadioTransformer,
    "mobileclip2":  MobileClipTransformer,
    "mobileclip":   MobileClipTransformer,
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

    def clear(self) -> None:
        """Drop every loaded model and hand the freed memory back to the GPU."""
        with self._lock:
            self.transformers.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

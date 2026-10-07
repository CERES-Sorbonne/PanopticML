"""Compute engines: the parts of the machine that turn preprocessed images into vectors.

Every model has its PyTorch engine (GPU / CPU). On Apple Silicon, some models get more:
  - CLIP (ViT): the Neural Engine (Core ML), next to PyTorch on the GPU
    (B/32 on an M4 Pro: GPU ~600 img/s + Neural Engine ~800 img/s)
  - SigLIP 2 NaFlex: the Neural Engine, next to PyTorch on the GPU
    (so400m on an M4 Pro: GPU ~27 img/s + Neural Engine ~40 img/s)
  - EmbeddingGemma 2: MLX replaces PyTorch for the vision tower on the GPU (~2x), plus
    the Neural Engine (GPU ~5.5 img/s + Neural Engine ~2.7 img/s, vs 2.3 for PyTorch fp32)
The Neural Engine and the GPU are separate hardware: a compute task runs one thread per
engine on a shared queue of batches, so each takes work at its own pace.

Stability: an extra engine is used only once its vectors match PyTorch's on probe images.
If it fails or returns non-finite values later, the batch is recomputed with PyTorch and
the engine is switched off until Panoptic restarts. Core ML runs in a worker process.
The Neural Engine computes in fp16: its vectors are slightly less precise than the GPU's
(cosine ~0.97-0.9998 vs fp32 on photos, against > 0.9998 for the GPU); the plugin's
`neural_engine` setting turns it off. Switches: PANOPTICML_ANE=0 / PANOPTICML_MLX=0 turn an
engine off; PANOPTICML_CACHE sets where converted Core ML models are stored (default
~/.cache/panopticml).
"""
from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .ane_models import ANE_MODELS_VERSION, CLIP_KIND, GEMMA_KIND, SIGLIP_KIND

if TYPE_CHECKING:
    from ..transformer import Transformer

WORKER = Path(__file__).with_name('coreml_worker.py')
MIN_PROBE_COSINE = 0.99      # Neural Engine fp16 gives ~0.998 vs fp32, broken conversions ~0.6
ANE_BATCH = {CLIP_KIND: 32, SIGLIP_KIND: 4, GEMMA_KIND: 1}   # fastest Core ML batch on the Neural Engine
CONVERT_TIMEOUT = 1800
_build_lock = threading.Lock()


def _log(message: str) -> None:
    print(f"[PanopticML] {message}", flush=True)


def _enabled(flag: str) -> bool:
    return os.environ.get(flag, '1').strip().lower() not in ('0', 'false', 'no', 'off')


def cache_dir() -> Path:
    return Path(os.environ.get('PANOPTICML_CACHE') or Path.home() / '.cache' / 'panopticml')


def probe_images(size: int, n: int = 4) -> list[np.ndarray]:
    """Deterministic test images (colour waves + noise) to check an engine against PyTorch."""
    rng = np.random.default_rng(1234)
    yy, xx = np.mgrid[0:size, 0:size] / size
    images = []
    for _ in range(n):
        f, phase = rng.uniform(1, 6, 3), rng.uniform(0, 2 * np.pi, 3)
        img = np.stack([0.5 + 0.5 * np.sin(2 * np.pi * (f[c] * xx + f[(c + 1) % 3] * yy) + phase[c])
                        for c in range(3)], axis=-1)
        img += rng.normal(0, 0.05, img.shape)
        images.append((np.clip(img, 0, 1) * 255).astype(np.uint8))
    return images


def _check_finite(vectors: np.ndarray, name: str) -> np.ndarray:
    if not np.isfinite(vectors).all():
        raise FloatingPointError(f"{name} returned non-finite values")
    return vectors


def _min_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))).min())


# ---------------------------------------------------------------------------

class Engine:
    name = 'engine'

    def __init__(self, transformer: Transformer):
        self.transformer = transformer
        self.batch_size = transformer.batch_size
        self.ready = threading.Event()
        self.failed = False

    def forward(self, arrays: list[np.ndarray]) -> np.ndarray:
        """L2-normalized vectors of preresized uint8 (H, W, 3) arrays."""
        raise NotImplementedError

    def validate(self, reference: Engine) -> None:
        probe = probe_images(self.transformer.preprocess_size)
        expected = reference.forward(probe)
        cos = _min_cosine(_check_finite(self.forward(probe), self.name), expected)
        if cos < MIN_PROBE_COSINE:
            raise ValueError(f"vectors differ from PyTorch's (cosine {cos:.4f} < {MIN_PROBE_COSINE})")
        _log(f"{self.transformer.name}: {self.name} engine ready (cosine vs PyTorch {cos:.4f})")

    def disable(self, reason: str) -> None:
        if not self.failed:
            self.failed = True
            _log(f"{self.transformer.name}: {self.name} engine switched off: {reason}")
        self.close()

    def close(self) -> None:
        pass


class TorchEngine(Engine):
    """The model's own PyTorch path (GPU or CPU). Always available."""

    def __init__(self, transformer: Transformer):
        super().__init__(transformer)
        self.name = f"torch-{transformer.device}"
        self.ready.set()

    def forward(self, arrays):
        t = self.transformer
        with t.device_lock:
            return np.concatenate([t.forward_from_arrays(arrays[i:i + self.batch_size])
                                   for i in range(0, len(arrays), self.batch_size)])


class MLXGemmaEngine(Engine):
    """EmbeddingGemma 2 on the GPU: vision tower on MLX, text model on PyTorch."""
    name = 'mlx-gpu'

    def __init__(self, transformer):
        super().__init__(transformer)
        from .mlx_gemma import MLXGemmaVision
        with transformer.device_lock:
            self.vision = MLXGemmaVision(transformer.model, transformer.vision_tables)

    def forward(self, arrays):
        t = self.transformer
        out = []
        for i in range(0, len(arrays), self.batch_size):
            soft = self.vision(t.image_patches(arrays[i:i + self.batch_size]))
            out.append(t.soft_tokens_to_vectors(soft))
        return _check_finite(np.concatenate(out), self.name)


class CoreMLEngine(Engine):
    """A Neural Engine model in a Core ML worker process. Converted from the PyTorch weights
    on first use (a few minutes, in the background) and cached as a compiled model."""
    name = 'coreml-ane'

    def __init__(self, transformer, kind: str):
        super().__init__(transformer)
        self.kind = kind
        self.model_batch = ANE_BATCH[kind]
        self.proc: subprocess.Popen | None = None
        self._io_lock = threading.Lock()
        self._closed = False

    # -- setup (background thread) -----------------------------------------------------

    def start(self, reference: Engine) -> None:
        threading.Thread(target=self._setup, args=(reference,), name='panopticml-coreml-setup',
                         daemon=True).start()

    def _setup(self, reference: Engine) -> None:
        try:
            model_dir = self._converted_model()
            if self._closed:
                return
            self._spawn(model_dir)
            self.validate(reference)
            self.ready.set()
        except Exception as e:
            self.disable(f"{type(e).__name__}: {e}")

    def _cache_key(self) -> str:
        import coremltools
        t = self.transformer
        revision = _hub_revision(t.name)
        model = re.sub(r'[^A-Za-z0-9._-]+', '_', t.name)
        return (f"{model}-{revision}-{t.preprocess_size}px-b{self.model_batch}"
                f"-v{ANE_MODELS_VERSION}-ct{coremltools.__version__}")

    def _converted_model(self) -> Path:
        root = cache_dir() / 'coreml'
        root.mkdir(parents=True, exist_ok=True)
        target = root / self._cache_key()
        if (target / 'model.mlmodelc').exists():
            return target
        with _file_lock(root / f"{target.name}.lock"):      # one conversion across processes
            if (target / 'model.mlmodelc').exists():
                return target
            _log(f"{self.transformer.name}: converting for the Neural Engine (once per model, "
                 f"a few minutes; vectors are computed on the GPU meanwhile)")
            t0 = time.perf_counter()
            with open(root / 'worker.log', 'ab') as log:
                result = subprocess.run(
                    [sys.executable, str(WORKER), 'convert', self.kind, self.transformer.name,
                     str(self.transformer.preprocess_size), str(self.model_batch), str(target)],
                    stdout=log, stderr=log, timeout=CONVERT_TIMEOUT, env=_worker_env(),
                )
            if result.returncode != 0 or not (target / 'model.mlmodelc').exists():
                raise RuntimeError(f"conversion failed (exit {result.returncode}), see {root / 'worker.log'}")
            _log(f"{self.transformer.name}: Neural Engine model converted in {time.perf_counter() - t0:.0f}s")
        return target

    def _spawn(self, model_dir: Path) -> None:
        from .coreml_worker import read_frame
        log = open(cache_dir() / 'coreml' / 'worker.log', 'ab')
        self.proc = subprocess.Popen(
            [sys.executable, str(WORKER), 'serve', str(model_dir)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, env=_worker_env(),
        )
        log.close()
        header, _ = read_frame(self.proc.stdout)
        if not header.get('ready'):
            raise RuntimeError(f"worker failed to start: {header}")

    # -- compute -------------------------------------------------------------------------

    def _predict(self, x: np.ndarray) -> np.ndarray:
        """Run the fixed-batch model on any number of inputs (last chunk padded)."""
        from .coreml_worker import read_frame, write_frame
        n, b = len(x), self.model_batch
        out = []
        with self._io_lock:
            if self.proc is None or self.proc.poll() is not None:
                raise RuntimeError("Core ML worker is not running")
            for i in range(0, n, b):
                chunk = x[i:i + b]
                if len(chunk) < b:
                    chunk = np.concatenate([chunk, np.repeat(chunk[-1:], b - len(chunk), axis=0)])
                write_frame(self.proc.stdin, {}, chunk)
                header, y = read_frame(self.proc.stdout)
                if 'error' in header:
                    raise RuntimeError(f"Core ML: {header['error']}")
                out.append(y)
        return np.concatenate(out)[:n]

    def forward(self, arrays):
        t = self.transformer
        if self.kind == CLIP_KIND:
            y = self._predict(np.stack(arrays).transpose(0, 3, 1, 2))    # uint8: the worker casts
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
        elif self.kind == SIGLIP_KIND:
            patches = t.processor(images=list(arrays), return_tensors="np")['pixel_values']   # (B, S, D)
            y = self._predict(patches.transpose(0, 2, 1)[:, :, None, :].astype(np.float32))
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
        else:
            patches = t.image_patches(arrays)                                                 # (B, P, D)
            soft = self._predict(patches.transpose(0, 2, 1)[:, :, None, :].astype(np.float32))
            y = t.soft_tokens_to_vectors(soft)
        return _check_finite(y, self.name)

    def close(self) -> None:
        self._closed = True
        proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            proc.stdin.close()               # the worker exits on EOF
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


def _hub_revision(model_id: str) -> str:
    """Commit of the cached HuggingFace snapshot: a model update gets a new Core ML model."""
    try:
        from huggingface_hub import try_to_load_from_cache
        path = try_to_load_from_cache(model_id, 'config.json')
        if isinstance(path, str):
            return Path(path).parent.name[:12]
    except Exception:
        pass
    return 'local'


@contextmanager
def _file_lock(path: Path):
    import fcntl  # Unix only: this module is imported on Windows too, Core ML only runs on macOS
    with open(path, 'w') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _worker_env() -> dict:
    return {**os.environ, 'TQDM_DISABLE': '1', 'PYTHONUNBUFFERED': '1'}


# ---------------------------------------------------------------------------

def _apple_silicon() -> bool:
    return sys.platform == 'darwin' and platform.machine() == 'arm64'


def _ane_kind(transformer) -> str | None:
    from ..transformer import CLIPTransformer, EmbeddingGemma2Transformer, SIGLIPTransformer
    if isinstance(transformer, EmbeddingGemma2Transformer):
        return GEMMA_KIND
    if isinstance(transformer, CLIPTransformer) and not transformer.patch_sequence:
        vision = getattr(transformer.model.config, 'vision_config', None)
        # OpenAI CLIP ViTs; other checkpoints (e.g. gelu ones) stay on the GPU
        if (getattr(vision, 'hidden_act', None) == 'quick_gelu'
                and getattr(vision, 'image_size', None) == transformer.preprocess_size):
            return CLIP_KIND
    if (isinstance(transformer, SIGLIPTransformer) and getattr(transformer, 'patch_sequence', False)
            and getattr(transformer.model.config, 'model_type', '') == 'siglip2'):
        return SIGLIP_KIND
    return None


def get_engines(transformer: Transformer, neural_engine: bool = True) -> list[Engine]:
    """The model's compute engines, primary (GPU) first. Built on first call; the Neural
    Engine one may still be getting ready (its `ready` event) or have been switched off."""
    with _build_lock:
        if transformer.engines is None:
            transformer.engines = _build(transformer)
        elif transformer.engines[0].failed:      # MLX switched off: PyTorch takes the GPU back
            transformer.engines[0] = TorchEngine(transformer)
        has_ane = any(isinstance(e, CoreMLEngine) for e in transformer.engines)
        if neural_engine and not has_ane:
            ane = _neural_engine(transformer)
            if ane is not None:
                transformer.engines.append(ane)
        elif not neural_engine and has_ane:      # turned off in the plugin settings
            for e in [e for e in transformer.engines if isinstance(e, CoreMLEngine)]:
                e.close()
                transformer.engines.remove(e)
        return [e for e in transformer.engines if not e.failed]


def _build(transformer) -> list[Engine]:
    torch_engine = TorchEngine(transformer)
    if not _apple_silicon() or transformer.device != 'mps':
        return [torch_engine]
    if _ane_kind(transformer) == GEMMA_KIND and _enabled('PANOPTICML_MLX'):
        try:
            mlx_engine = MLXGemmaEngine(transformer)
            mlx_engine.validate(torch_engine)
            mlx_engine.ready.set()
            return [mlx_engine]
        except ImportError:
            pass                         # MLX not installed: PyTorch stays on the GPU
        except Exception as e:
            _log(f"{transformer.name}: MLX engine unavailable, using PyTorch: {type(e).__name__}: {e}\n"
                 f"{traceback.format_exc()}")
    return [torch_engine]


def _neural_engine(transformer) -> CoreMLEngine | None:
    """A Core ML engine getting ready in the background, if the model and machine have one."""
    if not _apple_silicon() or transformer.device != 'mps' or not _enabled('PANOPTICML_ANE'):
        return None
    kind = _ane_kind(transformer)
    if kind is None:
        return None
    try:
        import coremltools  # noqa: F401
    except ImportError:
        return None
    engine = CoreMLEngine(transformer, kind)
    engine.start(reference=TorchEngine(transformer))
    return engine


def fallback_engine(transformer: Transformer) -> Engine:
    """Where a batch goes when another engine fails: the model's PyTorch path."""
    return TorchEngine(transformer)


def close_engines(transformer: Transformer) -> None:
    for engine in transformer.engines or []:
        engine.close()
    transformer.engines = None

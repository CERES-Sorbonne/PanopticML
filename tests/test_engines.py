"""Compute engines (compute/accel) and the multi-engine compute task.

Model tests follow PANOPTICML_TEST_MODELS like test_transformers.py. The Core ML test
converts a model for the Neural Engine (minutes on first run): opt in with PANOPTICML_TEST_ANE=1.
"""
import hashlib
import io
import os
import platform
import sys
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch  # noqa: F401  before faiss, see panopticml/compute/__init__.py
from PIL import Image

from panopticml.compute.accel import engines as E
from panopticml.compute_vector_task import ComputeVectorsTask, _preprocess_worker
from panopticml.panoptic_ml import ModelEnum

APPLE_SILICON = sys.platform == 'darwin' and platform.machine() == 'arm64'
RESOURCES = os.path.join(os.path.dirname(__file__), 'resources')


def _selected(name: str) -> bool:
    selection = os.environ.get('PANOPTICML_TEST_MODELS', '').strip()
    return not selection or selection == 'all' or name in [s.strip() for s in selection.split(',')]


def _photos(size: int) -> list[np.ndarray]:
    return [np.asarray(Image.open(os.path.join(RESOURCES, f)).convert('RGB').resize((size, size), Image.BICUBIC))
            for f in sorted(os.listdir(RESOURCES))]


def _min_cos(a, b) -> float:
    return E._min_cosine(np.asarray(a), np.asarray(b))


# ---------------------------------------------------------------------------
# Multi-engine task, with fake engines (no model)
# ---------------------------------------------------------------------------

class FakeTransformer:
    """forward_from_arrays: the mean colour of each image, as a vector."""
    name, device, preprocess_size, batch_size = 'fake', 'cpu', 8, 4

    def __init__(self):
        self.device_lock = threading.RLock()
        self.engines = None

    def forward_from_arrays(self, arrays):
        return np.stack([a.reshape(-1, 3).mean(0) + 1.0 for a in arrays]).astype(np.float32)


class FakeEngine(E.Engine):
    def __init__(self, transformer, name, fail_after=None, ready=True):
        super().__init__(transformer)
        self.name, self.fail_after, self.calls = name, fail_after, 0
        if ready:
            self.ready.set()

    def forward(self, arrays):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("engine broke")
        return self.transformer.forward_from_arrays(arrays)


class FakeProject:
    def __init__(self, n):
        self.images, self.vectors = {}, {}
        rng = np.random.default_rng(0)
        for _ in range(n):
            buf = io.BytesIO()
            Image.fromarray(rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)).save(buf, 'PNG')
            self.images[hashlib.sha1(buf.getvalue()).hexdigest()] = buf.getvalue()

    @contextmanager
    def _media_db(self):
        project = self

        class DB:
            conn = SimpleNamespace(execute=lambda *a: SimpleNamespace(fetchall=lambda: []))

            def get_image_types(self):
                return [SimpleNamespace(id=1, width=16, height=16)]

            def get_images(self, type_id, sha1):
                return [SimpleNamespace(sha1=s, data=project.images[s]) for s in sha1 if s in project.images]
        yield DB()

    def upsert_vectors(self, vectors):
        for v in vectors:
            self.vectors[v.sha1] = v.data


def _make_task(monkeypatch, transformer, engines, n=40, vec_type_id=1):
    transformer.engines = engines
    monkeypatch.setattr('panopticml.compute_vector_task.get_engines',
                        lambda t, neural_engine=True: [e for e in t.engines if not e.failed])
    project = FakeProject(n)
    plugin = SimpleNamespace(project=project, rebuild_index=lambda vt: None,
                             params=SimpleNamespace(neural_engine=True),
                             transformers=SimpleNamespace(get=lambda vt: transformer))
    task = ComputeVectorsTask(plugin, SimpleNamespace(id=vec_type_id, params={'model': 'fake', 'greyscale': False}),
                              [SimpleNamespace(sha1=s) for s in project.images])
    return task, project


def _run_task(monkeypatch, transformer, engines, n=40):
    task, project = _make_task(monkeypatch, transformer, engines, n)
    task.start()
    return task, project


def test_task_is_followed_from_its_vector_type(monkeypatch):
    """Panoptic's vector settings find the running task (progress bar, live counts) by the
    vector type id on its state."""
    from panoptic.core.task.task_manager import TaskManager
    t = FakeTransformer()
    task, project = _make_task(monkeypatch, t, [FakeEngine(t, 'gpu')], n=8, vec_type_id=7)
    manager = TaskManager()
    try:
        queued = manager.add_task(task)
        assert queued.id is not None and queued.state.vector_type_id == 7
        assert task._finished_event.wait(30)
    finally:
        manager.close()
    assert len(project.vectors) == 8 and task.state.done == 8


def test_engines_share_the_work(monkeypatch):
    t = FakeTransformer()
    a, b = FakeEngine(t, 'gpu'), FakeEngine(t, 'ane')
    task, project = _run_task(monkeypatch, t, [a, b])
    assert len(project.vectors) == 40 and task.state.failed == 0
    assert a.calls + b.calls == 10


def test_failing_extra_engine_falls_back_to_torch(monkeypatch):
    t = FakeTransformer()
    gpu, ane = FakeEngine(t, 'gpu'), FakeEngine(t, 'ane', fail_after=1)
    task, project = _run_task(monkeypatch, t, [gpu, ane])
    assert len(project.vectors) == 40 and task.state.failed == 0
    assert ane.failed and ane.calls == 2
    # the batch the engine failed on was computed with PyTorch, right
    for sha1, vec in project.vectors.items():
        _, arr, _ = _preprocess_worker((sha1, project.images[sha1], t.preprocess_size, False))
        np.testing.assert_allclose(vec, t.forward_from_arrays([arr])[0], rtol=1e-6)


def test_failing_primary_engine_is_replaced(monkeypatch):
    t = FakeTransformer()
    mlx = FakeEngine(t, 'mlx', fail_after=2)
    task, project = _run_task(monkeypatch, t, [mlx])
    assert len(project.vectors) == 40 and task.state.failed == 0 and mlx.failed


def test_engine_never_ready_does_not_block(monkeypatch):
    t = FakeTransformer()
    gpu, converting = FakeEngine(t, 'gpu'), FakeEngine(t, 'ane', ready=False)
    task, project = _run_task(monkeypatch, t, [gpu, converting])
    assert len(project.vectors) == 40 and converting.calls == 0


def test_non_finite_vectors_are_rejected():
    with pytest.raises(FloatingPointError):
        E._check_finite(np.array([[1.0, np.nan]]), 'x')


# ---------------------------------------------------------------------------
# Neural Engine rewrites (fp32, CPU: no coremltools needed)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _selected('clip'), reason="clip not selected")
def test_clip_ane_rewrite_matches_huggingface():
    from panopticml.compute.accel.ane_models import CLIP_KIND, build
    build(CLIP_KIND, ModelEnum.clip.value, 224)           # raises if cosine < 0.9999


@pytest.mark.skipif(not _selected('siglip'), reason="siglip not selected")
def test_siglip_ane_rewrite_matches_huggingface():
    from panopticml.compute.accel.ane_models import SIGLIP_KIND, build
    build(SIGLIP_KIND, ModelEnum.siglip.value, 256)       # raises if cosine < 0.9999


@pytest.mark.skipif(not _selected('embeddinggemma2'), reason="embeddinggemma2 not selected")
def test_gemma_ane_rewrite_matches_huggingface():
    from panopticml.compute.accel.ane_models import GEMMA_KIND, build
    build(GEMMA_KIND, ModelEnum.embeddinggemma2.value, 768)


# ---------------------------------------------------------------------------
# EmbeddingGemma 2 split pipeline and MLX engine
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def gemma():
    if not _selected('embeddinggemma2'):
        pytest.skip("embeddinggemma2 not selected")
    from panopticml.compute.transformer import get_transformer
    t = get_transformer(ModelEnum.embeddinggemma2.value)
    yield t
    E.close_engines(t)


def test_gemma_split_pipeline_matches_forward(gemma):
    """image_patches -> vision tower -> soft_tokens_to_vectors == forward_from_arrays."""
    arrays = _photos(gemma.preprocess_size)
    pixels = torch.from_numpy(gemma.image_patches(arrays)).to(gemma.device, gemma.dtype)
    _, position_ids, _ = __import__('panopticml.compute.accel.ane_models', fromlist=['x']).gemma_patches(
        gemma.processor, arrays[:1])
    with torch.no_grad():
        pos = position_ids[None].expand(len(arrays), -1, -1).to(gemma.device)
        soft = torch.stack(gemma.model.get_image_features(pixels, pos, return_dict=True).pooler_output)
    split = gemma.soft_tokens_to_vectors(soft.float().cpu().numpy())
    assert _min_cos(split, gemma.forward_from_arrays(arrays)) > 0.9999


def test_gemma_trimmed_padding_matches_padded(gemma):
    arrays = _photos(gemma.preprocess_size)
    padded = gemma._embed(gemma.processor(images=[[a] for a in arrays], return_tensors="pt"))
    assert _min_cos(gemma.forward_from_arrays(arrays), padded) > 0.9999


@pytest.mark.skipif(not APPLE_SILICON, reason="MLX needs Apple Silicon")
def test_gemma_mlx_engine_matches_torch(gemma):
    pytest.importorskip('mlx')
    mlx_engine = E.MLXGemmaEngine(gemma)
    arrays = _photos(gemma.preprocess_size)
    assert _min_cos(mlx_engine.forward(arrays), E.TorchEngine(gemma).forward(arrays)) > 0.999


# ---------------------------------------------------------------------------
# Core ML on the Neural Engine (opt in: converts the model, minutes on first run)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (APPLE_SILICON and os.environ.get('PANOPTICML_TEST_ANE') == '1'),
                    reason="set PANOPTICML_TEST_ANE=1 (Apple Silicon) to convert and test Core ML models")
@pytest.mark.parametrize('model', ['clip', 'siglip', 'embeddinggemma2'])
def test_coreml_engine_matches_torch(model):
    if not _selected(model):
        pytest.skip(f"{model} not selected")
    pytest.importorskip('coremltools')
    from panopticml.compute.transformer import get_transformer
    t = get_transformer(ModelEnum[model].value)
    engine = E._neural_engine(t)
    assert engine is not None
    try:
        while not (engine.ready.wait(1) or engine.failed):
            pass
        assert not engine.failed
        arrays = _photos(t.preprocess_size)
        assert _min_cos(engine.forward(arrays), E.TorchEngine(t).forward(arrays)) > 0.99
    finally:
        engine.close()

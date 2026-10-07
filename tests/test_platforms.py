"""Platform support: the plugin loads and computes everywhere, and the Apple Silicon compute
engines (compute/accel) are optional extras that fall back to PyTorch.

- Windows: no Unix-only module (fcntl, ...) may be imported when the plugin loads.
- Install: coremltools / MLX are installed on Apple Silicon only, and only where a wheel
  exists (a failed source build would fail the whole plugin install).
- Runtime: off Apple Silicon, or without coremltools / MLX, or when an engine breaks,
  vectors are computed with PyTorch alone.
"""
import ast
import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import torch  # noqa: F401  before faiss, see panopticml/compute/__init__.py
from packaging.requirements import Requirement

from panopticml.compute.accel import engines as E
from panopticml.compute.accel.ane_models import CLIP_KIND, GEMMA_KIND

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / 'panopticml'

UNIX_ONLY = {'fcntl', 'termios', 'tty', 'pty', 'pwd', 'grp', 'resource', 'posix', 'syslog', 'crypt', 'readline'}
APPLE_ONLY = {'mlx', 'coremltools'}
# imported lazily, only on Apple Silicon (MLXGemmaEngine)
APPLE_ONLY_MODULES = {PACKAGE / 'compute' / 'accel' / 'mlx_gemma.py'}


# ---------------------------------------------------------------------------
# Imports: Windows, and machines without the Apple libraries
# ---------------------------------------------------------------------------

def _module_level_imports(path: Path) -> set[str]:
    """Top-level names imported when the module itself is imported (not inside functions)."""
    names = set()

    def visit(nodes):
        for node in nodes:
            if isinstance(node, ast.Import):
                names.update(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module.split('.')[0])
            elif isinstance(node, (ast.If, ast.Try, ast.With)):
                visit(getattr(node, 'body', []) + getattr(node, 'orelse', [])
                      + getattr(node, 'finalbody', [])
                      + [n for h in getattr(node, 'handlers', []) for n in h.body])
            elif isinstance(node, ast.ClassDef):
                visit(node.body)
    visit(ast.parse(path.read_text(encoding='utf-8')).body)
    return names


@pytest.mark.parametrize('path', sorted(PACKAGE.rglob('*.py')), ids=lambda p: str(p.relative_to(ROOT)))
def test_no_platform_specific_module_level_import(path):
    imported = _module_level_imports(path)
    assert not imported & UNIX_ONLY, f"Unix-only import breaks Windows: {imported & UNIX_ONLY}"
    if path not in APPLE_ONLY_MODULES:
        assert not imported & APPLE_ONLY, f"Apple-only import at module level: {imported & APPLE_ONLY}"


# Run in a fresh interpreter: MLX / coremltools are uninstalled, and the plugin's own
# imports of Unix-only modules fail as on Windows (the libraries it uses keep theirs: on
# Windows they take their own code paths).
_SIMULATE_WINDOWS = """
import builtins, sys
UNIX_ONLY, APPLE_ONLY = {unix!r}, {apple!r}
for name in APPLE_ONLY:
    sys.modules[name] = None

real_import = builtins.__import__

def windows_import(name, globals=None, locals=None, fromlist=(), level=0):
    importer = (globals or {{}}).get('__name__', '')
    if level == 0 and name.split('.')[0] in UNIX_ONLY and importer.split('.')[0] == 'panopticml':
        raise ModuleNotFoundError(f"No module named {{name!r}} (simulated Windows, imported by {{importer}})")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = windows_import
import torch
import panopticml.panoptic_ml, panopticml.compute_vector_task
from panopticml.compute.accel import get_engines, fallback_engine, close_engines
from panopticml.compute.transformer import TransformerManager
TransformerManager().clear()       # imports compute.accel lazily
print('ok')
"""


def test_plugin_loads_without_unix_or_apple_modules():
    code = _SIMULATE_WINDOWS.format(unix=sorted(UNIX_ONLY), apple=sorted(APPLE_ONLY))
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True,
                            encoding='utf-8', errors='replace', timeout=300)
    assert result.returncode == 0 and result.stdout.strip().endswith('ok'), result.stderr[-3000:]


# ---------------------------------------------------------------------------
# Install: platform markers of the Apple Silicon libraries
# ---------------------------------------------------------------------------

def _requirements_txt() -> dict[str, Requirement]:
    lines = (ROOT / 'requirements.txt').read_text(encoding='utf-8').splitlines()
    reqs = [Requirement(line) for line in (l.strip() for l in lines) if line and not line.startswith('#')]
    return {r.name.lower(): r for r in reqs}


def _pyproject() -> dict[str, Requirement]:
    tomllib = pytest.importorskip('tomllib')       # Python 3.11+
    deps = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['project']['dependencies']
    return {r.name.lower(): r for r in map(Requirement, deps)}


def _env(sys_platform, machine, release, python):
    return {'sys_platform': sys_platform, 'platform_machine': machine, 'platform_release': release,
            'python_version': python, 'python_full_version': f"{python}.0",
            'platform_system': {'win32': 'Windows', 'linux': 'Linux', 'darwin': 'Darwin'}[sys_platform],
            'os_name': 'nt' if sys_platform == 'win32' else 'posix', 'implementation_name': 'cpython'}


ENVIRONMENTS = {
    # name: (environment, apple libraries installed)
    'windows':              (_env('win32', 'AMD64', '10', '3.12'), False),
    'windows-arm':          (_env('win32', 'ARM64', '11', '3.13'), False),
    'linux':                (_env('linux', 'x86_64', '6.8.0', '3.12'), False),
    'linux-arm':            (_env('linux', 'aarch64', '6.8.0', '3.12'), False),
    'intel-mac':            (_env('darwin', 'x86_64', '23.6.0', '3.12'), False),
    'apple-silicon':        (_env('darwin', 'arm64', '24.6.0', '3.12'), True),
}


@pytest.mark.parametrize('source', [_requirements_txt, _pyproject], ids=['requirements.txt', 'pyproject'])
@pytest.mark.parametrize('name', ENVIRONMENTS)
def test_apple_libraries_only_on_apple_silicon(source, name):
    env, expected = ENVIRONMENTS[name]
    reqs = source()
    for lib in APPLE_ONLY:
        assert lib in reqs, f"{lib} missing"
        assert reqs[lib].marker is not None and reqs[lib].marker.evaluate(env) == expected, (lib, name)


@pytest.mark.parametrize('source', [_requirements_txt, _pyproject], ids=['requirements.txt', 'pyproject'])
def test_apple_libraries_skipped_where_no_wheel_exists(source):
    reqs = source()
    # MLX has wheels for macOS 14+ (Darwin 23+) only, and no source distribution
    for release in ('20.6.0', '21.6.0', '22.6.0'):
        assert not reqs['mlx'].marker.evaluate(_env('darwin', 'arm64', release, '3.12'))
    for release in ('23.0.0', '24.6.0', '25.0.0'):
        assert reqs['mlx'].marker.evaluate(_env('darwin', 'arm64', release, '3.12'))
    # markers are not short-circuited: a platform_release version comparison crashes on e.g. Fedora kernels
    assert not reqs['mlx'].marker.evaluate(_env('linux', 'x86_64', '7.2.5-100.fc43.x86_64', '3.12'))
    # coremltools 9 has no Python 3.14 wheel: a source build would fail the install
    assert not reqs['coremltools'].marker.evaluate(_env('darwin', 'arm64', '24.6.0', '3.14'))


def test_requirements_txt_matches_pyproject():
    """Panoptic installs path / git plugins from requirements.txt, PyPI ones from pyproject."""
    txt, pyproject = _requirements_txt(), _pyproject()
    assert txt.keys() == pyproject.keys()
    for name in txt:
        assert txt[name].specifier == pyproject[name].specifier, name
        assert str(txt[name].marker) == str(pyproject[name].marker), name


# ---------------------------------------------------------------------------
# Runtime: engine selection and fallbacks (fake model, no download)
# ---------------------------------------------------------------------------

class FakeTransformer:
    name, preprocess_size, batch_size = 'fake', 8, 4

    def __init__(self, device):
        self.device = device
        self.device_lock = threading.RLock()
        self.engines = None

    def forward_from_arrays(self, arrays):
        return np.stack([a.reshape(-1, 3).mean(0) + 1.0 for a in arrays]).astype(np.float32)


@pytest.fixture
def apple_silicon(monkeypatch):
    """Pretend to run on an Apple Silicon Mac whose model is one with extra engines."""
    def setup(kind):
        monkeypatch.setattr(E, '_apple_silicon', lambda: True)
        monkeypatch.setattr(E, '_ane_kind', lambda t: kind)
        return FakeTransformer('mps')
    return setup


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_only_torch_off_apple_silicon(monkeypatch, device):
    monkeypatch.setattr(E, '_apple_silicon', lambda: False)
    t = FakeTransformer(device)
    engines = E.get_engines(t, neural_engine=True)
    assert [type(e) for e in engines] == [E.TorchEngine]
    assert engines[0].ready.is_set() and engines[0].name == f"torch-{device}"


def test_only_torch_on_apple_silicon_cpu(apple_silicon):
    """PANOPTICML_DEVICE=cpu on a Mac: the extra engines need the GPU path as reference."""
    t = apple_silicon(GEMMA_KIND)
    t.device = 'cpu'
    assert [type(e) for e in E.get_engines(t)] == [E.TorchEngine]


def test_torch_when_mlx_is_not_installed(apple_silicon, monkeypatch):
    t = apple_silicon(GEMMA_KIND)
    for name in ('mlx', 'mlx.core', 'mlx.nn', 'panopticml.compute.accel.mlx_gemma'):
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, 'coremltools', None)
    assert [type(e) for e in E.get_engines(t)] == [E.TorchEngine]


def test_no_neural_engine_when_coremltools_is_not_installed(apple_silicon, monkeypatch):
    t = apple_silicon(CLIP_KIND)
    monkeypatch.setitem(sys.modules, 'coremltools', None)
    assert [type(e) for e in E.get_engines(t, neural_engine=True)] == [E.TorchEngine]


def test_torch_when_mlx_fails_validation(apple_silicon, monkeypatch):
    class BrokenMLX(E.Engine):
        name = 'mlx-gpu'

        def forward(self, arrays):
            raise RuntimeError("Metal device not available")

    t = apple_silicon(GEMMA_KIND)
    monkeypatch.setattr(E, 'MLXGemmaEngine', BrokenMLX)
    monkeypatch.setitem(sys.modules, 'coremltools', None)
    assert [type(e) for e in E.get_engines(t)] == [E.TorchEngine]


def test_torch_when_mlx_import_crashes(apple_silicon, monkeypatch):
    """Installed but unusable (e.g. a broken Metal install): any error falls back, not only ImportError."""
    class CrashingMLX(E.Engine):
        def __init__(self, transformer):
            raise OSError("libmlx.dylib: image not found")

    t = apple_silicon(GEMMA_KIND)
    monkeypatch.setattr(E, 'MLXGemmaEngine', CrashingMLX)
    monkeypatch.setitem(sys.modules, 'coremltools', None)
    assert [type(e) for e in E.get_engines(t)] == [E.TorchEngine]


def test_mlx_switch(apple_silicon, monkeypatch):
    monkeypatch.setenv('PANOPTICML_MLX', '0')
    monkeypatch.setitem(sys.modules, 'coremltools', None)
    monkeypatch.setattr(E, 'MLXGemmaEngine', lambda t: pytest.fail("MLX used despite PANOPTICML_MLX=0"))
    t = apple_silicon(GEMMA_KIND)
    assert [type(e) for e in E.get_engines(t)] == [E.TorchEngine]


def test_neural_engine_switch(apple_silicon, monkeypatch):
    monkeypatch.setenv('PANOPTICML_ANE', '0')
    monkeypatch.setattr(E.CoreMLEngine, '__init__', lambda self, t, k: pytest.fail("Core ML used despite PANOPTICML_ANE=0"))
    t = apple_silicon(CLIP_KIND)
    assert [type(e) for e in E.get_engines(t, neural_engine=True)] == [E.TorchEngine]


def test_neural_engine_setting_off(apple_silicon, monkeypatch):
    monkeypatch.setattr(E.CoreMLEngine, '__init__', lambda self, t, k: pytest.fail("Core ML used despite neural_engine=False"))
    t = apple_silicon(CLIP_KIND)
    assert [type(e) for e in E.get_engines(t, neural_engine=False)] == [E.TorchEngine]


def test_neural_engine_conversion_failure_switches_it_off(apple_silicon, monkeypatch, tmp_path):
    """The Core ML conversion fails (no Neural Engine, coremltools error...): the engine is
    switched off and only PyTorch remains."""
    pytest.importorskip('coremltools')
    def conversion_fails(self):
        raise RuntimeError("conversion failed")

    monkeypatch.setenv('PANOPTICML_CACHE', str(tmp_path))
    monkeypatch.setattr(E.CoreMLEngine, '_converted_model', conversion_fails)
    t = apple_silicon(CLIP_KIND)
    assert isinstance(E.get_engines(t, neural_engine=True)[0], E.TorchEngine)
    ane = next(e for e in t.engines if isinstance(e, E.CoreMLEngine))
    for _ in range(100):
        if ane.failed:
            break
        ane.ready.wait(0.05)
    assert ane.failed and not ane.ready.is_set()
    assert [type(e) for e in E.get_engines(t, neural_engine=True)] == [E.TorchEngine]


def test_file_lock(tmp_path):
    """Only used on macOS (Core ML conversion), but must not break elsewhere when unused."""
    if sys.platform == 'win32':
        pytest.skip("Core ML conversion lock: macOS only")
    with E._file_lock(tmp_path / 'x.lock'):
        pass

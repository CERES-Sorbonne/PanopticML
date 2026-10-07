import importlib.util
import os
import sys
from pathlib import Path

# Avoid segfaults on macOS: faiss and torch each bundle their own libomp, and two OpenMP runtimes
# in one process crash. DYLD_LIBRARY_PATH on torch/lib makes faiss load torch's copy; dyld only
# reads it at process start, hence the re-exec (same as panoptic/macos.py in Panoptic).
if sys.platform == 'darwin' and not os.environ.get('PANOPTICML_DYLD_REEXEC'):
    _torch = importlib.util.find_spec('torch')
    if _torch and _torch.origin:
        _torch_lib = str(Path(_torch.origin).parent / 'lib')
        _current = [p for p in os.environ.get('DYLD_LIBRARY_PATH', '').split(os.pathsep) if p]
        if _torch_lib not in _current:
            os.environ['DYLD_LIBRARY_PATH'] = os.pathsep.join([_torch_lib] + _current)
            os.environ['PANOPTICML_DYLD_REEXEC'] = '1'  # SIP may strip DYLD_* on exec: never loop
            sys.stdout.flush()
            sys.stderr.flush()
            os.execv(sys.executable, sys.orig_argv)

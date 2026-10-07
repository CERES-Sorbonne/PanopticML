"""Core ML worker process (macOS, Apple Silicon).

Core ML runs here rather than in Panoptic's process: coremltools' predict() holds the GIL
for the whole call, which starves the PyTorch / MLX threads computing on the GPU at the
same time, and a crash in Core ML only takes this process down. Standalone on purpose (it
must not import the plugin package): ane_models.py is loaded by path.

  python coreml_worker.py convert KIND MODEL_ID SIZE BATCH OUT_DIR
      trace + convert the Neural Engine model and write the compiled model to OUT_DIR
  python coreml_worker.py serve MODEL_DIR
      load the compiled model, then answer frames on stdin / stdout:
      <u32 header length><JSON header {"shape", "dtype"}><raw array bytes>
"""
import importlib.util
import json
import os
import shutil
import struct
import sys
import tempfile
from pathlib import Path

INPUT, OUTPUT = 'x', 'y'


def _ane_models():
    spec = importlib.util.spec_from_file_location(
        'panopticml_ane_models', Path(__file__).with_name('ane_models.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def convert(kind: str, model_id: str, size: int, batch: int, out_dir: str) -> None:
    import coremltools as ct
    import numpy as np
    import torch

    module, example = _ane_models().build(kind, model_id, size)
    example = example[:1].expand(batch, *example.shape[1:]).contiguous()
    with torch.no_grad():
        traced = torch.jit.trace(module, example)
    mlmodel = ct.convert(
        traced, inputs=[ct.TensorType(name=INPUT, shape=tuple(example.shape))],
        outputs=[ct.TensorType(name=OUTPUT)], compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15, convert_to="mlprogram",
    )
    out = Path(out_dir)
    tmp = Path(tempfile.mkdtemp(prefix=out.name + '.', dir=out.parent))
    try:
        package = tmp / 'model.mlpackage'
        mlmodel.save(str(package))
        ct.utils.compile_model(str(package), str(tmp / 'model.mlmodelc'))
        shutil.rmtree(package)
        os.replace(tmp, out)                 # atomic: a half-written cache is never used
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # First Neural Engine load compiles for the device (seconds to a minute); macOS caches
    # that per model path, so do it now rather than on the first compute task.
    model = ct.models.CompiledMLModel(str(out / 'model.mlmodelc'), compute_units=ct.ComputeUnit.CPU_AND_NE)
    model.predict({INPUT: example.numpy().astype(np.float32)})


def _read_exact(stream, n: int) -> bytes:
    data = stream.read(n)
    if data is None or len(data) < n:
        raise EOFError
    return data


def read_frame(stream):
    import numpy as np
    (n,) = struct.unpack('<I', _read_exact(stream, 4))
    header = json.loads(_read_exact(stream, n))
    if 'shape' not in header:
        return header, None
    dtype = np.dtype(header['dtype'])
    size = int(np.prod(header['shape'])) * dtype.itemsize
    return header, np.frombuffer(_read_exact(stream, size), dtype=dtype).reshape(header['shape'])


def write_frame(stream, header: dict, array=None) -> None:
    import numpy as np
    if array is not None:
        array = np.ascontiguousarray(array)
        header = {**header, 'shape': list(array.shape), 'dtype': array.dtype.str}
    data = json.dumps(header).encode()
    stream.write(struct.pack('<I', len(data)) + data)
    if array is not None:
        stream.write(array.tobytes())
    stream.flush()


def serve(model_dir: str) -> None:
    # frames go to the real stdout; anything printed (by us or by native code) goes to stderr
    out = os.fdopen(os.dup(1), 'wb')
    os.dup2(2, 1)
    inp = sys.stdin.buffer
    import coremltools as ct
    import numpy as np
    model = ct.models.CompiledMLModel(str(Path(model_dir) / 'model.mlmodelc'),
                                      compute_units=ct.ComputeUnit.CPU_AND_NE)
    write_frame(out, {'ready': True})
    while True:
        try:
            _, x = read_frame(inp)
        except EOFError:
            return                            # parent closed the pipe (or exited)
        try:
            y = model.predict({INPUT: np.array(x, dtype=np.float32)})[OUTPUT]   # writable copy
            write_frame(out, {}, np.asarray(y, dtype=np.float32))
        except Exception as e:
            write_frame(out, {'error': f"{type(e).__name__}: {e}"})


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'convert':
        convert(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
    elif mode == 'serve':
        serve(sys.argv[2])
    else:
        raise SystemExit(f"unknown mode {mode!r}")

"""Temporary diagnostic: why do CLIP / SigLIP give collapsed vectors on macOS runners?

Runs CLIP image + text embeddings under several configurations and prints, for each,
the image/image similarity spread and whether text->image retrieval is right.
Usage: python diagnose_macos.py [--no-faiss] [config ...]
"""
import os
import pathlib
import sys
import types

if '--no-faiss' in sys.argv:
    # panopticml.compute imports faiss at module level; stub it to test without its libomp
    import importlib.machinery
    stub = types.ModuleType('faiss')
    stub.__spec__ = importlib.machinery.ModuleSpec('faiss', None)
    stub.IndexFlatIP = object
    sys.modules['faiss'] = stub

import numpy as np
import torch

import panopticml.compute.transformer as T
from panopticml.utils import preprocess_image

print(f"torch {torch.__version__} | mps available={torch.backends.mps.is_available()} "
      f"built={torch.backends.mps.is_built()} | faiss loaded={'faiss' in sys.modules and hasattr(sys.modules['faiss'], 'normalize_L2')}")
print(f"resolved device: {T.resolve_device()} | torch threads={torch.get_num_threads()}")
print(torch.__config__.parallel_info())

MODEL = 'openai/clip-vit-base-patch32'
RES = pathlib.Path(__file__).resolve().parent.parent / 'tests' / 'resources'
IMAGES = ['bird.gif', 'chat.png', 'dog.jpg', 'spider.jpg']
TEXTS = {'A bird': 'bird.gif', 'A small grey cat': 'chat.png', 'A happy dog': 'dog.jpg', 'A jumping spider': 'spider.jpg'}


def run(label, device, threads=None, attn=None):
    if threads:
        torch.set_num_threads(threads)
    T.resolve_device = lambda: device
    orig = T.from_pretrained
    if attn:
        T.from_pretrained = lambda loader, name, **kw: orig(loader, name, **({**kw, 'attn_implementation': attn} if 'device_map' in kw else kw))
    try:
        t = T.get_transformer(MODEL)
        imgs = [preprocess_image((RES / n).read_bytes(), {}) for n in IMAGES]
        v = np.stack([t.to_vector(i) for i in imgs])
        sims = v @ v.T
        off = sims[~np.eye(len(IMAGES), dtype=bool)]
        ok = 0
        for text, expected in TEXTS.items():
            tv = t.to_text_vector(text)
            ok += IMAGES[int(np.argmax(v @ tv))] == expected
        print(f"[{label}] image/image sims min={off.min():.3f} max={off.max():.3f} | "
              f"text->image {ok}/{len(TEXTS)} | finite={np.isfinite(v).all()} | v[0][:4]={np.round(v[0][:4], 4)}", flush=True)
    except Exception as e:
        print(f"[{label}] ERROR {type(e).__name__}: {e}", flush=True)
    finally:
        T.from_pretrained = orig


configs = {
    'default': lambda: run('default', T.resolve_device()),
    'cpu': lambda: run('cpu', 'cpu'),
    'cpu-eager': lambda: run('cpu eager attention', 'cpu', attn='eager'),
    'mps': lambda: run('mps', 'mps') if torch.backends.mps.is_available() else print('[mps] not available'),
    'mps-eager': lambda: run('mps eager attention', 'mps', attn='eager') if torch.backends.mps.is_available() else None,
    # last: set_num_threads is global
    'cpu-1thread': lambda: run('cpu 1 thread', 'cpu', threads=1),
}
wanted = [a for a in sys.argv[1:] if not a.startswith('--')] or list(configs)
for name in wanted:
    configs[name]()

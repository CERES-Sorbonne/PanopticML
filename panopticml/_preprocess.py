"""Subprocess-side image decode + resize.

Deliberately free of `panoptic` and `torch` imports. The process pool no longer
starts workers with `fork` (see compute_vector_task), so each worker re-imports
the module defining the worker function in order to unpickle it — dragging the
whole plugin in would cost seconds per worker.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image


def preprocess_worker(args: tuple):
    """Runs in a subprocess, no shared state."""
    sha1, jpeg_bytes, size, greyscale = args
    try:
        img = Image.open(io.BytesIO(jpeg_bytes))
        img = img.convert('L').convert('RGB') if greyscale else img.convert('RGB')
        if img.size != (size, size):
            img = img.resize((size, size), Image.BICUBIC)
        return sha1, np.asarray(img, dtype=np.uint8)
    except Exception:
        return None

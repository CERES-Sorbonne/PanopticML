from __future__ import annotations

import io
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from .panoptic_ml import PanopticML

from panoptic.core.task.task import Task
from panoptic.core.databases.data.models import Instance
from panoptic.core.databases.media.models import ImageType, Vector, VectorType

logger = logging.getLogger('PanopticML')

BATCH_SIZE      = 128   # images per GPU forward pass
IO_WORKERS      = 8     # parallel threads for decode + resize
PREFETCH_QUEUED = 4     # preprocessed batches buffered ahead of the GPU
WRITE_QUEUED    = 2     # GPU results buffered ahead of the DB writer
FETCH_BATCH     = 512   # sha1s fetched from DB per round-trip

# Decode + resize run in a thread pool, not a process pool. PIL releases the GIL while it
# decodes, converts and resizes, so threads scale almost as well here. A process pool has
# to pickle the worker by module name, and a pool child can't import that module: the
# plugin is loaded under its registered name (e.g. "PanopticML"), which is not the
# package folder name for pip installs. The pool then breaks and the task waits forever.
# Each child would also re-import torch and the whole plugin just to unpickle the worker.


def _pick_image_type(image_types: list[ImageType], input_size: int) -> int:
    """Id of the stored rendition to embed from.

    The smallest rendition at least twice the model input, so both axes are downsampled,
    else the largest. New projects store 'small' (256px) and 'large' (1024px). Projects
    converted from Panoptic 0.x can have other renditions and ids.
    """
    sized = [(max(t.width or 0, t.height or 0), t.id) for t in image_types if t.width or t.height]
    if not sized:
        return image_types[0].id
    big_enough = [s for s in sized if s[0] >= 2 * input_size]
    return min(big_enough)[1] if big_enough else max(sized)[1]


def _preprocess_worker(args: tuple):
    """Decode + resize one stored image. Returns None if it can't be decoded."""
    sha1, jpeg_bytes, size, greyscale = args
    try:
        img = Image.open(io.BytesIO(jpeg_bytes))
        img = img.convert('L').convert('RGB') if greyscale else img.convert('RGB')
        if img.size != (size, size):
            img = img.resize((size, size), Image.BICUBIC)
        return sha1, np.asarray(img, dtype=np.uint8)
    except Exception:
        return None


class ComputeVectorsTask(Task):
    """
    3-stage pipeline to keep the GPU busy continuously:
      stage 1 — thread pool  : fetch JPEG from DB + decode + resize
      stage 2 — GPU          : forward pass on a batch of preprocessed arrays
      stage 3 — DB writer    : upsert vectors to DB while GPU runs the next batch
    """

    def __init__(self, plugin: PanopticML, vec_type: VectorType,
                 instances: list[Instance]):
        super().__init__()
        self.project     = plugin.project
        self.plugin      = plugin
        self.vec_type    = vec_type
        self.instances   = instances
        self.name        = f"{vec_type.params['model']} Vectors ({vec_type.id})"
        self.key        += f"_vec{vec_type.id}"
        self.transformer = None
        # `failed` is bumped from the producer, GPU and writer threads
        self._failed_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Task entry point
    # ------------------------------------------------------------------

    def start(self) -> None:
        self.transformer = self.plugin.transformers.get(self.vec_type)

        with self.project._media_db() as db:
            rows = db.conn.execute(
                "SELECT sha1 FROM vectors WHERE type_id = ?", (self.vec_type.id,)
            ).fetchall()
        existing = {row[0] for row in rows}

        # vectors are per sha1: instances sharing an image are computed once
        sha1s = list(dict.fromkeys(
            inst.sha1 for inst in self.instances if inst.sha1 and inst.sha1 not in existing
        ))

        self.state.total = len(sha1s)
        self._notify()

        if not sha1s:
            return

        batch_queue = Queue(maxsize=PREFETCH_QUEUED)
        write_queue = Queue(maxsize=WRITE_QUEUED)

        producer_thread = threading.Thread(
            target=self._producer, args=(sha1s, batch_queue), daemon=True
        )
        writer_thread = threading.Thread(
            target=self._writer, args=(write_queue,), daemon=True
        )

        producer_thread.start()
        writer_thread.start()

        # GPU loop — runs in the task thread
        t_start    = time.perf_counter()
        t_gpu_sum  = 0.0
        done_count = 0

        while True:
            item = batch_queue.get()
            if item is None:
                break
            if self.is_cancelled():
                continue  # keep draining so the producer can reach its sentinel
            sha1s_batch, arrays = item
            try:
                t_gpu_0 = time.perf_counter()
                vectors = self.transformer.forward_from_arrays(arrays)
                t_gpu_sum += time.perf_counter() - t_gpu_0
            except Exception as e:
                logger.error(f"GPU forward pass failed: {e}")
                self._add_failed(len(sha1s_batch))
                continue
            write_queue.put((sha1s_batch, vectors))
            done_count += len(sha1s_batch)

        write_queue.put(None)
        writer_thread.join()
        producer_thread.join()

        t_total = time.perf_counter() - t_start
        imgs_per_sec = done_count / t_total if t_total > 0 else 0
        gpu_pct      = 100 * t_gpu_sum / t_total if t_total > 0 else 0

        print(
            f"\n[PanopticML] Vector compute done: {done_count} images "
            f"in {t_total:.1f}s  ({imgs_per_sec:.1f} img/s) | "
            f"GPU time {t_gpu_sum:.1f}s ({gpu_pct:.0f}%)\n"
        )

    def on_last(self) -> None:
        self.plugin.rebuild_index(self.vec_type)

    # ------------------------------------------------------------------
    # Stage 1 — producer: DB fetch + parallel decode/resize
    # ------------------------------------------------------------------

    def _producer(self, sha1s: list[str], out: Queue) -> None:
        size      = self.transformer.preprocess_size
        greyscale = self.vec_type.params.get('greyscale', False)

        try:
            # The plugin interface has no public image access yet, hence _media_db().
            with self.project._media_db() as db:
                image_type = _pick_image_type(db.get_image_types(), size)

            with ThreadPoolExecutor(max_workers=IO_WORKERS, thread_name_prefix='panopticml-decode') as pool:
                batch_sha1s:  list[str] = []
                batch_arrays: list      = []

                for i in range(0, len(sha1s), FETCH_BATCH):
                    if self.is_cancelled():
                        break

                    chunk = sha1s[i:i + FETCH_BATCH]
                    with self.project._media_db() as db:
                        images = db.get_images(type_id=image_type, sha1=chunk)
                    sha1_to_bytes = {img.sha1: img.data for img in images}

                    args_list = [
                        (sha1, sha1_to_bytes[sha1], size, greyscale)
                        for sha1 in chunk if sha1 in sha1_to_bytes
                    ]
                    failed = len(chunk) - len(args_list)  # no stored image to embed

                    futures = [pool.submit(_preprocess_worker, a) for a in args_list]
                    for fut in as_completed(futures):
                        if self.is_cancelled():
                            break
                        result = fut.result()
                        if result is None:
                            failed += 1
                            continue
                        sha1, arr = result
                        batch_sha1s.append(sha1)
                        batch_arrays.append(arr)

                        if len(batch_sha1s) >= BATCH_SIZE:
                            out.put((batch_sha1s, batch_arrays))
                            batch_sha1s  = []
                            batch_arrays = []

                    if failed:
                        self._add_failed(failed)

                if batch_sha1s:
                    out.put((batch_sha1s, batch_arrays))
        except Exception as e:
            logger.error(f"Image loading failed: {e}")
        finally:
            out.put(None)  # sentinel: always sent, the GPU loop waits for it

    # ------------------------------------------------------------------
    # Stage 3 — writer: DB upserts off the GPU path
    # ------------------------------------------------------------------

    def _writer(self, queue: Queue) -> None:
        while True:
            item = queue.get()
            if item is None:
                break
            sha1s, vectors = item
            try:
                self.project.upsert_vectors([
                    Vector(type_id=self.vec_type.id, sha1=sha1, data=vec)
                    for sha1, vec in zip(sha1s, vectors)
                ])
            except Exception as e:
                logger.error(f"Vector write failed: {e}")
                self._add_failed(len(sha1s))
                continue
            # counted once written: `done` never reports vectors that aren't in the DB
            self.state.done += len(sha1s)
            self._notify()

    def _add_failed(self, n: int) -> None:
        with self._failed_lock:
            self.state.failed += n
        self._notify()

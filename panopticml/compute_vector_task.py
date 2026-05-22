from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from queue import Queue
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .panoptic_ml import PanopticML

from panoptic2.core.task.task import Task
from panoptic.core.databases.media.models import Vector, VectorType
from panoptic.models.data import Instance

logger = logging.getLogger('PanopticML')

BATCH_SIZE      = 128   # images per GPU forward pass
IO_WORKERS      = 8     # parallel processes for decode + resize
PREFETCH_QUEUED = 4     # preprocessed batches buffered ahead of the GPU
WRITE_QUEUED    = 2     # GPU results buffered ahead of the DB writer
FETCH_BATCH     = 512   # sha1s fetched from DB per round-trip


def _preprocess_worker(args: tuple):
    """Module-level worker: runs in a subprocess, no shared state."""
    sha1, jpeg_bytes, size, greyscale = args
    try:
        import io as _io
        import numpy as _np
        from PIL import Image as _Image
        img = _Image.open(_io.BytesIO(jpeg_bytes))
        img = img.convert('L').convert('RGB') if greyscale else img.convert('RGB')
        if img.size != (size, size):
            img = img.resize((size, size), _Image.BICUBIC)
        return sha1, _np.asarray(img, dtype=_np.uint8)
    except Exception:
        return None


class ComputeVectorsTask(Task):
    """
    3-stage pipeline to keep the GPU busy continuously:
      stage 1 — process pool : fetch JPEG from DB + decode + resize (true multiprocessing)
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

        to_compute = [inst for inst in self.instances if inst.sha1 not in existing]

        self.state.total   = len(to_compute)
        self.state.running = True
        self._notify()

        if not to_compute:
            self._finish()
            return

        sha1s = [inst.sha1 for inst in to_compute]

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
            sha1s_batch, arrays = item
            try:
                t_gpu_0 = time.perf_counter()
                vectors = self.transformer.forward_from_arrays(arrays)
                t_gpu_sum += time.perf_counter() - t_gpu_0
                write_queue.put((sha1s_batch, vectors))
            except Exception as e:
                logger.error(f"GPU forward pass failed: {e}")

            done_count         += len(sha1s_batch)
            self.state.done    += len(sha1s_batch)
            self._notify()

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

        self._finish()

    def on_last(self) -> None:
        self.plugin.rebuild_index(self.vec_type)

    # ------------------------------------------------------------------
    # Stage 1 — producer: DB fetch + parallel decode/resize
    # ------------------------------------------------------------------

    def _producer(self, sha1s: list[str], out: Queue) -> None:
        size      = self.transformer.preprocess_size
        greyscale = self.vec_type.params.get('greyscale', False)

        with ProcessPoolExecutor(max_workers=IO_WORKERS) as pool:
            batch_sha1s:  list[str] = []
            batch_arrays: list      = []

            for i in range(0, len(sha1s), FETCH_BATCH):
                if self._cancel_event.is_set():
                    break

                chunk = sha1s[i:i + FETCH_BATCH]
                with self.project._media_db() as db:
                    images = db.get_images(type_id=2, sha1=chunk)
                sha1_to_bytes = {img.sha1: img.data for img in images}

                args_list = [
                    (sha1, sha1_to_bytes[sha1], size, greyscale)
                    for sha1 in chunk if sha1 in sha1_to_bytes
                ]

                futures = [pool.submit(_preprocess_worker, a) for a in args_list]
                for fut in as_completed(futures):
                    if self._cancel_event.is_set():
                        break
                    result = fut.result()
                    if result is None:
                        continue
                    sha1, arr = result
                    batch_sha1s.append(sha1)
                    batch_arrays.append(arr)

                    if len(batch_sha1s) >= BATCH_SIZE:
                        out.put((batch_sha1s, batch_arrays))
                        batch_sha1s  = []
                        batch_arrays = []

            if batch_sha1s:
                out.put((batch_sha1s, batch_arrays))

        out.put(None)  # sentinel

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

    # ------------------------------------------------------------------

    def _finish(self) -> None:
        self.state.running  = False
        self.state.finished = True
        self._finished_event.set()
        self._notify()

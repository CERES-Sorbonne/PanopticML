from __future__ import annotations

import logging
import multiprocessing
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from queue import Empty, Queue
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .panoptic_ml import PanopticML

from panoptic.core.task.task import Task
from panoptic.core.databases.media.models import Vector, VectorType
from panoptic.models.models import Instance

from ._preprocess import preprocess_worker

logger = logging.getLogger('PanopticML')

BATCH_SIZE      = 128   # images per GPU forward pass
IO_WORKERS      = 8     # parallel processes for decode + resize
PREFETCH_QUEUED = 4     # preprocessed batches buffered ahead of the GPU
WRITE_QUEUED    = 2     # GPU results buffered ahead of the DB writer
FETCH_BATCH     = 512   # sha1s fetched from DB per round-trip

# The pool is built from a worker thread, in a process where umap has already pulled in
# numba (and, on Linux, its TBB threading layer) and torch may have initialized CUDA.
# `fork` copies both into the child in an undefined state — TBB says so out loud
# ("Attempted to fork from a non-main thread"). forkserver forks from a clean server
# process instead; spawn is the fallback where forkserver is unavailable.
_START_METHOD = 'forkserver' if 'forkserver' in multiprocessing.get_all_start_methods() else 'spawn'


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
        self._write_error: Exception | None = None

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
        failure: Exception | None = None

        while True:
            item = batch_queue.get()
            if item is None:
                break
            sha1s_batch, arrays = item
            try:
                t_gpu_0 = time.perf_counter()
                vectors = self.transformer.forward_from_arrays(arrays)
                t_gpu_sum += time.perf_counter() - t_gpu_0
            except Exception as e:
                # A forward-pass failure is systematic (shape, dtype, OOM), not per-image.
                # Abort loudly rather than mark every remaining batch as done while
                # writing nothing — that reports success over an empty vector table.
                failure = e
                self._cancel_event.set()
                break
            write_queue.put((sha1s_batch, vectors))

            done_count         += len(sha1s_batch)
            self.state.done    += len(sha1s_batch)
            self._notify()

        write_queue.put(None)
        writer_thread.join()
        self._drain(batch_queue, producer_thread)
        producer_thread.join()

        if failure is not None:
            raise RuntimeError(
                f"Vector computation aborted after {done_count}/{self.state.total} images: "
                f"forward pass failed on {self.vec_type.params['model']}"
            ) from failure
        if self._write_error is not None:
            raise RuntimeError(
                f"Vector computation failed: {done_count} images embedded but the DB "
                f"write failed"
            ) from self._write_error

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

        pool_ctx = multiprocessing.get_context(_START_METHOD)
        with ProcessPoolExecutor(max_workers=IO_WORKERS, mp_context=pool_ctx) as pool:
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

                futures = [pool.submit(preprocess_worker, a) for a in args_list]
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
                if self._write_error is None:
                    self._write_error = e

    # ------------------------------------------------------------------

    @staticmethod
    def _drain(queue: Queue, producer: threading.Thread) -> None:
        """Unblock a producer parked on a full batch_queue so it can observe the cancel."""
        while producer.is_alive():
            try:
                queue.get(timeout=0.1)
            except Empty:
                pass

    # ------------------------------------------------------------------

    def _finish(self) -> None:
        self.state.running  = False
        self.state.finished = True
        self._finished_event.set()
        self._notify()

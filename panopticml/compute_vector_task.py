from __future__ import annotations

import io
import threading
import time
import traceback
from collections import Counter
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

from .compute.accel import Engine, fallback_engine, get_engines

IO_WORKERS      = 8     # parallel threads for decode + resize
PREFETCH_QUEUED = 4     # preprocessed batches buffered per compute engine
WRITE_QUEUED    = 4     # computed batches buffered ahead of the DB writer
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
    """Decode + resize one stored image. Returns (sha1, array, None), or (sha1, None, error)."""
    sha1, jpeg_bytes, size, greyscale = args
    try:
        img = Image.open(io.BytesIO(jpeg_bytes))
        img = img.convert('L').convert('RGB') if greyscale else img.convert('RGB')
        if img.size != (size, size):
            img = img.resize((size, size), Image.BICUBIC)
        return sha1, np.asarray(img, dtype=np.uint8), None
    except Exception as e:
        return sha1, None, e


class ComputeVectorsTask(Task):
    """
    3-stage pipeline to keep the compute hardware busy continuously:
      stage 1 — thread pool  : fetch JPEG from DB + decode + resize
      stage 2 — engines      : one thread per compute engine (GPU, plus the Neural Engine
                               on Apple Silicon for some models), each taking batches
                               from the shared queue at its own pace
      stage 3 — DB writer    : upsert vectors to DB while the engines run the next batches
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
        # `failed` is bumped from the producer, engine and writer threads
        self._failed_lock = threading.Lock()
        self._fail_reasons: Counter[str] = Counter()
        self._batch_size = 1
        # set once every batch has been handed to an engine: engines still getting ready stop waiting
        self._drained = threading.Event()

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

        engines = get_engines(self.transformer, neural_engine=self.plugin.params.neural_engine)
        # the GPU engine's batch: small enough that engines share the work evenly
        self._batch_size = engines[0].batch_size
        batch_queue = Queue(maxsize=PREFETCH_QUEUED * len(engines))
        write_queue = Queue(maxsize=WRITE_QUEUED)

        producer_thread = threading.Thread(
            target=self._producer, args=(sha1s, batch_queue), daemon=True
        )
        writer_thread = threading.Thread(
            target=self._writer, args=(write_queue,), daemon=True
        )
        producer_thread.start()
        writer_thread.start()

        t_start = time.perf_counter()
        stats: dict[str, list] = {}            # engine name -> [images, busy seconds]
        engine_threads = [
            threading.Thread(target=self._consume, args=(engine, False, batch_queue, write_queue, stats),
                             name=f'panopticml-{engine.name}', daemon=True)
            for engine in engines[1:]
        ]
        for thread in engine_threads:
            thread.start()
        self._consume(engines[0], True, batch_queue, write_queue, stats)   # primary engine: this thread
        for thread in engine_threads:
            thread.join()

        write_queue.put(None)
        writer_thread.join()
        producer_thread.join()

        t_total = time.perf_counter() - t_start
        done_count = sum(n for n, _ in stats.values())
        per_engine = ', '.join(
            f"{name} {n} ({n / busy:.1f} img/s busy {100 * busy / t_total:.0f}%)"
            for name, (n, busy) in stats.items() if busy > 0
        )
        print(
            f"\n[PanopticML] Vector compute done: {done_count} images "
            f"in {t_total:.1f}s  ({done_count / t_total if t_total > 0 else 0:.1f} img/s) | "
            f"{per_engine}\n"
        )
        if self._fail_reasons:
            lines = '\n'.join(f"  {n:>8}  {reason}" for reason, n in self._fail_reasons.most_common())
            print(f"[PanopticML] {self.name}: {self.state.failed} image(s) failed:\n{lines}\n")

    def on_last(self) -> None:
        self.plugin.rebuild_index(self.vec_type)

    # ------------------------------------------------------------------
    # Stage 2 — compute engines
    # ------------------------------------------------------------------

    def _consume(self, engine: Engine, primary: bool, batches: Queue, results: Queue, stats: dict) -> None:
        """Embed batches from the shared queue until it is drained. An extra engine that
        fails hands its batch to PyTorch and stops; a failing primary (MLX) is replaced by
        PyTorch for the rest of the task."""
        while not engine.ready.wait(0.5):       # Neural Engine model still converting / loading
            if engine.failed or self._drained.is_set():
                return
        if engine.failed:
            return
        while True:
            item = batches.get()
            if item is None:
                self._drained.set()
                batches.put(None)                # leave the sentinel for the other engines
                return
            if self.is_cancelled():
                continue                         # keep draining so the producer can reach its sentinel
            sha1s_batch, arrays = item
            t0 = time.perf_counter()
            name, stop = engine.name, False
            try:
                vectors = engine.forward(arrays)
            except Exception as e:
                vectors = self._fallback(engine, sha1s_batch, arrays, e)
                name, stop = fallback_engine(self.transformer).name, not primary
                if primary:
                    engine = fallback_engine(self.transformer)
            if vectors is not None:
                entry = stats.setdefault(name, [0, 0.0])
                entry[0] += len(sha1s_batch)
                entry[1] += time.perf_counter() - t0
                results.put((sha1s_batch, vectors))
            if stop:
                return

    def _fallback(self, engine: Engine, sha1s_batch: list[str], arrays: list, error: Exception):
        """Vectors of a batch `engine` failed on, from the PyTorch path (None if that fails too)."""
        torch_engine = fallback_engine(self.transformer)
        if engine.name == torch_engine.name:
            self._add_failed(len(sha1s_batch), f"{engine.name} forward pass failed ({type(error).__name__})", error)
            return None
        engine.disable(f"forward pass failed: {type(error).__name__}: {error}")
        try:
            return torch_engine.forward(arrays)
        except Exception as e:
            self._add_failed(len(sha1s_batch), f"{torch_engine.name} forward pass failed ({type(e).__name__})", e)
            return None

    # ------------------------------------------------------------------
    # Stage 1 — producer: DB fetch + parallel decode/resize
    # ------------------------------------------------------------------

    def _producer(self, sha1s: list[str], out: Queue) -> None:
        size      = self.transformer.preprocess_size
        greyscale = self.vec_type.params.get('greyscale', False)

        try:
            # The plugin interface has no public image access yet, hence _media_db().
            with self.project._media_db() as db:
                image_types = db.get_image_types()
            image_type = _pick_image_type(image_types, size)
            print(
                f"[PanopticML] {self.name}: embedding from image type {image_type} "
                f"(available: {[(t.id, t.width, t.height) for t in image_types]})"
            )

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
                    missing = [sha1 for sha1 in chunk if sha1 not in sha1_to_bytes]
                    if missing:
                        self._add_failed(
                            len(missing), f"no stored image of type {image_type}",
                            f"e.g. sha1 {missing[0]}",
                        )

                    futures = [pool.submit(_preprocess_worker, a) for a in args_list]
                    for fut in as_completed(futures):
                        if self.is_cancelled():
                            break
                        sha1, arr, error = fut.result()
                        if error is not None:
                            self._add_failed(
                                1, f"image decode failed ({type(error).__name__})",
                                f"sha1 {sha1}: {error}",
                            )
                            continue
                        batch_sha1s.append(sha1)
                        batch_arrays.append(arr)

                        if len(batch_sha1s) >= self._batch_size:
                            out.put((batch_sha1s, batch_arrays))
                            batch_sha1s  = []
                            batch_arrays = []

                if batch_sha1s:
                    out.put((batch_sha1s, batch_arrays))
        except Exception:
            print(f"[PanopticML] {self.name}: image loading failed\n{traceback.format_exc()}")
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
                self._add_failed(len(sha1s), f"vector write failed ({type(e).__name__})", e)
                continue
            # counted once written: `done` never reports vectors that aren't in the DB
            self.state.done += len(sha1s)
            self._notify()

    def _add_failed(self, n: int, reason: str, detail: str | BaseException = '') -> None:
        """Count `n` failures. Each distinct reason is printed once (with a traceback for
        exceptions), so a run that fails on every image doesn't flood the console; the
        totals per reason are printed when the task ends."""
        with self._failed_lock:
            self.state.failed += n
            first = reason not in self._fail_reasons
            self._fail_reasons[reason] += n
        if first:
            if isinstance(detail, BaseException):
                detail = ''.join(traceback.format_exception(detail)).rstrip()
            print(f"[PanopticML] {self.name}: {reason}" + (f"\n{detail}" if detail else ''))
        self._notify()

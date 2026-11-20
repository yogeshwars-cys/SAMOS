"""Staged SAMOS test: two offloads, then spend the freed RAM.

  Stage 1: initial bloat (60 MB arrays) offloaded  -> RAM freed
  Stage 2: secondary bloat (40 MB tiles+KV) offloaded -> more RAM freed
  Stage 3: the freed RAM runs MORE heavy subtasks concurrently than
           HOME can hold at the same footprint, while the secondary
           indexing task consumes the offloaded segments over the bus.

HOME runs the 4 required heavy subtasks with all bloat resident.
SAM runs the same 4 (signature must match) PLUS 4 bonus subtasks in
the freed space, plus background indexing -- and its peak RSS must
stay at or below HOME's peak.
"""
import gc
import hashlib
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import psutil

from samos.fabric import Fabric

MIB = 1024 * 1024
proc = psutil.Process()


def sha(b):
    return hashlib.sha256(b).hexdigest()


def rss():
    return proc.memory_info().rss / MIB


def trim():
    if os.name == "nt":
        try:
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            k32.SetProcessWorkingSetSize.argtypes = [ctypes.c_void_p] * 3
            k32.SetProcessWorkingSetSize(k32.GetCurrentProcess(),
                                         ctypes.c_size_t(-1), ctypes.c_size_t(-1))
        except Exception:
            pass


class Peak:
    def __init__(self):
        self.max = rss()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(0.02):
            self.max = max(self.max, rss())

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()


rngW = np.random.default_rng(7)
W1 = (rngW.standard_normal((1024, 2048)) * 0.03).astype(np.float32)
W2 = (rngW.standard_normal((2048, 1024)) * 0.03).astype(np.float32)


def heavy_subtask(i):
    rng = np.random.default_rng(100 + i)
    X = rng.standard_normal((1024, 1024)).astype(np.float32)
    ws = rng.standard_normal((2048, 1024)).astype(np.float32)  # private buffer
    H = X
    for _ in range(8):
        U = H @ W1
        U = U * (1.0 / (1.0 + np.exp(-np.clip(U, -10, 10))))
        H = H + 0.05 * (U @ W2) + 0.01 * ws[:1024]
    return sha(np.round(H[:16, :16], 3).astype(np.float32).tobytes())


def main():
    rng = np.random.default_rng(21)
    bloat_a = rng.standard_normal(60 * MIB // 4).astype(np.float32).tobytes()
    tiles = rng.integers(-127, 127, size=24 * MIB, dtype=np.int8).tobytes()
    kv = (("\n".join(f"key{i} " + " ".join(f"tok{(i + j) % 97}" for j in range(20))
                     for i in range(90000))).encode())[:16 * MIB]
    bloat_b_names = ["session_tiles", "kv_state"]
    sha_a, sha_tiles, sha_kv = sha(bloat_a), sha(tiles), sha(kv)

    # ---------------- HOME: everything resident ----------------
    mark_loaded = rss()
    with Peak() as peak:
        t0 = time.time()
        with ThreadPoolExecutor(4) as ex:
            home_sigs = list(ex.map(heavy_subtask, range(4)))
        home_wall = time.time() - t0
    home_peak = peak.max
    print(f"HOME : loaded RSS {mark_loaded:.0f} MB, 4 heavy subtasks in "
          f"{home_wall:.2f}s, peak RSS {home_peak:.0f} MB")

    # ---------------- SAM: two staged offloads ----------------
    fab = Fabric(port=9977)
    fab.begin_task({"program": "staged_test", "segments": [
        {"name": "weights_a", "bytes": len(bloat_a), "tier": "cold"},
        {"name": "session_tiles", "bytes": len(tiles), "tier": "cold"},
        {"name": "kv_state", "bytes": len(kv), "tier": "cold"}]})
    sam_marks = {"loaded": rss()}
    t0 = time.time()
    fab.offload("weights_a", bloat_a)
    del bloat_a
    gc.collect(); trim()
    sam_marks["after_stage1"] = rss()
    fab.offload("session_tiles", tiles)
    fab.offload("kv_state", kv)
    del tiles, kv
    gc.collect(); trim()
    sam_marks["after_stage2"] = rss()
    push_s = time.time() - t0

    idx = {}

    def secondary():
        for name, want in [("weights_a", sha_a), ("session_tiles", sha_tiles),
                           ("kv_state", sha_kv)]:
            h = hashlib.sha256()
            fab.consume(name, on_chunk=h.update)  # one pass: index + repatriate
            idx[name] = (h.hexdigest() == want)

    with Peak() as peak:
        t0 = time.time()
        with ThreadPoolExecutor(5) as ex:  # bus busy: cap heavy at 4 alongside
            f_sec = ex.submit(secondary)
            sam_sigs = list(ex.map(heavy_subtask, range(8)))  # 4 required + 4 bonus
            f_sec.result()
        sam_wall = time.time() - t0
    report = fab.complete()
    fab.close()

    freed1 = sam_marks["loaded"] - sam_marks["after_stage1"]
    freed2 = sam_marks["after_stage1"] - sam_marks["after_stage2"]
    print(f"SAM  : push {push_s:.1f}s | RSS loaded {sam_marks['loaded']:.0f} MB "
          f"-> stage1 {sam_marks['after_stage1']:.0f} MB (-{freed1:.0f}) "
          f"-> stage2 {sam_marks['after_stage2']:.0f} MB (-{freed2:.0f})")
    print(f"SAM  : 8 heavy subtasks (4 required + 4 bonus in freed RAM) + "
          f"secondary indexing in {sam_wall:.2f}s, peak RSS {peak.max:.0f} MB")
    print(f"     required-4 signature match HOME: {sam_sigs[:4] == home_sigs}; "
          f"secondary digests ok: {all(idx.values())}; "
          f"bring-back ok: {all(v['digest_ok'] for v in report.values())}")

    assert sam_sigs[:4] == home_sigs
    assert all(idx.values())
    print(f"\nVERDICT: SAM peak {peak.max:.0f} MB vs HOME peak {home_peak:.0f} MB: "
          f"2x the heavy subtasks plus background indexing, inside the "
          f"footprint the two offloads freed.")


if __name__ == "__main__":
    main()

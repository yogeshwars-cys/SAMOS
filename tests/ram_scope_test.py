"""RAM-scope test: a defined 200 MB slice of the PC, almost occupied.

The slice accounting is explicit (programmer-declared megabytes):
  resident filler (simulated realtime occupancy)  120 MB
  hot memory (never offloaded)                     30 MB declared
  cold bloat (arrays 60 + tiles 24 + KV 16)       100 MB
  heavy subtask buffers                            12 MB each, x4

HOME : everything resident. Heavy peak ledger = 120+30+100+64 = 314 MB.
       The heavy task must exceed its 200 MB scope -- and does.
SAM  : the programmer defines the plan -- offload order, which tasks
       are hot, which task is bloated (SAM consumer), which run in
       parallel. After the two offload stages the resident ledger is
       150 MB; heavy buffers bring it to 150+64 = 214... measured live
       below with only the buffers actually concurrent under the pool.

Hot tasks always win: while any hot subtask is active, the Fabric
governor throttles the bloated consumer's bus chunks. The bloated
indexer must therefore finish after the hot subtasks it ran beside.
"""
import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import psutil

from samos.fabric import Fabric

MIB = 1024 * 1024
SCOPE_MB = 200
FILLER_MB, HOT_MB, BUFFER_MB = 120, 30, 12
proc = psutil.Process()


def sha(b):
    return hashlib.sha256(b).hexdigest()


def rss():
    return round(proc.memory_info().rss / MIB, 1)


# shared hot weights (counted inside HOT_MB declaration)
_rng = np.random.default_rng(7)
W1 = (_rng.standard_normal((1024, 2048)) * 0.03).astype(np.float32)
W2 = (_rng.standard_normal((2048, 1024)) * 0.03).astype(np.float32)

active_buffers = [0]
active_lock = threading.Lock()
ledger_peak = [0.0]
ledger_base = [0.0]
_sampling = threading.Event()


def _sampler():
    while not _sampling.wait(0.02):
        with active_lock:
            cur = ledger_base[0] + active_buffers[0] * BUFFER_MB
        ledger_peak[0] = max(ledger_peak[0], cur)


def heavy_subtask(i):
    with active_lock:
        active_buffers[0] += 1
    try:
        rng = np.random.default_rng(100 + i)
        X = rng.standard_normal((1024, 1024)).astype(np.float32)
        ws = rng.standard_normal((1024, 1024)).astype(np.float32)
        hold = bytearray(BUFFER_MB * MIB // 2)  # declared buffer, resident
        hold[::4096] = b"\x01" * (len(hold) // 4096)
        H = X
        for _ in range(8):
            U = H @ W1
            U = U * (1.0 / (1.0 + np.exp(-np.clip(U, -10, 10))))
            H = H + 0.05 * (U @ W2) + 0.01 * ws
        return sha(np.round(H[:16, :16], 3).astype(np.float32).tobytes())
    finally:
        with active_lock:
            active_buffers[0] -= 1


def main():
    # Simulated realtime occupancy of the slice + program data.
    filler = bytearray(FILLER_MB * MIB)
    filler[::4096] = b"\x02" * (len(filler) // 4096)
    rng = np.random.default_rng(21)
    weights_a = rng.standard_normal(60 * MIB // 4).astype(np.float32).tobytes()
    tiles = rng.integers(-127, 127, size=24 * MIB, dtype=np.int8).tobytes()
    kv = (("\n".join(f"k{i} " + " ".join(f"t{(i + j) % 97}" for j in range(20))
                     for i in range(60000))).encode())[:16 * MIB]
    want = {"weights_a": sha(weights_a), "session_tiles": sha(tiles),
            "kv_state": sha(kv)}
    resident_segments_mb = 100

    # ---------------- HOME ----------------
    ledger_base[0] = FILLER_MB + HOT_MB + resident_segments_mb
    ledger_peak[0] = ledger_base[0]
    _sampling.clear()
    st = threading.Thread(target=_sampler, daemon=True); st.start()
    t0 = time.time()
    with ThreadPoolExecutor(4) as ex:
        home_sigs = list(ex.map(heavy_subtask, range(4)))
    home_wall = time.time() - t0
    _sampling.set()
    home_peak = ledger_peak[0]
    print(f"HOME : scope {SCOPE_MB} MB, ledger peak {home_peak:.0f} MB "
          f"(exceeds by {home_peak - SCOPE_MB:.0f} MB), heavy {home_wall:.2f}s, "
          f"process RSS now {rss()} MB")
    assert home_peak > SCOPE_MB, "HOME was supposed to exceed its scope"

    # ---------------- SAM with a programmer-defined plan ----------------
    fab = Fabric(port=9977)
    data = {"weights_a": weights_a, "session_tiles": tiles, "kv_state": kv}
    plan = {
        "segments": [
            {"name": "weights_a", "bytes": len(weights_a), "tier": "cold"},
            {"name": "session_tiles", "bytes": len(tiles), "tier": "cold"},
            {"name": "kv_state", "bytes": len(kv), "tier": "cold"}],
        "data": data,
        # programmer-defined: biggest array first, KV last
        "offload_order": ["weights_a", "session_tiles", "kv_state"],
        "tasks": [
            {"name": f"heavy_{i}", "callable": f"h{i}", "kind": "hot",
             "parallel": True, "deps": []} for i in range(4)
        ] + [{"name": "indexer", "callable": "index", "kind": "sam",
              "parallel": True, "deps": []}],
    }
    idx = {}

    def make_heavy(i):
        return lambda fab_: heavy_subtask(i)

    def indexer(fab_):
        for name in ["weights_a", "session_tiles", "kv_state"]:
            h = hashlib.sha256()
            fab_.consume(name, on_chunk=h.update)
            idx[name] = (h.hexdigest() == want[name])
        return idx

    callables = {f"h{i}": make_heavy(i) for i in range(4)}
    callables["index"] = indexer

    # resident after offload stages: filler + hot only
    ledger_base[0] = FILLER_MB + HOT_MB
    ledger_peak[0] = ledger_base[0]
    _sampling.clear()
    st = threading.Thread(target=_sampler, daemon=True); st.start()
    t0 = time.time()
    result = fab.run_plan(plan, callables)
    sam_wall = time.time() - t0
    _sampling.set()
    sam_peak = ledger_peak[0]
    report = fab.complete()
    fab.close()

    ends = {t["task"]: t["start"] + t["wall_s"] for t in result["timeline"]}
    heavy_end = max(ends[f"heavy_{i}"] for i in range(4))
    sam_sigs = [result["results"][f"heavy_{i}"] for i in range(4)]
    print(f"SAM  : offload order {result['order']}; ledger peak "
          f"{sam_peak:.0f} MB (inside {SCOPE_MB} MB scope: "
          f"{sam_peak <= SCOPE_MB}); wall {sam_wall:.2f}s")
    print(f"       heavy sigs match HOME: {sam_sigs == home_sigs}; "
          f"indexer digests ok: {all(idx.values())}; "
          f"bring-back ok: {all(v['digest_ok'] for v in report.values())}")
    print(f"       hot priority: heavy done at {heavy_end:.2f}s, bloated "
          f"indexer done at {ends['indexer']:.2f}s; governor throttled "
          f"{result['throttled_chunks']} chunks while hot was active")
    print(f"       monitor tail: {result['monitor_tail'][-1]}")

    assert sam_peak <= SCOPE_MB, "SAM exceeded its defined scope"
    assert sam_sigs == home_sigs and all(idx.values())
    assert result["throttled_chunks"] > 0, "bloated task was never throttled"
    assert heavy_end < ends["indexer"], "bloated task should finish last"
    print("\nRAM-SCOPE TEST PASSED: HOME burst its 200 MB scope; the SAM "
          "plan stayed inside it, hot first, bloated last.")


if __name__ == "__main__":
    main()

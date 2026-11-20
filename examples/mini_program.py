"""A miniature real program that uses the SAMOS Fabric.

It has HOT memory (inputs + working set it cannot function without,
pinned best-effort) and ~100 MB of COLD bloat (weights cache, session
tiles, KV text state). In --mode sam the cold memory is offloaded to
the phone, the heavy task runs on hot memory while a secondary
indexing task consumes the cold segments over the bus (repatriating
each as it finishes), and completion brings everything home verified.

Run: python mini_program.py --mode sam|home --status status.jsonl
"""
import argparse
import gc
import hashlib
import json
import os
import threading
import time

import numpy as np
import psutil

from samos.fabric import Fabric, HotPin

MIB = 1024 * 1024


def sha(b):
    return hashlib.sha256(b).hexdigest()


def build_cold():
    rng = np.random.default_rng(11)
    weights = rng.standard_normal(12 * MIB // 4).astype(np.float32)   # 48 MiB
    tiles = rng.integers(-127, 127, size=32 * MIB, dtype=np.int8)     # 32 MiB
    vocab = [f"tok{i}" for i in range(512)]
    lines = []
    total = 0
    i = 0
    while total < 20 * MIB:
        line = f"key{i} " + " ".join(vocab[(i * 7 + j) % 512] for j in range(24)) + "\n"
        lines.append(line)
        total += len(line)
        i += 1
    kv = "".join(lines).encode()[:20 * MIB]
    return {"weights_cache": weights.tobytes(),
            "session_tiles": tiles.tobytes(),
            "kv_state": kv}


def heavy_task(hot, layers=16):
    X, W1, W2, ws = hot
    H = X.copy()
    for _ in range(layers):
        U = H @ W1
        U = U * (1.0 / (1.0 + np.exp(-np.clip(U, -10, 10))))
        H = H + 0.05 * (U @ W2) + 0.01 * ws[:H.shape[0]]
        H = (H - H.mean(axis=1, keepdims=True)) / (np.sqrt(H.var(axis=1, keepdims=True)) + 1e-5)
    return sha(np.round(H[:32, :16], 3).astype(np.float32).tobytes())


def index_bytes(data, is_text):
    tokens = 0
    if is_text:
        tokens = len(data.split())
    return sha(data), tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sam", "home"], required=True)
    ap.add_argument("--port", type=int, default=9977)
    ap.add_argument("--status", default="mini_status.jsonl")
    args = ap.parse_args()

    proc = psutil.Process()
    t_start = time.time()

    def mark(stage, **kw):
        row = {"stage": stage, "t": round(time.time() - t_start, 2),
               "rss_mb": round(proc.memory_info().rss / MIB, 1), **kw}
        with open(args.status, "a") as f:
            f.write(json.dumps(row) + "\n")

    # HOT memory: the absolute-necessary RAM. Never offloaded.
    rng = np.random.default_rng(5)
    hot = (rng.standard_normal((1024, 1024)).astype(np.float32),
           rng.standard_normal((1024, 2048)).astype(np.float32) * 0.03,
           rng.standard_normal((2048, 1024)).astype(np.float32) * 0.03,
           rng.standard_normal((2048, 1024)).astype(np.float32))
    pins = [HotPin(a) for a in hot]
    cold = build_cold()
    cold_sha = {k: sha(v) for k, v in cold.items()}
    mark("loaded", hot_locked=sum(p.locked for p in pins))

    out = {"mode": args.mode}
    if args.mode == "home":
        t0 = time.time()
        sig = heavy_task(hot)
        out["heavy_s"] = round(time.time() - t0, 2)
        t0 = time.time()
        idx = {k: index_bytes(v, k == "kv_state") for k, v in cold.items()}
        out["secondary_s"] = round(time.time() - t0, 2)
        out["heavy_sig"] = sig
        out["index_ok"] = all(idx[k][0] == cold_sha[k] for k in cold)
        mark("completed")
    else:
        fab = Fabric(port=args.port)
        spec = {"program": "mini_program", "segments": [
            {"name": k, "bytes": len(v), "tier": "cold",
             "kind": "text" if k == "kv_state" else "array"}
            for k, v in cold.items()]}
        fab.begin_task(spec)
        t0 = time.time()
        for k, v in cold.items():
            fab.offload(k, v)
        out["push_s"] = round(time.time() - t0, 2)
        del cold
        gc.collect()
        # What a real optimizer does after offloading: release the
        # freed pages back to Windows instead of letting the runtime
        # keep them in its arenas.
        if os.name == "nt":
            try:
                import ctypes
                k32 = ctypes.windll.kernel32
                k32.GetCurrentProcess.restype = ctypes.c_void_p
                k32.SetProcessWorkingSetSize.argtypes = [
                    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
                k32.SetProcessWorkingSetSize(
                    k32.GetCurrentProcess(), ctypes.c_size_t(-1), ctypes.c_size_t(-1))
            except Exception:
                pass
        mark("offloaded", degraded=False)

        results = {}
        degraded_seen = []

        def run_heavy():
            t0 = time.time()
            results["heavy_sig"] = heavy_task(hot)
            results["heavy_s"] = round(time.time() - t0, 2)
            mark("heavy_done")

        def run_secondary():
            t0 = time.time()
            idx = {}
            for name in ["weights_cache", "session_tiles", "kv_state"]:
                h = hashlib.sha256()
                tokens = [0]
                def feed(chunk, h=h, tokens=tokens, name=name):
                    h.update(chunk)
                    if name == "kv_state":
                        tokens[0] += len(chunk.split())
                    if fab.degraded.is_set():
                        degraded_seen.append(name)
                fab.consume(name, on_chunk=feed)  # one pass: index + repatriate
                idx[name] = (h.hexdigest(), tokens[0])
            results["index"] = idx
            results["secondary_s"] = round(time.time() - t0, 2)
            mark("secondary_done")

        th1 = threading.Thread(target=run_heavy)
        th2 = threading.Thread(target=run_secondary)
        th1.start(); th2.start(); th1.join(); th2.join()
        report = fab.complete()
        out.update({"heavy_sig": results["heavy_sig"],
                    "heavy_s": results["heavy_s"],
                    "secondary_s": results["secondary_s"],
                    "index_ok": all(results["index"][k][0] == cold_sha[k] for k in cold_sha),
                    "bringback": report,
                    "degraded_seen": sorted(set(degraded_seen))})
        mark("completed")
        fab.close()

    out["wall_s"] = round(time.time() - t_start, 2)
    out["rss_mb"] = round(proc.memory_info().rss / MIB, 1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()

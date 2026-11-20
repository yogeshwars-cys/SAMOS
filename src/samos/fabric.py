"""SAMOS Fabric -- the optimizer programs talk to.

A program explicitly declares its memory in a spec:

    spec = {
      "program": "my_app",
      "segments": [
        {"name": "weights_cache", "bytes": 48*1024*1024, "tier": "cold"},
        {"name": "session_kv",    "bytes": 20*1024*1024, "tier": "cold"},
      ],
    }

Lifecycle (the SAM contract):
  1. begin_task(spec) reserves coordinates in the phone's SAM arena.
  2. The program offloads every segment except its HOT memory (the
     absolute-necessary RAM, which never leaves the laptop).
  3. While the task runs, secondary work may read segments back
     chunk-wise over the bus. A segment the secondary work has consumed
     is repatriated (prefetched back) immediately.
  4. On completion, all remaining segments are brought back and
     digest-verified before the task closes. Nothing is left on SAM.
  5. Cable pull: a heartbeat monitor detects link loss. Hot memory is
     local by construction, so the primary task cannot break. On link
     return, the fabric prefetches everything home first, then lets
     secondary work finish locally. Continuity, not magic.
  6. Last resort: if laptop RAM cannot hold a bring-back, that segment
     lands in a memory-mapped paging file (the Windows paging idea,
     demoted to a file) instead of RAM, and says so.

Hot pinning uses Windows VirtualLock where available (the AWE idea:
pages that must not be paged out), best-effort and reported.
"""
import hashlib
import mmap
import os
import threading
import time

import numpy as np
import psutil

from samos.client import Engine

SEG_BYTES = 1024 * 1024
MIB = 1024 * 1024
CHUNK = 256 * 1024
HERE = os.path.dirname(os.path.abspath(__file__))


def _sha(b):
    return hashlib.sha256(b).hexdigest()


class HotPin:
    """Best-effort VirtualLock of a numpy array's pages (Windows)."""

    def __init__(self, arr):
        self.arr = arr
        self.locked = False
        if os.name == "nt":
            try:
                import ctypes
                buf = (ctypes.c_char * arr.nbytes).from_address(arr.ctypes.data)
                self._buf = buf
                self.locked = bool(ctypes.windll.kernel32.VirtualLock(
                    ctypes.byref(buf), ctypes.c_size_t(arr.nbytes)))
            except Exception:
                self.locked = False

    def release(self):
        if self.locked and os.name == "nt":
            try:
                import ctypes
                ctypes.windll.kernel32.VirtualUnlock(
                    ctypes.byref(self._buf), ctypes.c_size_t(self.arr.nbytes))
            except Exception:
                pass
            self.locked = False


class Fabric:
    def __init__(self, host="127.0.0.1", port=9977, link_wait_s=30.0,
                 force_paging=False, paging_dir=None, bloat_throttle_s=0.004):
        self.host, self.port = host, port
        self.link_wait_s = link_wait_s
        self.force_paging = force_paging
        self.paging_dir = paging_dir or os.path.join(os.getcwd(), "samos_paging")
        self.bloat_throttle_s = bloat_throttle_s
        self._eng = None
        self._segments = {}   # name -> {segs:[ids], nbytes, sha, kind}
        self._next_seg = 0
        self.degraded = threading.Event()
        self._stop_monitor = threading.Event()
        # Conscious routing state: who is hot, how much is where.
        self._hot_active = 0
        self._hot_lock = threading.Lock()
        self.stats = {"throttled_chunks": 0, "sam_bytes": 0, "push_bytes": 0}
        self.history = []
        self._connect()
        self.hello = self._eng.hello()
        self._monitor = threading.Thread(target=self._heartbeat, daemon=True)
        self._monitor.start()
        self._watcher = threading.Thread(target=self._watch, daemon=True)
        self._watcher.start()

    # -- link management -------------------------------------------------
    def _connect(self):
        self._eng = Engine(self.host, self.port)

    def _heartbeat(self):
        # Own short-lived connection: never share the task socket,
        # interleaved probes would corrupt the protocol stream.
        while not self._stop_monitor.wait(0.5):
            try:
                probe = Engine(self.host, self.port, timeout=3.0)
                probe.hello()
                probe.close()
                self.degraded.clear()
            except Exception:
                self.degraded.set()

    def _call(self, fn):
        """Run an engine op; on link loss wait for the cable, then retry.
        Push/fetch by coordinate are idempotent, so retry is safe."""
        deadline = time.time() + self.link_wait_s
        while True:
            try:
                return fn(self._eng)
            except (ConnectionError, OSError, IOError):
                self.degraded.set()
                if time.time() > deadline:
                    raise
                time.sleep(0.5)
                try:
                    self._connect()
                except Exception:
                    pass

    def close(self):
        self._stop_monitor.set()
        try:
            self._eng.close()
        except Exception:
            pass

    # -- task lifecycle ----------------------------------------------------
    def begin_task(self, spec):
        for seg in spec["segments"]:
            if seg["tier"] == "hot":
                continue  # hot memory never leaves the laptop
            nseg = (seg["bytes"] + SEG_BYTES - 1) // SEG_BYTES
            ids = list(range(self._next_seg, self._next_seg + nseg))
            self._next_seg += nseg
            self._segments[seg["name"]] = {
                "segs": ids, "nbytes": seg["bytes"], "sha": None,
                "kind": seg.get("kind", "bytes"), "home": None, "tier": None,
            }
        return self

    def offload(self, name, data):
        meta = self._segments[name]
        assert len(data) == meta["nbytes"], (name, len(data), meta["nbytes"])
        meta["sha"] = _sha(data)
        mv = memoryview(data)
        for i, seg in enumerate(meta["segs"]):
            part = mv[i * SEG_BYTES:(i + 1) * SEG_BYTES]
            for off in range(0, len(part), CHUNK):
                chunk = bytes(part[off:off + CHUNK])
                self._call(lambda e, s=seg, o=off, c=chunk: e.push_segment(s, c, o))
                self.stats["push_bytes"] += len(chunk)
        return {"name": name, "segments": meta["segs"], "bytes": len(data)}

    # -- conscious monitoring + hot-priority governance ------------------
    def hot_scope(self):
        """Context manager: while any hot task is inside, bloated SAM
        consumers intentionally lose bandwidth (throttled chunks)."""
        fab = self

        class _Hot:
            def __enter__(self):
                with fab._hot_lock:
                    fab._hot_active += 1
                return self

            def __exit__(self, *a):
                with fab._hot_lock:
                    fab._hot_active -= 1
        return _Hot()

    def monitor_snapshot(self):
        on_sam = sum(m["nbytes"] for m in self._segments.values()
                     if m["home"] is None)
        home = sum(m["nbytes"] for m in self._segments.values()
                   if m["home"] is not None)
        vm = psutil.virtual_memory()
        return {"rss_mb": round(psutil.Process().memory_info().rss / MIB, 1),
                "sys_avail_mb": round(vm.available / MIB),
                "on_sam_mb": round(on_sam / MIB, 1),
                "home_mb": round(home / MIB, 1),
                "hot_active": self._hot_active,
                "degraded": self.degraded.is_set(),
                "throttled_chunks": self.stats["throttled_chunks"]}

    def _watch(self):
        while not self._stop_monitor.wait(0.25):
            self.history.append(self.monitor_snapshot())
            if len(self.history) > 400:
                self.history = self.history[-400:]

    def _govern(self):
        """Bloated consumers yield while hot work is active."""
        if self._hot_active > 0 and self.bloat_throttle_s > 0:
            self.stats["throttled_chunks"] += 1
            time.sleep(self.bloat_throttle_s)

    def run_plan(self, plan, callables):
        """Execute a programmer-defined plan.

        plan = {
          "segments": [ {name, bytes, tier, kind} ... ],   # declarations
          "data": {name: bytes},                            # segment payloads
          "offload_order": [name, ...],                     # push order
          "tasks": [ {"name", "callable", "kind": "hot"|"sam"|"compute",
                      "parallel": bool, "deps": [task names]} ... ],
        }
        Offloads happen in the programmer's order. Tasks then run in
        dependency waves; inside a wave, hot tasks go first and
        parallel tasks share a pool. SAM ("bloated") tasks are
        throttled whenever a hot task is active.
        """
        self.begin_task(plan)
        order = plan.get("offload_order") or [s["name"] for s in plan["segments"]
                                              if s["tier"] != "hot"]
        for name in order:
            self.offload(name, plan["data"][name])

        tasks = {t["name"]: t for t in plan.get("tasks", [])}
        done = {}
        timeline = []
        pending = dict(tasks)
        while pending:
            ready = [t for t in pending.values()
                     if all(d in done for d in t.get("deps", []))]
            if not ready:
                raise RuntimeError("task dependency cycle in plan")
            ready.sort(key=lambda t: {"hot": 0, "compute": 1, "sam": 2}[t["kind"]])
            par = [t for t in ready if t.get("parallel")]
            seq = [t for t in ready if not t.get("parallel")]

            def run_one(t):
                t0 = time.time()
                fn = callables[t["callable"]]
                if t["kind"] == "hot":
                    with self.hot_scope():
                        r = fn(self)
                else:
                    r = fn(self)
                timeline.append({"task": t["name"], "kind": t["kind"],
                                 "start": round(t0, 2),
                                 "wall_s": round(time.time() - t0, 2)})
                return r

            for t in seq:
                done[t["name"]] = run_one(t)
                del pending[t["name"]]
            if par:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(len(par)) as ex:
                    futs = {ex.submit(run_one, t): t for t in par}
                    for f, t in futs.items():
                        done[t["name"]] = f.result()
                        del pending[t["name"]]
        return {"order": order, "results": done, "timeline": timeline,
                "monitor_tail": self.history[-8:],
                "throttled_chunks": self.stats["throttled_chunks"]}

    def _fetch_range(self, meta, start, length):
        out = bytearray()
        got = 0
        while got < length:
            pos = start + got
            seg = meta["segs"][pos // SEG_BYTES]
            off = pos % SEG_BYTES
            want = min(CHUNK, length - got, SEG_BYTES - off)
            data, _ = self._call(lambda e, s=seg, o=off, w=want:
                                 e.fetch_segment(s, o, w))
            self._govern()
            self.stats["sam_bytes"] += want
            out.extend(data)
            got += want
        return bytes(out)

    def iter_chunks(self, name, chunk=CHUNK):
        meta = self._segments[name]
        pos = 0
        while pos < meta["nbytes"]:
            yield pos, self._fetch_range(meta, pos, min(chunk, meta["nbytes"] - pos))
            pos += chunk

    def consume(self, name, on_chunk=None):
        """Stream a segment over the bus ONCE: on_chunk sees every chunk
        (secondary work), and the same pass repatriates the segment
        home. No second fetch for bring-back."""
        meta = self._segments[name]
        if meta["home"] is not None:
            if on_chunk is not None:
                data = self.home_bytes(name)
                for i in range(0, len(data), CHUNK):
                    on_chunk(data[i:i + CHUNK])
            return meta["tier"]
        need = meta["nbytes"]
        avail = psutil.virtual_memory().available
        if self.force_paging or avail < need * 1.2:
            os.makedirs(self.paging_dir, exist_ok=True)
            path = os.path.join(self.paging_dir, name + ".page")
            with open(path, "wb") as f:
                f.truncate(need)
            with open(path, "r+b") as f:
                mm = mmap.mmap(f.fileno(), need)
                for start, chunk in self.iter_chunks(name):
                    mm[start:start + len(chunk)] = chunk
                    if on_chunk is not None:
                        on_chunk(chunk)
                mm.flush()
                mm.close()
            meta["home"] = path
            meta["tier"] = "paging-file"
        else:
            buf = bytearray()
            for _, chunk in self.iter_chunks(name):
                buf.extend(chunk)
                if on_chunk is not None:
                    on_chunk(chunk)
            meta["home"] = bytes(buf)
            meta["tier"] = "ram"
        return meta["tier"]

    def prefetch_back(self, name):
        """Bring one segment home now (RAM tier, or paging tier if RAM
        cannot hold it). Returns where it landed."""
        meta = self._segments[name]
        if meta["home"] is not None:
            return meta["tier"]
        need = meta["nbytes"]
        avail = psutil.virtual_memory().available
        if self.force_paging or avail < need * 1.2:
            os.makedirs(self.paging_dir, exist_ok=True)
            path = os.path.join(self.paging_dir, name + ".page")
            with open(path, "wb") as f:
                f.truncate(need)
            with open(path, "r+b") as f:
                mm = mmap.mmap(f.fileno(), need)
                pos = 0
                for start, chunk in self.iter_chunks(name):
                    mm[start:start + len(chunk)] = chunk
                    pos += len(chunk)
                mm.flush()
                mm.close()
            meta["home"] = path
            meta["tier"] = "paging-file"
        else:
            meta["home"] = self._fetch_range(meta, 0, need)
            meta["tier"] = "ram"
        return meta["tier"]

    def home_bytes(self, name):
        meta = self._segments[name]
        if isinstance(meta["home"], (bytes, bytearray)):
            return bytes(meta["home"])
        with open(meta["home"], "rb") as f:
            return f.read()

    def complete(self):
        """Completion bring-back: everything home, digest-verified."""
        report = {}
        for name, meta in self._segments.items():
            tier = self.prefetch_back(name)
            ok = _sha(self.home_bytes(name)) == meta["sha"]
            report[name] = {"tier": tier, "digest_ok": ok,
                            "bytes": meta["nbytes"]}
            assert ok, f"bring-back digest mismatch: {name}"
        return report

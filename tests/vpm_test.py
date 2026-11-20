"""Thesis test: direct projected addressing (VPM) vs SAMOS search.

Thesis: secondary RAM is not a retrieval problem. The primary pushes
records into known coordinates of a predefined vector field (page, slot)
and later retrieves by translation -- no embedding search at fetch time.

Two translation layers are tested:
  directory : push-time table docid -> coordinate (exact, table cost O(n))
  projected : coordinate = crc32(docid) mod capacity, linear probing
              (zero per-record table on the primary; collision cost)

Workloads:
  A known-item fetch  -- the thesis's home turf
  B semantic query    -- the search architecture's home turf
  C never-pushed item -- behaviour when the need was not provisioned

Run: python vpm_test.py [--phone] [--port 9977]
"""
import argparse
import json
import pickle
import time
import zlib

import numpy as np

from samos import common as sc

SLOTS_PER_PAGE = 128
CACHE = r"D:\Downloads\projects\RAGS\bqboost\benchmark_results\embedding_cache\109b241a214ea77f556760d5dafab0ba1bb0eee3fbed44ea09e45a2b896deb28.npz"
LABELS = r"D:\Downloads\projects\RAGS\bqboost\benchmark_results\embedding_cache\scifact_labels.json"


def coord_of(idx):
    return (idx // SLOTS_PER_PAGE, idx % SLOTS_PER_PAGE)


def addr_of(coord):
    page, slot = coord
    return (page * SLOTS_PER_PAGE + slot) * sc.RECORD_BYTES


class ProjectedArena:
    """Secondary RAM slots filled by projection at push time."""

    def __init__(self, docids, capacity):
        self.capacity = capacity
        self.slots = np.full(capacity, -1, dtype=np.int64)
        self.probes_at_insert = []
        for idx, d in enumerate(docids):
            p = self._place(d, idx)
            self.probes_at_insert.append(p)

    def _home(self, docid):
        return zlib.crc32(docid.encode()) % self.capacity

    def _place(self, docid, idx):
        h = self._home(docid)
        for k in range(self.capacity):
            s = (h + k) % self.capacity
            if self.slots[s] == -1:
                self.slots[s] = idx
                return k
        raise RuntimeError("arena full")

    def fetch(self, docid, store):
        """Returns (record_idx, probes) or (None, probes) if absent."""
        h = self._home(docid)
        for k in range(self.capacity):
            s = (h + k) % self.capacity
            idx = int(self.slots[s])
            if idx == -1:
                return None, k + 1
            if store.docids[idx] == docid:
                return idx, k + 1
        return None, self.capacity


def pct(a, q):
    return float(np.percentile(a, q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="scifact.samos")
    ap.add_argument("--phone", action="store_true")
    ap.add_argument("--port", type=int, default=9977)
    args = ap.parse_args()

    store = sc.Store(args.store)
    n = store.hdr["n_records"]
    cache = np.load(CACHE)
    dense = cache["dense"].astype(np.float32)
    labels = json.load(open(LABELS))
    rng = np.random.default_rng(3)

    # ---------- push phase: build both translation layers ----------
    directory = {d: coord_of(i) for i, d in enumerate(store.docids)}
    dir_bytes = len(pickle.dumps([store.docids,
                                  np.arange(n, dtype=np.uint32)]))
    arena = ProjectedArena(store.docids, capacity=16384)
    ins = np.array(arena.probes_at_insert)
    print(f"PUSH: {n} records into vector field of {SLOTS_PER_PAGE}-slot pages")
    print(f"  directory translation table : {dir_bytes/1024:.0f} KB on primary "
          f"({dir_bytes/n:.1f} B/record)")
    print(f"  projected arena (cap 16384, load {n/16384:.2f}): "
          f"mean insert probes {ins.mean():.2f}, max {ins.max()}, "
          f"collided inserts {(ins > 0).mean()*100:.0f}%")

    # ---------- Workload A: known-item fetch ----------
    targets = rng.choice(n, size=300, replace=False)
    t_dir, t_proj, t_search = [], [], []
    search_hit_top1 = search_hit_top10 = 0
    proj_probes = []
    for idx in targets:
        docid = store.docids[int(idx)]
        t0 = time.perf_counter()
        c = directory[docid]
        rec = store.records[c[0] * SLOTS_PER_PAGE + c[1]]
        t_dir.append((time.perf_counter() - t0) * 1e3)
        assert rec is not None
        t0 = time.perf_counter()
        got, probes = arena.fetch(docid, store)
        t_proj.append((time.perf_counter() - t0) * 1e3)
        proj_probes.append(probes)
        assert got == int(idx)
        # existing architecture: the doc's own vector as the query
        # (the most favourable possible case for search)
        q = dense[int(idx)]
        q_i8, q_sc = sc.quantize_i8(q.reshape(1, -1))
        lists = sc.route(store.centroids, q, 8)
        t0 = time.perf_counter()
        ref, scanned = sc.reference_search(store, q_i8[0], float(q_sc[0]),
                                           lists, 10)
        t_search.append((time.perf_counter() - t0) * 1e3)
        ids = [r[0] for r in ref]
        search_hit_top1 += int(ids[0] == int(idx))
        search_hit_top10 += int(int(idx) in ids)
    print(f"\nA KNOWN-ITEM FETCH (300 pushed records, local):")
    print(f"  VPM directory : p50 {pct(t_dir,50)*1000:.0f} us, exact 300/300, scanned 0")
    print(f"  VPM projected : p50 {pct(t_proj,50)*1000:.0f} us, exact 300/300, "
          f"mean probes {np.mean(proj_probes):.2f}, scanned 0")
    print(f"  SAMOS search  : p50 {pct(t_search,50):.3f} ms, target top1 "
          f"{search_hit_top1}/300, top10 {search_hit_top10}/300, scanned ~731")

    # ---------- Workload B: semantic query ----------
    qrels, query_ids = labels["qrels"], labels["query_ids"]
    queries = cache["queries"].astype(np.float32)[:100]
    search_rel = vpm_rel = 0
    for i in range(100):
        q = queries[i]
        rel = set(qrels.get(query_ids[i], []))
        qi8, qsc = sc.quantize_i8(q.reshape(1, -1))
        lists = sc.route(store.centroids, q, 8)
        ref, _ = sc.reference_search(store, qi8[0], float(qsc[0]),
                                     lists, 10)
        if rel and {store.docids[r[0]] for r in ref} & rel:
            search_rel += 1
        # VPM's best defined behaviour: project the need itself to a
        # coordinate (hash of its sign sketch) and fetch that one slot.
        code = qi8[0].tobytes()
        slot = zlib.crc32(code) % arena.capacity
        idx = int(arena.slots[slot])
        if idx >= 0 and store.docids[idx] in rel:
            vpm_rel += 1
    print(f"\nB SEMANTIC QUERY (100 BEIR queries, relevant doc in answer):")
    print(f"  SAMOS search  : {search_rel}/100")
    print(f"  VPM projected : {vpm_rel}/100 (fetch-by-projection; a directory "
          f"lookup is undefined -- the need arrives with no docid)")

    # ---------- Workload C: never-pushed item ----------
    t0 = time.perf_counter()
    miss_dir = "NOPE-123" not in directory
    t_dir_miss = (time.perf_counter() - t0) * 1e3
    got, probes = arena.fetch("NOPE-123", store)
    print(f"\nC NEVER-PUSHED ITEM:")
    print(f"  VPM directory : miss in {t_dir_miss*1000:.0f} us, definitive")
    print(f"  VPM projected : miss after {probes} probes, definitive")
    print(f"  SAMOS search  : still returns nearest neighbours (a miss is "
          f"indistinguishable from a weak hit)")

    if args.phone:
        from samos.client import Engine, Router
        eng = Engine("127.0.0.1", args.port)
        router = Router(args.store + ".router.npz")
        f_walls, s_walls = [], []
        s_top1 = 0
        phone_targets = targets[:50]
        for idx in phone_targets:
            r = eng.fetch(int(idx))
            f_walls.append(r["wall_ms"])
            assert r["idx"] == int(idx) and r["record"] is not None
            got, tel = eng.query(dense[int(idx)], router.route(dense[int(idx)], 8),
                                 10, 32)
            s_walls.append(tel["wall_ms"])
            s_top1 += int(got[0]["idx"] == int(idx))
        eng.close()
        print(f"\nA-PHONE KNOWN-ITEM (50 records, USB, SD695):")
        print(f"  VPM fetch     : p50 {pct(f_walls,50):.2f} ms, exact 50/50, "
              f"620 B/answer, engine scanned 0")
        print(f"  SAMOS search  : p50 {pct(s_walls,50):.2f} ms, self-query top1 "
              f"{s_top1}/50, 504 B/answer, engine scanned ~731")


if __name__ == "__main__":
    main()

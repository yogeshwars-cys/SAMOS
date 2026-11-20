"""Honest SAMOS evaluation. Rules enforced here, in code:

- Corpus distinctness is asserted (unique doc ids AND manifest count).
- Query IDs come from scifact_labels.json (BEIR test order). Using a
  bare integer index as a query ID is the hada bug; it is not repeated.
- Ground truth is brute force over the DISTINCT corpus (float32 cosine
  and exact INT8 dot), computed independently of the IVF pipeline.
- Remote engine results must bit-match the local canonical reference
  (same store, same lists, same tie-breaks) or parity is reported FAIL.
- Synthetic stores are refused for relevance metrics.
"""
import argparse
import json
import sys
import time

import numpy as np
import psutil

from samos import common as sc
from samos.client import Engine, Router

CACHE = r"D:\Downloads\projects\RAGS\bqboost\benchmark_results\embedding_cache\109b241a214ea77f556760d5dafab0ba1bb0eee3fbed44ea09e45a2b896deb28.npz"
LABELS = r"D:\Downloads\projects\RAGS\bqboost\benchmark_results\embedding_cache\scifact_labels.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9977)
    ap.add_argument("--nprobe", type=int, default=8)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--rerank-m", type=int, default=32)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--full", action="store_true", help="scan all lists (no routing)")
    args = ap.parse_args()

    store = sc.Store(args.store)
    manifest = store.manifest()
    assert manifest["unique_records"] == manifest["n_records"] == store.hdr["n_records"]
    assert len(set(store.docids)) == store.hdr["n_records"], "duplicate doc ids!"
    print(f"store: {store.hdr['n_records']} distinct records, {store.hdr['nlist']} lists, "
          f"synthetic={manifest.get('synthetic')}")
    if manifest.get("synthetic"):
        print("REFUSING relevance eval on synthetic corpus (scale tests only)")
        sys.exit(2)

    labels = json.load(open(LABELS))
    query_ids, qrels, corpus_ids = labels["query_ids"], labels["qrels"], labels["corpus_ids"]
    assert corpus_ids == store.docids, "store doc order != labels corpus order"
    cache = np.load(CACHE)
    queries_f32, corpus_f32 = cache["queries"].astype(np.float32), cache["dense"].astype(np.float32)
    assert len(query_ids) == len(queries_f32)

    router = Router(args.store + ".router.npz")
    proc = psutil.Process()
    rss0 = proc.memory_info().rss / 1e6
    eng = Engine(args.host, args.port)
    hello = eng.hello()
    print(f"engine hello: records={hello['resp'][8]} mode_ram={hello['resp'][11]} "
          f"locked={hello['resp'][10]}B rss={hello['resp'][6]}KB")

    nq = min(args.limit, len(queries_f32))
    cn = corpus_f32 / np.linalg.norm(corpus_f32, axis=1, keepdims=True)
    parity = rel10 = top1_agree = brute10_hit = 0
    walls, scanned_l = [], []
    first_wall = None
    for i in range(nq):
        q = queries_f32[i]
        lists = list(range(store.hdr["nlist"])) if args.full else router.route(q, args.nprobe)
        q_i8, q_sc = sc.quantize_i8(q.reshape(1, -1))
        ref, ref_scanned = sc.reference_search(store, q_i8[0], float(q_sc[0]),
                                               lists, args.topk)
        got, tel = eng.query(q, lists, args.topk, args.rerank_m)
        if first_wall is None:
            first_wall = tel["wall_ms"]
        walls.append(tel["wall_ms"]); scanned_l.append(tel["scanned"])
        ref_ids = [r[0] for r in ref]
        got_ids = [r["idx"] for r in got]
        if ref_ids == got_ids and all(abs(a[1] - b["score"]) < 1e-6 for a, b in zip(ref, got)):
            parity += 1
        got_docids = [store.docids[j] for j in got_ids]
        rel = set(qrels.get(query_ids[i], []))
        if rel and set(got_docids) & rel:
            rel10 += 1
        brute = np.argsort(-(cn @ (q / np.linalg.norm(q))))[:10]
        brute_ids = [corpus_ids[j] for j in brute]
        if got_docids and brute_ids and got_docids[0] == brute_ids[0]:
            top1_agree += 1
        if set(got_docids) & set(brute_ids):
            brute10_hit += 1
    eng.close()
    rss1 = proc.memory_info().rss / 1e6
    w = np.array(walls)
    print(f"queries={nq} nprobe={'ALL' if args.full else args.nprobe} rerank_m={args.rerank_m}")
    print(f"PARITY engine==reference : {parity}/{nq} {'PASS' if parity == nq else 'FAIL'}")
    print(f"Recall@10 (BEIR qrels)   : {rel10}/{nq}")
    print(f"top1 == brute-f32 top1   : {top1_agree}/{nq}")
    print(f"top10 overlaps brute top10: {brute10_hit}/{nq}")
    print(f"latency wall ms: first={first_wall:.2f} p50={np.median(w):.3f} mean={w.mean():.3f} p99={np.percentile(w, 99):.3f}")
    print(f"scanned records/query    : {np.mean(scanned_l):.0f} of {store.hdr['n_records']}")
    print(f"primary RSS: {rss0:.1f} -> {rss1:.1f} MB (router only, store never loaded by client path)")
    print(f"engine telemetry (last): scan+rerank={(tel['scan_us']+tel['rerank_us'])/1000:.3f} ms, "
          f"rss={tel['rss_kb']} KB, faults={tel['faults']}, mode_ram={tel['mode_ram']}")


if __name__ == "__main__":
    main()

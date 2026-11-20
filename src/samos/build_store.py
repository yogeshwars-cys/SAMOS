"""Build a SAMOS store from distinct records. Duplicates are refused.

Two corpus sources:
  --source scifact : BEIR SciFact test split via the local embedding
                     cache + scifact_labels.json (correct query/corpus ID
                     mapping; the hada experiment mixed train/test IDs).
  --source synth   : distinctly generated clustered vectors, labeled
                     SYNTHETIC in the manifest. For engine scale tests
                     only; never used for relevance claims.
"""
import argparse
import hashlib
import json
import struct
import time

import numpy as np

from samos import common as sc


def spherical_kmeans(vecs_n, nlist, iters=20, seed=42):
    rng = np.random.default_rng(seed)
    init = rng.choice(len(vecs_n), size=min(nlist, len(vecs_n)), replace=False)
    cent = vecs_n[init].copy()
    labels = np.zeros(len(vecs_n), dtype=np.int32)
    for _ in range(iters):
        labels = np.argmax(vecs_n @ cent.T, axis=1)
        for c in range(len(cent)):
            m = vecs_n[labels == c]
            if len(m):
                v = m.mean(axis=0)
                n = np.linalg.norm(v)
                if n > 1e-12:
                    cent[c] = v / n
    labels = np.argmax(vecs_n @ cent.T, axis=1)
    return cent.astype(np.float32), labels


def dedupe_check(docids, i8_vecs):
    if len(set(docids)) != len(docids):
        raise SystemExit("FATAL: duplicate doc ids in corpus; refusing to build")
    seen = set()
    for i in range(len(i8_vecs)):
        h = hashlib.md5(i8_vecs[i].tobytes()).digest()
        if h in seen:
            raise SystemExit(f"FATAL: duplicate vector at row {i}; refusing to build")
        seen.add(h)


def build(vecs_f32, docids, payloads, nlist, out_path, manifest_extra):
    t0 = time.perf_counter()
    n = len(docids)
    i8, scales = sc.quantize_i8(vecs_f32)
    dedupe_check(docids, i8)
    norms = np.linalg.norm(vecs_f32, axis=1, keepdims=True)
    vecs_n = (vecs_f32 / np.maximum(norms, 1e-12)).astype(np.float32)
    centroids, labels = spherical_kmeans(vecs_n, nlist)

    records = np.zeros((n, sc.RECORD_BYTES), dtype=np.uint8)
    records[:, 0:4] = np.frombuffer(scales.tobytes(), dtype=np.uint8).reshape(n, 4)
    records[:, 4:388] = i8.view(np.uint8)
    for i, p in enumerate(payloads):
        raw = p.encode("utf-8", "ignore")[: sc.PAYLOAD_BYTES - 1]
        records[i, 388:388 + len(raw)] = list(raw)

    lists = [np.sort(np.where(labels == c)[0].astype(np.uint32)) for c in range(nlist)]
    list_data = np.concatenate(lists) if lists else np.zeros(0, dtype=np.uint32)
    list_off = np.zeros(nlist, dtype=np.uint64)
    pos = 0
    list_cnt = np.zeros(nlist, dtype=np.uint32)
    for c in range(nlist):
        list_off[c] = pos
        list_cnt[c] = len(lists[c])
        pos += len(lists[c])

    doc_blob = b"".join(d.encode("utf-8") for d in docids)
    doc_tab = np.zeros(n + 1, dtype=np.uint64)
    pos = 0
    for i, d in enumerate(docids):
        doc_tab[i] = pos
        pos += len(d.encode("utf-8"))
    doc_tab[n] = pos

    crcs = np.array([sc.crc32(records[i].tobytes()) for i in range(n)], dtype=np.uint32)

    manifest = {
        "format": "SAMOSDB1", "version": sc.VERSION, "dim": sc.DIM,
        "n_records": n, "unique_records": n, "nlist": nlist,
        "record_bytes": sc.RECORD_BYTES, "built_unix": int(time.time()),
        **manifest_extra,
    }
    manifest_b = json.dumps(manifest, indent=1).encode("utf-8")

    off = {}
    cur = sc.HEADER_BYTES
    def place(name, size):
        nonlocal cur
        off[name] = cur
        cur += size
    place("records", n * sc.RECORD_BYTES)
    place("centroids", nlist * sc.DIM * 4)
    place("lists", nlist * 12)
    place("list_data", len(list_data) * 4)
    place("docids", (n + 1) * 8 + len(doc_blob))
    place("crc", n * 4)
    place("manifest", len(manifest_b))

    lists_section = bytearray()
    for c in range(nlist):
        lists_section += struct.pack("<QI", int(list_off[c]), int(list_cnt[c]))

    with open(out_path, "wb") as f:
        f.write(sc.pack_header(n, nlist, off, len(manifest_b)))
        f.write(records.tobytes())
        f.write(centroids.astype("<f4").tobytes())
        f.write(bytes(lists_section))
        f.write(list_data.astype("<u4").tobytes())
        f.write(doc_tab.astype("<u8").tobytes())
        f.write(doc_blob)
        f.write(crcs.astype("<u4").tobytes())
        f.write(manifest_b)
    router_path = out_path + ".router.npz"
    np.savez(router_path, centroids=centroids, nlist=nlist)
    print(f"built {out_path}: {n} distinct records, {nlist} lists, "
          f"{cur/1e6:.2f} MB in {time.perf_counter()-t0:.1f}s")
    print(f"router (primary-side, centroids only): {router_path} "
          f"({centroids.nbytes/1024:.0f} KB)")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["scifact", "synth"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nlist", type=int, default=64)
    ap.add_argument("--cache", default=r"D:\Downloads\projects\RAGS\bqboost\benchmark_results\embedding_cache\109b241a214ea77f556760d5dafab0ba1bb0eee3fbed44ea09e45a2b896deb28.npz")
    ap.add_argument("--labels", default=r"D:\Downloads\projects\RAGS\bqboost\benchmark_results\embedding_cache\scifact_labels.json")
    ap.add_argument("--texts", default=r"D:\Downloads\projects\RAGS\bqboost\benchmark_results\embedding_cache\scifact_texts.json")
    ap.add_argument("--synth-records", type=int, default=262144)
    ap.add_argument("--synth-clusters", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if args.source == "scifact":
        data = np.load(args.cache)
        corpus = data["dense"].astype(np.float32)
        labels = json.load(open(args.labels))
        corpus_ids = labels["corpus_ids"]
        assert len(corpus_ids) == len(corpus), "corpus id/vector misalignment"
        texts = {d["_id"]: d for d in json.load(open(args.texts))["corpus"]}
        payloads = [f"{cid}|{texts[cid]['title'][:80]}" for cid in corpus_ids]
        sha = hashlib.sha256(open(args.cache, "rb").read()).hexdigest()
        build(corpus, corpus_ids, payloads, args.nlist, args.out, {
            "corpus": "BEIR SciFact (test-split queries; corpus = all docs)",
            "synthetic": False, "embedding_cache_sha256": sha,
            "query_ids_file": "scifact_labels.json:query_ids (BEIR test order)",
        })
    else:
        rng = np.random.default_rng(args.seed)
        centers = rng.standard_normal((args.synth_clusters, sc.DIM)).astype(np.float32)
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        assign = rng.integers(0, args.synth_clusters, size=args.synth_records)
        vecs = centers[assign] + 0.35 * rng.standard_normal((args.synth_records, sc.DIM)).astype(np.float32)
        vecs = (vecs / np.linalg.norm(vecs, axis=1, keepdims=True)).astype(np.float32)
        docids = [f"SYN{i:07d}" for i in range(args.synth_records)]
        payloads = [f"{d}|synthetic scale-test record {i}" for i, d in enumerate(docids)]
        build(vecs, docids, payloads, args.nlist, args.out, {
            "corpus": "synthetic distinct clustered vectors (scale test only)",
            "synthetic": True, "seed": args.seed,
            "synth_clusters": args.synth_clusters,
        })


if __name__ == "__main__":
    main()

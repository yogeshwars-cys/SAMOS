# Thesis test: secondary RAM with push-known placement (VPM) vs SAMOS search

Thesis: SAMOS is not a retrieval problem. Records are pushed into known
coordinates of a predefined vector field (page, slot) inside secondary
RAM; when the need comes, a translation layer converts the coordinate
directly to an address. No embedding, no search.

Implementation tested (`vpm_test.py`, engine cmd=2 FETCH):
- Field: 128-slot pages over the SAMOS store; coordinate = (page, slot);
  address = (page*128 + slot) * 512.
- directory mode: push-time table docid -> coordinate on the primary.
- projected mode: coordinate = crc32(docid) mod capacity, linear probing;
  no per-record table anywhere.

## Results (SciFact store, 5,183 distinct records)

| Test | VPM direct | SAMOS search (existing) |
|---|---|---|
| A. Known-item fetch, local | **3–9 µs, exact 300/300, scanned 0** | 0.90 ms p50, scanned ~731 (top1 300/300 only because the query was the record's own vector) |
| A. Known-item fetch, phone/USB | **5.62 ms p50, exact 50/50, 620 B, scanned 0** | 7.76 ms p50, 504 B, scanned ~731 |
| B. Semantic query (BEIR) | **0/100** — undefined for directory; fetch-by-projection finds nothing relevant | **71/100** relevant doc in top-10 |
| C. Never-pushed item | Definitive miss in ~1 µs / 1 probe | Returns neighbours anyway; a miss looks like a weak hit |

Costs and failure modes of the direct design:
- Directory table: 14.7 B/record on the primary (74 KB @ 5k records;
  ~29 MB @ 2M records, mostly docid strings).
- Projected mode: zero table, but mean 1.28 probes at load factor 0.32,
  17% of inserts collided, worst probe chain 14. Probe cost grows with
  load factor: the arena must be over-provisioned (wasted RAM) or
  rehashed on growth.
- Phone latency is transport-bound: direct fetch is only 1.4x faster
  than search at 5k records because USB RTT dominates both. The gap
  scales with corpus size — search cost grows with records scanned
  (262k full scan: 73.5 ms on phone), translation stays one round trip.

## Verdict

The thesis is **confirmed for provisioned needs and false as a total
replacement**:

- When the need arrives as an identity (record, tile, tensor block,
  session state — anything the primary itself pushed), direct translated
  fetch is exact, O(1), two orders of magnitude cheaper in compute,
  and can prove absence. Search is pure waste on this path.
- When the need arrives as a description ("something about X"), there
  is no coordinate to translate. Computing need -> coordinate *is*
  indexing/search by another name; VPM alone scores 0/100.

Correct architecture: **VPM is the memory layer; search demotes to an
optional need->coordinate translator** used only for unprovisioned,
semantic needs. SAMOS keeps both: cmd=1 (search, discovery) and cmd=2
(fetch, translation) now coexist in `samos_engine.c`.

# Fact-check: `hada_phone_experiment` (verified 2026-11-21)

Method: read every `.py`/`.c` source, re-derived every file size and
offset, independently re-ran the BQ scan + INT8 rerank in NumPy (no
bqboost), materialized the clustered 1 GiB slice on the laptop, queried
the live phone (RMX3660, SM6375 = Snapdragon 695, Android 14), and
compared all seven `*_results.json` files against recomputation.

## Valid (reproduced)

| # | Claim | Evidence |
|---|-------|----------|
| V1 | 1 GiB arena = 2,097,152 × 512 B; BQ index = 2,097,152 × 48 B = 100,663,296 B; base corpus = 5,183 × 512 B; centroids = (256, 384) f32 | File sizes exact to the byte |
| V2 | Arena needle doc `13734012` at record 1,234,567 (byte 632,098,304), payload == `base_corpus[2232]` | Read from `real_arena.sfdb` directly |
| V3 | Needle BQ rank 1 and INT8 rerank rank 1 (`benchmark_results.json`) | Independently reproduced; needle Hamming distance 129 is the global minimum; rerank score 0.39487 matches `firmware_benchmark_results.json` needle score 0.3948745 |
| V4 | Binaries are real AArch64 ELFs; phone files, sizes and firmware logs match the laptop copies; slice firmware parity hash `0x57d6c2647dcc239a` matches `head_to_head_1gb_ram.json` and the live phone log | ELF machine=183; `adb shell ls/cat` |
| V5 | Phone RAM-slice retrieval works live today | Fresh query over `adb forward tcp:9997`: region-220 query returns record 1,806,969 (needle), phone BQ 0.34 ms, wall 5.0 ms |
| V6 | Cluster spec is internally consistent: 256 lists, member counts sum 5,182, needle at region 220 offset 4729 = global 1,806,969 | Parsed `cluster_spec.bin` |

## Invalid / overstated (why SAMOS was built)

| # | Problem | Evidence |
|---|---------|----------|
| F1 | **The "2,097,152-record / 1 GiB knowledge base" contains 5,183 distinct documents.** Background docs are tiled `members[j % cnt]` into 8,192-slot regions. Regions hold 1–62 unique members (median 19). | Materialized slice: exactly 5,183 unique records; first 200k arena records contain 5,182 unique hashes. All "scan 65,536 records" numbers are ~99.8% redundant duplicate work; latency/recall do not describe a 2M-distinct corpus. |
| F2 | **`benchmark_client.py` is not a search benchmark.** Index is `np.random` unrelated to the arena; candidates come from `argpartition` on raw XOR integers of word 0 only (not popcount Hamming distance); the rerank uses a *new random* 388-dim query (including scale+payload bytes) different from the search query. Only the USB fetch timing is real. | Source lines 33, 50, 55, 90–94 |
| F3 | **Query-ID provenance is wrong.** The embedding cache holds BEIR **test** queries: `scifact_labels.json:query_ids = ['1','3','5',…]`. Scripts index it as train IDs (`queries[int("2")]`, train qrels file). The "Query 2 needle" is actually test query `5`; it passes only because train-Q2 and test-Q5 share doc `13734012`. In `real_benchmark.py` / `firmware_caller_bench.py`, queries 1..99 are random Gaussian vectors — no relevance meaning. | `scifact_labels.json` vs `generate_real_arena.py:70`, `real_benchmark.py:148` |
| F4 | **Two incompatible generations share one folder.** Arena generation: needle @1,234,567 (`real_arena.sfdb`, `bq_index.bin`, port 9998 firmware). Clustered generation: needle @1,806,969 (`cluster_spec.bin`, port 9997 slice firmware). `bq_index.bin` does not describe the clustered slice (the slice firmware recomputes BQ). Cross-generation result files are presented as one system. | V2 vs V6; port/magic audit |
| F5 | **Parity is not guaranteed; one run records the failure.** `sam_offload_results.json`: `parity_exact: false`. Causes: local search keeps per-region top-16 then merges, the phone keeps one joint top-16 (different candidate sets); `hada_ram_firmware.c` rerank has no index tie-break (`score > best` only). The 100/100 agreements elsewhere are inflated by F1: identical duplicate vectors make ties resolve identically almost by construction. | JSON + `hada_ram_firmware.c:250`, `sam_offload_bench.py:94-128` vs `hada_ram_slice_fw.c:222-275` |
| F6 | **"1.10 GiB resident in phone LPDDR" is not pinned.** Slice firmware uses plain `malloc` (no `mlock`). Live recheck: the same process reported 1,150,840 kB at benchmark time and 681,300 kB later (zRAM/reclaim; phone MemFree ≈ 110 MB). Unpinned "RAM" latency can include swap-in. | `hada_ram_slice_fw.c:137`, live `/proc/<pid>/status` |
| F7 | **"1,000 real queries" are ≤300 unique.** `big_task` samples with `rng.choice(..., replace=True)`; dual-task uses `(i*7+3) % 300`. | `big_task_sam_vs_normal.py`, `dual_task_sam_showdown.py:155` |
| F8 | **Headline ratios exclude real costs.** Per-query latency excludes arena generation, 1 GiB transfer to phone, index build and 0.53 s slice materialization. "131,072× data reduction" divides total arena size by per-query wire bytes — not by any baseline result size. | benchmark scripts' timing scopes |
| F9 | **No integrity or versioning anywhere.** Arena/BQ/spec files have no magic, version, or checksum; the stub zero-fills failed `pread`s silently and accepts unbounded offsets. No tests exist in the folder. | `hada_block_stub.c:98-102`, full-folder audit |

Note on the needle: brute-force float32 cosine over the distinct corpus
ranks doc `3413083` (0.4071) above the needle (0.3938, rank 2) for that
query. The pipeline still returns the needle because the BQ prefilter
drops the true top-1. "Needle rank 1" proves plumbing, not retrieval
correctness. SAMOS therefore reports Recall@10 against BEIR qrels and
agreement with brute-force ground truth instead of a needle alone.

## Verdict

Core plumbing is real: files, binaries, needle plumbing, live phone
retrieval, and the reported phone timings are consistent with
recomputation. The scientific claims built on top — a 2M-record corpus,
several of the benchmarks, bit-exact parity, pinned-RAM residency, and
query provenance — are not valid as stated. Rebuilt as SAMOS (this
folder) with distinct records, one canonical algorithm, checksummed
storage, pinned-RAM reporting, and ground-truth evaluation.

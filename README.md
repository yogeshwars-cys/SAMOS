![SAMOS logo](logo.png)

# SAMOS -- Secondary Access Memory Optimizing System

SAMOS turns a phone, SSD, or USB device into **SAM (Secondary Access Memory)**:
addressable overflow memory for programs that need to *have* more data than
they should *hold* in laptop RAM.

A program explicitly declares which of its memory is hot and which is
secondary. The hot part -- the absolute-necessary working set -- never leaves
the laptop. The secondary part is pushed to SAM under known coordinates and
brought back when the task completes. Secondary work can consume SAM data in
the background, more slowly than RAM, while the primary task runs at full
speed.

## What SAMOS is good for

- **Bloated desktop software:** move cold caches, tiles, session state, model
  weights, and background working sets out of the main RAM footprint.
- **RAM extension over USB:** a connected phone becomes overflow memory with
  measured ~5 ms record access instead of an opaque swap file.
- **Parallel task splitting:** Stage offloads free RAM, then use the freed
  headroom to run more heavy subtasks concurrently.
- **Background secondary work:** indexing, checksumming, aggregation, and
  verification can stream SAM segments while foreground work stays resident.
- **Vector/document stores:** a checksummed off-device corpus with routed
  exact INT8 search, deterministic parity against a local reference, and
  honest residency/fault telemetry.
- **Interruptible workflows:** if the cable is pulled, hot work keeps running;
  on reconnect the Fabric prefetches data home before secondary work resumes.

SAMOS is **not** a DRAM replacement and not yet a transparent Windows pagefile.
It is an explicit, program-controlled secondary-memory layer.

## Two usage paths

### 1. Fabric: secondary RAM for live programs

`samos_fabric.py` is the optimizer library. Programs declare segments in a
spec, offload non-hot data to the phone arena, run background work over the
bus, and complete with digest-verified bring-back:

```python
from samos_fabric import Fabric

fab = Fabric(port=9977)
fab.begin_task({"program": "my_app", "segments": [
    {"name": "weights_cache", "bytes": 48 * 1024 * 1024, "tier": "cold"},
    {"name": "session_kv", "bytes": 20 * 1024 * 1024, "tier": "cold"}]})
fab.offload("weights_cache", weights_bytes)
fab.consume("session_kv", on_chunk=index_chunk)   # one pass: work + repatriate
report = fab.complete()                            # everything home, verified
```

Fabric guarantees and fallbacks:

- **Completion bring-back:** every offloaded segment returns and is SHA-256
  verified before the task closes.
- **Cable-pull continuity:** a heartbeat detects link loss; hot memory is
  local by construction, and reconnect triggers prefetch-back.
- **Consumed-segment repatriation:** background work brings each segment home
  as soon as it has consumed it, in the same single bus pass.
- **Paging last resort:** if laptop RAM cannot hold a bring-back, the segment
  lands in a memory-mapped paging file and reports that tier.
- **Hot pinning:** hot arrays are best-effort `VirtualLock`ed on Windows.
- **Programmer-defined plans:** `run_plan` executes an explicit plan --
  offload order, which tasks are hot, which are bloated SAM consumers,
  which parallelize, and their dependencies. A conscious monitor tracks
  process RSS, system free RAM, on-SAM vs at-home bytes, and link state;
  while any hot task is active the governor throttles bloated consumers'
  bus bandwidth, so hot work always finishes first.

### 2. Store + engine: checksummed off-device corpora

`build_store.py` creates a `SAMOSDB1` store and refuses duplicate records.
The store carries fixed-size records, IVF lists, centroids, document IDs,
per-record CRC32 seals, and a provenance manifest in one file.
`samos_engine.c` serves it from the secondary device (mmap or RAM-staged +
`mlock`) and also serves direct coordinate FETCH and the writable Fabric
arena (1 MiB segments).

Canonical search is intentionally simple: route by centroids, then exact
INT8 dot over the selected lists ordered by `(score DESC, idx ASC)`. The C
engine must bit-match `samos_common.reference_search`; evaluation reports
FAIL if it does not.

## Measured results (2026-11-21)

Hardware: laptop + RMX3660 (Snapdragon 695, Android 14) over USB/adb forward.
All numbers are observed, sequential single-client, and transport-bound.

### Retrieval store (SciFact, 5,183 distinct docs, 64 lists, nprobe=8)

| Metric | Host engine | Phone engine |
|---|---:|---:|
| Engine/reference parity (100 queries) | 100/100 | 100/100 |
| Recall@10 (BEIR test qrels) | 73/100 | 73/100 |
| Top-1 agreement with brute-force f32 | 93/100 | 93/100 |
| Search wall latency p50 | 0.57 ms | 10.52 ms |
| Direct record FETCH p50 | -- | 4.87 ms |
| Records scanned/query | 731 | 731 |

Phone search compute is 1.43 ms of the 10.52 ms wall; the rest is USB/adb
transport. Store v2 is 2.88 MB and carries no sign-code section.

### Fabric offload test (miniature real Windows program)

A test program with ~28 MB hot memory and ~100 MB cold bloat (weights cache,
session tiles, KV text) offloads the bloat to the phone, runs a heavy matrix
task while indexing the offloaded segments, and survives a simulated cable
pull (forward removed for 3 s mid-run):

| Metric | HOME | SAM |
|---|---:|---:|
| Heavy task | 1.08 s | 1.19 s |
| Secondary indexing (100 MB) | 0.22 s local | 3.24 s over bus incl. outage |
| Push 100 MB to phone | -- | 3.26 s |
| Final correctness | baseline | Heavy + indexing digests identical |

### Staged offload test (freed RAM buys concurrency)

| Stage | Observed |
|---|---|
| HOME: 4 heavy subtasks, bloat resident | 1.14 s, peak RSS 289 MB |
| SAM stage 1 offload (60 MB) | RSS 154 -> 95 MB |
| SAM stage 2 offload (40 MB) | RSS 95 -> 60 MB |
| SAM: 8 heavy subtasks + indexing | 5.87 s, peak RSS 219 MB |

The freed ~94 MB ran twice the heavy subtasks plus background indexing at a
peak 70 MB below HOME. SAM is slower in wall time; its win is footprint and
capacity.

## Files

| File | Role |
|---|---|
| [samos_common.py](samos_common.py) | Store format, quantization, canonical reference search |
| [build_store.py](build_store.py) | Builds distinct-record SAMOS stores + laptop router |
| [samos_engine.c](samos_engine.c) | Secondary-side engine: search, FETCH, Fabric arena |
| [samos_client.py](samos_client.py) | Primary-side wire client |
| [samos_fabric.py](samos_fabric.py) | Optimizer library (declare/offload/consume/bring-back) |
| [mini_program.py](mini_program.py) | Miniature real program using the Fabric |
| [fabric_test.py](fabric_test.py) | HOME vs SAM offload test with cable pull |
| [staged_offload_test.py](staged_offload_test.py) | Staged offload -> concurrent heavy subtasks |
| [samos_eval.py](samos_eval.py) | Parity + recall evaluation vs brute-force truth |
| [vpm_test.py](vpm_test.py) | Direct-addressed (VPM) vs search thesis test |
| [test_samos.py](test_samos.py) | Store format, CRC tamper, host engine parity tests |
| [ram_scope_test.py](ram_scope_test.py) | 200 MB defined RAM scope: HOME exceeds it, SAM plan stays inside |

## Install and run

SAMOS is an installable Python library plus a portable C engine:

```powershell
git clone https://github.com/yogeshwars-cys/SAMOS.git
cd SAMOS
pip install -e .

# 1. Build a store (SciFact defaults point at a local embedding cache;
#    override with --cache/--labels/--texts)
samos-build --source scifact --out scifact.samos --nlist 64

# 2. Start the engine on this machine (builds the host binary if needed;
#    on the phone, cross-compile samos_engine.c with the NDK once and
#    run the same command line there)
samos-engine scifact.samos --mode ram --arena 64 --port 9977

# 3. Talk to it: identity, then a full offload/consume/bring-back run
samos-fabric hello
samos-fabric demo --mb 16

# 4. Evaluate retrieval parity/recall against a running engine
samos-eval --store scifact.samos --port 9977 --limit 100
```

Entry points: `samos-engine` (launch the C engine), `samos-fabric`
(hello/demo against the memory bus), `samos-build` (store builder),
`samos-eval` (parity + recall evaluation). The Fabric is also a plain
library: `from samos_fabric import Fabric` (see the Fabric example
above). Tests and experiment harnesses (`test_samos.py`,
`fabric_test.py`, `staged_offload_test.py`, `ram_scope_test.py`,
`vpm_test.py`, `mini_program.py`) are scripts in the repo root, not
part of the installed package.

## Build and test

```powershell
# Build the real store (paths to the SciFact embedding cache are arguments)
python build_store.py --source scifact --out scifact.samos --nlist 64

# Host engine + tests (Windows/MinGW)
gcc -O2 -o samos_engine_host.exe samos_engine.c -lws2_32
python test_samos.py

# Phone engine (Android NDK) + Fabric arena
aarch64-linux-android31-clang -O3 -fPIE -pie -o samos_engine_arm samos_engine.c
adb push samos_engine_arm /data/local/tmp/samos_engine
adb push scifact.samos /data/local/tmp/scifact.samos
adb shell "chmod +x /data/local/tmp/samos_engine"
adb shell "cd /data/local/tmp && nohup ./samos_engine ./scifact.samos --ram --verify --arena 112 --port 9977 > samos.log 2>&1 < /dev/null &"
adb forward tcp:9977 tcp:9977
python fabric_test.py
python staged_offload_test.py
python ram_scope_test.py
```

Generated stores (`*.samos`), routers, binaries, logs, and paging files are
build artifacts and are not committed.

## Honest limits

- The real corpus is 5,183 documents. Scale was separately measured on
  262,144 distinct synthetic records (host routed search p50 2.5 ms on the
  prior build); synthetic data is never used for relevance claims.
- Phone numbers are dominated by USB/adb transport (~9 ms of a 10.5 ms
  search). Faster links change the headline; the compute does not hide it.
- Android may reap a shell-launched engine under memory pressure. The phone
  engine reports locked bytes, RSS, and fault deltas per response, but a
  production deployment should run it under a supervisor/foreground service.
- Passive SSD/USB secondaries (no device-side compute) fit the same store
  and coordinate model with the laptop doing the scan, but that deployment
  mode is designed, not yet benchmarked.


## License

MIT -- see [LICENSE](LICENSE).

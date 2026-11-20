"""Fabric end-to-end test with a miniature real Windows process.

HOME run : mini_program keeps its ~100 MB cold bloat in laptop RAM.
SAM run  : the same program offloads the bloat to the phone arena,
           runs the heavy task on hot RAM while the secondary indexing
           task consumes segments over the bus -- and the cable is
           pulled (adb forward removed) mid-run, then restored.

Pass criteria: identical heavy signature, identical indexing digests,
SAM process RSS after offload far below its loaded RSS, and the SAM
run completes despite the pull.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ADB = r"D:\dev-cache\android-sdk\platform-tools\adb.exe"


def adb(*a):
    return subprocess.run([ADB, *a], capture_output=True, text=True)


def read_status(path):
    rows = []
    if os.path.exists(path):
        for line in open(path):
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run_program(mode, pull_after_offload=False):
    status = os.path.join(HERE, f"_status_{mode}.jsonl")
    if os.path.exists(status):
        os.remove(status)
    p = subprocess.Popen(
        [sys.executable, os.path.join(HERE, "mini_program.py"),
         "--mode", mode, "--status", status],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    pulled = False
    while p.poll() is None:
        if pull_after_offload and not pulled:
            stages = [r["stage"] for r in read_status(status)]
            if "offloaded" in stages:
                time.sleep(1.0)
                adb("forward", "--remove", "tcp:9977")   # cable pull
                print("  [harness] cable pulled mid-run")
                time.sleep(3.0)
                adb("forward", "tcp:9977", "tcp:9977")   # cable restored
                print("  [harness] cable restored; fabric must prefetch-back and continue")
                pulled = True
        time.sleep(0.2)
    out, err = p.communicate()
    if p.returncode != 0:
        print(out)
        print(err[-3000:])
        raise SystemExit(f"mini_program --mode {mode} failed rc={p.returncode}")
    return json.loads(out.strip().splitlines()[-1]), read_status(status)


def main():
    print("[1] HOME run (bloat stays in laptop RAM)")
    home, home_st = run_program("home")
    print(f"    heavy {home['heavy_s']}s, secondary {home['secondary_s']}s, "
          f"wall {home['wall_s']}s, final RSS {home['rss_mb']} MB")

    print("[2] SAM run (100 MB offloaded to phone, cable pulled mid-run)")
    sam, sam_st = run_program("sam", pull_after_offload=True)
    loaded = next(r for r in sam_st if r["stage"] == "loaded")
    offloaded = next(r for r in sam_st if r["stage"] == "offloaded")
    print(f"    push {sam['push_s']}s, heavy {sam['heavy_s']}s, "
          f"secondary {sam['secondary_s']}s, wall {sam['wall_s']}s")
    print(f"    process RSS: loaded {loaded['rss_mb']} MB -> offloaded "
          f"{offloaded['rss_mb']} MB (bloat reduction "
          f"{loaded['rss_mb'] - offloaded['rss_mb']:.0f} MB)")
    print(f"    bring-back tiers: "
          f"{ {k: v['tier'] for k, v in sam['bringback'].items()} }, "
          f"digests ok: {all(v['digest_ok'] for v in sam['bringback'].values())}")
    print(f"    degraded (link-loss) segments observed: {sam['degraded_seen']}")

    assert sam["heavy_sig"] == home["heavy_sig"], "heavy result diverged"
    assert sam["index_ok"] and home["index_ok"], "secondary indexing diverged"
    assert loaded["rss_mb"] - offloaded["rss_mb"] > 60, "bloat did not leave RAM"
    print("\nFABRIC TEST PASSED: identical results, bloat offloaded, "
          "cable pull survived")


if __name__ == "__main__":
    main()

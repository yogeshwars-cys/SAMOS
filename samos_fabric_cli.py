"""Command line access to the SAMOS Fabric memory bus.

    samos-fabric hello [--host H] [--port P]
    samos-fabric demo  [--host H] [--port P] [--mb 16] [--force-paging]

`demo` is the smallest end-to-end run: it declares one segment,
offloads it to a running engine's arena, consumes it back in one pass,
and completes with digest-verified bring-back. No test harness needed.
The engine must have been started with an arena at least --mb in size
(see `samos-engine ... --arena 64`).
"""
import argparse
import hashlib
import json
import sys
import time

import numpy as np

from samos_client import Engine
from samos_fabric import Fabric

MIB = 1024 * 1024


def cmd_hello(args) -> int:
    eng = Engine(args.host, args.port)
    try:
        hello = eng.hello()
    finally:
        eng.close()
    resp = hello["resp"]
    print(json.dumps({
        "records": resp[8], "unique_records": resp[9],
        "locked_bytes": resp[10], "mode_ram": bool(resp[11]),
        "engine_rss_kb": resp[6],
        "corpus": hello["manifest"].get("corpus"),
        "synthetic": hello["manifest"].get("synthetic"),
    }, indent=1))
    return 0


def cmd_demo(args) -> int:
    fab = Fabric(host=args.host, port=args.port,
                 force_paging=args.force_paging)
    try:
        data = np.random.default_rng(42).integers(
            0, 256, size=args.mb * MIB, dtype=np.uint8).tobytes()
        fab.begin_task({"program": "samos-fabric-demo", "segments": [
            {"name": "demo_data", "bytes": len(data), "tier": "cold"}]})
        t0 = time.time()
        fab.offload("demo_data", data)
        push_s = time.time() - t0

        digest = hashlib.sha256()
        t0 = time.time()
        tier = fab.consume("demo_data", on_chunk=digest.update)
        consume_s = time.time() - t0

        report = fab.complete()["demo_data"]
        print(json.dumps({
            "offloaded_mb": args.mb,
            "push_s": round(push_s, 2),
            "push_mb_s": round(args.mb / max(push_s, 1e-9), 1),
            "consume_s": round(consume_s, 2),
            "landed_tier": report["tier"],
            "digest_ok": report["digest_ok"] and
                         digest.hexdigest() == hashlib.sha256(data).hexdigest(),
            "engine_snapshot": fab.monitor_snapshot(),
        }, indent=1))
        return 0 if report["digest_ok"] else 1
    finally:
        fab.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="samos-fabric")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9977)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("hello", help="show engine/store identity")
    demo = sub.add_parser("demo", help="offload/consume/bring-back smoke run")
    demo.add_argument("--mb", type=int, default=16)
    demo.add_argument("--force-paging", action="store_true",
                      help="land the bring-back in the paging-file tier")
    args = ap.parse_args(argv)
    return cmd_hello(args) if args.cmd == "hello" else cmd_demo(args)


if __name__ == "__main__":
    sys.exit(main())

"""SAMOS tests: format, CRC tamper detection, canonical parity.

Run: python test_samos.py   (plain asserts, no pytest needed)
Requires scifact.samos built and samos_engine_host.exe compiled.
"""
import os
import struct
import subprocess
import sys
import time

import numpy as np

import samos_common as sc
from samos_client import Engine, Router

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(HERE, "scifact.samos")


def test_store_format():
    s = sc.Store(STORE)
    assert s.hdr["dim"] == 384 and s.hdr["record_bytes"] == 512
    # v2 store: no sign-code section (reserved header slot must be zero)
    with open(STORE, "rb") as f:
        hdr = f.read(64)
    version, = struct.unpack_from("<I", hdr, 8)
    reserved, = struct.unpack_from("<Q", hdr, 48)
    assert version == 2 and reserved == 0, "store must not carry a sign-code section"
    assert len(set(s.docids)) == s.hdr["n_records"], "doc ids not unique"
    vecs = s.i8
    hashes = {hash(vecs[i].tobytes()) for i in range(len(vecs))}
    # quantized vectors may legitimately collide after INT8 rounding for
    # near-duplicate docs; builder dedupe runs pre-rounding on float input
    # via int8 of distinct embeddings -- report, do not hide:
    print(f"  distinct int8 vectors: {len(hashes)}/{len(vecs)}")
    listed = sorted(np.concatenate(
        [s.list_data[s.list_off[c]:s.list_off[c] + s.list_cnt[c]]
         for c in range(s.hdr['nlist'])]).tolist())
    assert listed == list(range(s.hdr["n_records"])), "IVF lists must partition records"
    print("  format OK: header, partition, docids")


def test_crc_tamper():
    s = sc.Store(STORE)
    assert s.verify_crc() == 0, "clean store must verify"
    raw = bytearray(open(STORE, "rb").read())
    off = s.hdr["offsets"]["records"] + 1000
    raw[off] ^= 0xFF
    bad_path = os.path.join(HERE, "_tampered.samos")
    open(bad_path, "wb").write(bytes(raw))
    s2 = sc.Store(bad_path)
    assert s2.verify_crc() >= 1, "tampered record must be detected"
    s2.mm._mmap.close()
    del s2
    os.remove(bad_path)
    print("  crc OK: clean passes, 1-byte tamper detected")


def test_engine_parity():
    proc = subprocess.Popen(
        [os.path.join(HERE, "samos_engine_host.exe"), STORE, "--ram", "--port", "9979"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        time.sleep(1.0)
        s = sc.Store(STORE)
        router = Router(STORE + ".router.npz")
        cache = np.load(r"D:\Downloads\projects\RAGS\bqboost\benchmark_results\embedding_cache\109b241a214ea77f556760d5dafab0ba1bb0eee3fbed44ea09e45a2b896deb28.npz")
        eng = Engine("127.0.0.1", 9979)
        hello = eng.hello()
        assert hello["resp"][8] == s.hdr["n_records"]
        ok = 0
        for i in range(20):
            q = cache["queries"][i].astype(np.float32)
            lists = router.route(q, 8)
            q_i8, q_sc = sc.quantize_i8(q.reshape(1, -1))
            ref, _ = sc.reference_search(s, q_i8[0], float(q_sc[0]), lists, 10)
            got, tel = eng.query(q, lists, 10, 32, want_payload=True)
            assert [r["idx"] for r in got] == [r[0] for r in ref], f"parity idx q{i}"
            assert all(abs(r[1] - g["score"]) < 1e-6 for r, g in zip(ref, got)), f"parity score q{i}"
            assert got[0]["payload"].split(b"\x00")[0].decode().startswith(got[0]["docid"] + "|")
            ok += 1
        eng.close()
        print(f"  engine parity OK: {ok}/20 queries bit-match reference (RAM mode)")
    finally:
        proc.terminate()


if __name__ == "__main__":
    print("[1] store format"); test_store_format()
    print("[2] crc tamper"); test_crc_tamper()
    print("[3] engine parity (host)"); test_engine_parity()
    print("ALL TESTS PASSED")

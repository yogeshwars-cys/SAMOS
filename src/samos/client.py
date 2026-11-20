"""SAMOS primary-side client. Holds ONLY the router (centroids).

The primary never loads the store. It normalizes/quantizes the query
(shared code in samos.common, so encodings cannot drift), routes to
nprobe IVF lists, and asks the secondary engine for the canonical
search over those lists.
"""
import socket
import struct
import time

import numpy as np

from samos import common as sc

MAGIC = 0x4F4D4153
_REQ = struct.Struct("<IHHIIIf")  # + 48B padding + 384B i8 -> 456 bytes total
_RESP = struct.Struct("<IIIIQQIIIIQII")
_RESULT = struct.Struct("<IfI32s")
assert _RESP.size == 64 and _RESULT.size == 44


class Engine:
    def __init__(self, host="127.0.0.1", port=9977, timeout=30.0):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(timeout)
        self.sock.connect((host, port))

    def _read(self, n):
        buf = bytearray()
        while len(buf) < n:
            ch = self.sock.recv(n - len(buf))
            if not ch:
                raise ConnectionError("engine closed connection")
            buf.extend(ch)
        return bytes(buf)

    def hello(self):
        self.sock.sendall(_REQ.pack(MAGIC, 0, 0, 0, 0, 0, 0.0) + b"\x00" * 432)
        hdr = _RESP.unpack(self._read(_RESP.size))
        mlen, = struct.unpack("<I", self._read(4))
        import json
        return {"resp": hdr, "manifest": json.loads(self._read(mlen).decode())}

    def query(self, query_f32, list_ids, topk=10, rerank_m=32, want_payload=False):
        q = np.asarray(query_f32, dtype=np.float32).reshape(1, -1)
        q_i8, q_sc = sc.quantize_i8(q)
        hdr = _REQ.pack(MAGIC, 1, 1 if want_payload else 0,
                        len(list_ids), topk, rerank_m, float(q_sc[0]))
        pkt = hdr + b"\x00" * 48 + q_i8.astype(np.int8).tobytes()
        pkt += struct.pack(f"<{len(list_ids)}I", *[int(x) for x in list_ids])
        t0 = time.perf_counter()
        self.sock.sendall(pkt)
        resp = _RESP.unpack(self._read(_RESP.size))
        results = []
        for _ in range(resp[2]):
            idx, score, ham, docid = _RESULT.unpack(self._read(_RESULT.size))
            payload = self._read(124) if want_payload else None
            results.append({"idx": idx, "score": score, "ham": ham,
                            "docid": docid.split(b"\x00")[0].decode(),
                            "payload": payload})
        wall_ms = (time.perf_counter() - t0) * 1000.0
        telemetry = {"scanned": resp[3], "scan_us": resp[4], "rerank_us": resp[5],
                     "rss_kb": resp[6], "faults": resp[7], "mode_ram": resp[11],
                     "locked_bytes": resp[10], "wall_ms": wall_ms}
        return results, telemetry

    def fetch(self, idx, want_record=True):
        """Direct translated read (cmd=2): the primary computed this
        coordinate at push time; the engine does no search."""
        hdr = _REQ.pack(MAGIC, 2, 1 if want_record else 0, int(idx), 0, 0, 0.0)
        t0 = time.perf_counter()
        self.sock.sendall(hdr + b"\x00" * 432)
        resp = _RESP.unpack(self._read(_RESP.size))
        if resp[1] != 0:
            raise IndexError(f"fetch idx {idx}: engine status {resp[1]}")
        ridx, score, ham, docid = _RESULT.unpack(self._read(_RESULT.size))
        record = self._read(512) if want_record else None
        wall_ms = (time.perf_counter() - t0) * 1000.0
        return {"idx": ridx, "docid": docid.split(b"\x00")[0].decode(),
                "record": record, "wall_ms": wall_ms,
                "faults": resp[7], "rss_kb": resp[6]}

    def push_segment(self, seg, data, offset=0):
        """Fabric bus write (cmd=3): place bytes at (segment, offset)."""
        hdr = _REQ.pack(MAGIC, 3, 0, int(seg), int(offset), len(data), 0.0)
        t0 = time.perf_counter()
        self.sock.sendall(hdr + b"\x00" * 432 + bytes(data))
        resp = _RESP.unpack(self._read(_RESP.size))
        if resp[1] != 0:
            raise IOError(f"push_segment seg {seg}: engine status {resp[1]}")
        return (time.perf_counter() - t0) * 1000.0

    def fetch_segment(self, seg, offset, length):
        """Fabric bus read (cmd=4): bytes previously pushed to (segment, offset)."""
        hdr = _REQ.pack(MAGIC, 4, 0, int(seg), int(offset), int(length), 0.0)
        t0 = time.perf_counter()
        self.sock.sendall(hdr + b"\x00" * 432)
        resp = _RESP.unpack(self._read(_RESP.size))
        if resp[1] != 0:
            raise IOError(f"fetch_segment seg {seg}: engine status {resp[1]}")
        data = self._read(length)
        return data, {"wall_ms": (time.perf_counter() - t0) * 1000.0,
                      "faults": resp[7], "rss_kb": resp[6]}

    def close(self):
        self.sock.close()


class Router:
    def __init__(self, router_npz):
        d = np.load(router_npz)
        self.centroids = d["centroids"]

    def route(self, query_f32, nprobe):
        return sc.route(self.centroids, query_f32, nprobe)

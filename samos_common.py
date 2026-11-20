"""SAMOS shared definitions: store layout, quantization, canonical search.

Canonical query semantics (implemented identically in samos_engine.c):
  Exact INT8 dot product over every record in the selected IVF lists
  (or all lists in full mode): score = doc_scale * query_scale *
  dot(doc_i8, query_i8), ordered by (score DESC, record_idx ASC).
  The dot product is exact in int32; the scale product is float32.
  There is no approximate prefilter stage.

Quantization is done once, here, so the C engine and this reference can
never disagree about query encoding.
"""
import json
import struct
import zlib

import numpy as np

MAGIC = b"SAMOSDB1"
VERSION = 2  # v2: no sign-code section in the store
HEADER_BYTES = 128
DIM = 384
RECORD_BYTES = 512
VEC_OFF = 4
PAYLOAD_OFF = 388
PAYLOAD_BYTES = 124
PAD_WORDS = 6  # wire-protocol padding words (unused by the engine)

_HDR = struct.Struct("<8sIIIIQIIQQQQQQII")  # 8+4*4+8+8+8*6... see pack_header
assert _HDR.size <= HEADER_BYTES


def pack_header(n_records, nlist, offsets, manifest_len):
    b = bytearray(HEADER_BYTES)
    struct.pack_into("<8s", b, 0, MAGIC)
    struct.pack_into("<I", b, 8, VERSION)
    struct.pack_into("<I", b, 12, HEADER_BYTES)
    struct.pack_into("<I", b, 16, DIM)
    struct.pack_into("<I", b, 20, RECORD_BYTES)
    struct.pack_into("<Q", b, 24, n_records)
    struct.pack_into("<I", b, 32, nlist)
    struct.pack_into("<I", b, 36, 0)
    struct.pack_into("<Q", b, 40, offsets["records"])
    struct.pack_into("<Q", b, 48, 0)
    struct.pack_into("<Q", b, 56, offsets["centroids"])
    struct.pack_into("<Q", b, 64, offsets["lists"])
    struct.pack_into("<Q", b, 72, offsets["list_data"])
    struct.pack_into("<Q", b, 80, offsets["docids"])
    struct.pack_into("<Q", b, 88, offsets["crc"])
    struct.pack_into("<I", b, 96, manifest_len)
    struct.pack_into("<Q", b, 104, offsets["manifest"])
    return bytes(b)


def parse_header(buf):
    if len(buf) < HEADER_BYTES or buf[0:8] != MAGIC:
        raise ValueError("not a SAMOS store (bad magic)")
    version, = struct.unpack_from("<I", buf, 8)
    if version != VERSION:
        raise ValueError(f"unsupported SAMOS store version {version}")
    dim, rec_bytes = struct.unpack_from("<II", buf, 16)
    n_records, = struct.unpack_from("<Q", buf, 24)
    nlist, = struct.unpack_from("<I", buf, 32)
    offs = {}
    for i, name in enumerate(["records", "reserved", "centroids", "lists",
                              "list_data", "docids", "crc"]):
        offs[name], = struct.unpack_from("<Q", buf, 40 + i * 8)
    manifest_len, = struct.unpack_from("<I", buf, 96)
    offs["manifest"], = struct.unpack_from("<Q", buf, 104)
    return {"n_records": n_records, "nlist": nlist, "dim": dim,
            "record_bytes": rec_bytes,
            "offsets": offs, "manifest_len": manifest_len}


def crc32(data):
    return zlib.crc32(data) & 0xFFFFFFFF


def quantize_i8(vecs):
    """Per-row symmetric INT8. Returns (int8 array, float32 scales)."""
    vecs = np.asarray(vecs, dtype=np.float32)
    amax = np.maximum(np.abs(vecs).max(axis=1, keepdims=True), 1e-12)
    scales = (amax / 127.0).astype(np.float32).squeeze(-1)
    q = np.clip(np.round(vecs * (127.0 / amax)), -127, 127).astype(np.int8)
    return q, scales


class Store:
    """Memory-mapped read-only SAMOS store (the storage-native layout)."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            hdr_buf = f.read(HEADER_BYTES)
        self.hdr = parse_header(hdr_buf)
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")
        o = self.hdr["offsets"]
        n = self.hdr["n_records"]
        self.records = self.mm[o["records"]:o["records"] + n * RECORD_BYTES].reshape(n, RECORD_BYTES)
        self.scales = np.frombuffer(
            self.mm[o["records"]:o["records"] + n * RECORD_BYTES].tobytes(),
            dtype=np.uint8).reshape(n, RECORD_BYTES)[:, 0:4]
        self.scales = np.ascontiguousarray(self.scales).view("<f4").reshape(n)
        self.i8 = np.ascontiguousarray(
            np.frombuffer(self.records.tobytes(), dtype=np.uint8)
            .reshape(n, RECORD_BYTES)[:, VEC_OFF:VEC_OFF + DIM]).view(np.int8).reshape(n, DIM)
        nlist = self.hdr["nlist"]
        cent_raw = self.mm[o["centroids"]:o["centroids"] + nlist * DIM * 4]
        self.centroids = np.frombuffer(cent_raw.tobytes(), dtype="<f4").reshape(nlist, DIM).copy()
        lists_raw = self.mm[o["lists"]:o["lists"] + nlist * 12]
        lv = np.frombuffer(lists_raw.tobytes(), dtype=np.uint8).reshape(nlist, 12)
        self.list_off = np.ascontiguousarray(lv[:, 0:8]).view("<u8").reshape(nlist)
        self.list_cnt = np.ascontiguousarray(lv[:, 8:12]).view("<u4").reshape(nlist)
        total_listed = int(self.list_cnt.sum())
        ld_raw = self.mm[o["list_data"]:o["list_data"] + total_listed * 4]
        self.list_data = np.frombuffer(ld_raw.tobytes(), dtype="<u4").copy()
        self.docids = self._load_docids()

    def _load_docids(self):
        o = self.hdr["offsets"]
        n = self.hdr["n_records"]
        tab_raw = self.mm[o["docids"]:o["docids"] + (n + 1) * 8]
        tab = np.frombuffer(tab_raw.tobytes(), dtype="<u8")
        blob_start = o["docids"] + (n + 1) * 8
        blob_len = int(tab[-1])
        blob = bytes(self.mm[blob_start:blob_start + blob_len])
        return [blob[tab[i]:tab[i + 1]].decode("utf-8") for i in range(n)]

    def manifest(self):
        o = self.hdr["offsets"]
        raw = bytes(self.mm[o["manifest"]:o["manifest"] + self.hdr["manifest_len"]])
        return json.loads(raw.decode("utf-8"))

    def payload(self, idx):
        return bytes(self.records[idx, PAYLOAD_OFF:RECORD_BYTES]).split(b"\x00")[0].decode("utf-8", "ignore")

    def verify_crc(self):
        """Verify every record CRC. Returns number of corrupt records."""
        o = self.hdr["offsets"]
        n = self.hdr["n_records"]
        stored = np.frombuffer(
            bytes(self.mm[o["crc"]:o["crc"] + n * 4]), dtype="<u4")
        bad = 0
        rec = np.frombuffer(self.records.tobytes(), dtype=np.uint8).reshape(n, RECORD_BYTES)
        for i in range(n):
            if crc32(rec[i].tobytes()) != int(stored[i]):
                bad += 1
        return bad


def route(centroids, query_f32, nprobe):
    """Return list ids ordered by (centroid cosine DESC, list id ASC)."""
    q = np.asarray(query_f32, dtype=np.float32)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    sims = centroids @ q
    order = sorted(range(len(sims)), key=lambda i: (-float(sims[i]), i))
    return order[:nprobe]


def reference_search(store, query_i8, query_scale, list_ids, topk=10):
    """Canonical search: exact INT8 dot over every record in the lists.

    Returns list of (idx, score, 0), best first, and the scanned count.
    """
    q_i8 = np.asarray(query_i8, dtype=np.int8)
    out = []
    scanned = 0
    for lid in list_ids:
        start = int(store.list_off[lid])
        cnt = int(store.list_cnt[lid])
        if cnt == 0:
            continue
        scanned += cnt
        members = store.list_data[start:start + cnt]
        dots = store.i8[members].astype(np.int32) @ q_i8.astype(np.int32)
        for j in range(cnt):
            idx = int(members[j])
            score = float(np.float32(store.scales[idx]) * np.float32(query_scale)
                          * np.float32(int(dots[j])))
            out.append((idx, score, 0))
    out.sort(key=lambda t: (-t[1], t[0]))
    return out[:topk], scanned

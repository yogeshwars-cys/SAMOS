"""Launch the SAMOS C engine on this machine.

    samos-engine <store.samos> [--port 9977] [--mode ram|mmap]
                 [--arena 112] [--verify] [--binary PATH] [--no-build]

If no engine binary is found next to this module (or in the current
directory), the host engine is built from samos_engine.c with the
platform compiler (gcc on Windows plus -lws2_32, cc elsewhere).
For Android, cross-compile once with the NDK and pass --binary:

    aarch64-linux-android31-clang -O3 -fPIE -pie -o samos_engine engine/samos_engine.c
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _engine_c() -> Path | None:
    for cand in [os.environ.get("SAMOS_ENGINE_C"),
                 Path(__file__).resolve().parents[2] / "engine" / "samos_engine.c",
                 Path(__file__).with_name("samos_engine.c"),
                 Path.cwd() / "engine" / "samos_engine.c",
                 Path.cwd() / "samos_engine.c"]:
        if cand and Path(cand).exists():
            return Path(cand)
    return None


def _default_binary() -> Path:
    name = "samos_engine_host.exe" if os.name == "nt" else "samos_engine"
    for base in [Path(__file__).resolve().parents[2], Path(__file__).parent, Path.cwd()]:
        if (base / name).exists():
            return base / name
    return Path(__file__).resolve().parents[2] / name


def build_engine(binary: Path) -> None:
    src = _engine_c()
    if src is None:
        raise SystemExit("samos_engine.c not found; clone the SAMOS repo "
                         "or set SAMOS_ENGINE_C to the engine source.")
    if os.name == "nt":
        cc = shutil.which("gcc") or shutil.which("clang")
        if not cc:
            raise SystemExit("need gcc (MinGW) on PATH to build the host engine")
        cmd = [cc, "-O2", "-o", str(binary), str(src), "-lws2_32"]
    else:
        cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
        if not cc:
            raise SystemExit("need a C compiler on PATH to build the host engine")
        cmd = [cc, "-O2", "-o", str(binary), str(src)]
    print("building:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="samos-engine")
    ap.add_argument("store", help="path to a .samos store file")
    ap.add_argument("--port", type=int, default=9977)
    ap.add_argument("--mode", choices=["ram", "mmap"], default="ram")
    ap.add_argument("--arena", type=int, default=0,
                    help="writable Fabric arena size in MiB (0 = none)")
    ap.add_argument("--verify", action="store_true",
                    help="verify all record CRCs at boot and refuse corruption")
    ap.add_argument("--binary", default=None, help="prebuilt engine binary")
    ap.add_argument("--no-build", action="store_true",
                    help="fail instead of building a missing host binary")
    args = ap.parse_args(argv)

    binary = Path(args.binary) if args.binary else _default_binary()
    if not binary.exists():
        if args.no_build:
            raise SystemExit(f"engine binary not found: {binary}")
        build_engine(binary)

    cmd = [str(binary), args.store, f"--{args.mode}",
           "--port", str(args.port)]
    if args.arena > 0:
        cmd += ["--arena", str(args.arena)]
    if args.verify:
        cmd.append("--verify")
    print("starting:", " ".join(cmd))
    proc = subprocess.Popen(cmd)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        return 130


if __name__ == "__main__":
    sys.exit(main())

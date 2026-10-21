#!/usr/bin/env python
"""How many rounds of fragments are needed to reconstruct client 0.

One source only: fragments_eps_1e+30_csf1e8 (FedBYOL, CSF=1e8, ε=1e30).
Server round 1 saved nothing (empty rounds are not written), so the files on
disk are round_002.pt .. round_100.pt. For k = 1..100 this runs the same
scoring path as dp_eps_sweep.postprocess on every saved round whose server
index is <= k:

    regroup_fragments.py   (eval_model_eps_1e+30_csf1e8.pth, default K_NN=99)
    infer_fragments.py     (reconstruction_out/best_leak_down32_lpips.pt,
                            MIN_VIEWS=8)
    compare_cid0.py        LPIPS bins

The recorded score is the compare_cid0 image counts

    success = (n[0.00-0.20] + n[0.20-0.30] + n[0.30-0.40] / 2) / 100

Client 0 holds 10000 images, so this is the percent of that client with full
credit for LPIPS < 0.30 and half credit for the 0.30-0.40 bin. The full
99-file run that produced the reference table scores 39.915.

The raw fragment directory is only read, through symlinks. Cluster copies and
reconstruction PNGs are deleted after each row. compare_cid0 writes
cid0_compare/; the pre-sweep montages are parked and put back at exit.

Resumable: a through_round already present in the csv is not repeated.

    python round_frag_sweep.py
"""
from __future__ import annotations

import csv
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRAG_SRC = ROOT / "fragments_eps_1e+30_csf1e8"
EVAL_CKPT = ROOT / "eval_model_eps_1e+30_csf1e8.pth"
RECON_CKPT = ROOT / "reconstruction_out" / "best_leak_down32_lpips.pt"
WORK = ROOT / "round_sweep_work"
LINK_DIR = WORK / "frags"
CLUSTER_DIR = WORK / "clustered"
RECON_DIR = WORK / "recons"
OUT = ROOT / "round_sweep_out"
CSV_PATH = OUT / "fragment_rounds.csv"
PID_PATH = OUT / "driver.pid"
QA_DIR = ROOT / "cid0_compare"
QA_BAK = ROOT / "cid0_compare_pre_round_sweep"

LAST_ROUND = 100
MIN_VIEWS = "8"
MIN_FREE_GIB = 100.0
PY = sys.executable

BIN_KEYS = (
    "0.00-0.20", "0.20-0.30", "0.30-0.40", "0.40-0.50",
    "0.50-0.60", "0.60-0.72", "0.72-1.00",
)
# Full credit below 0.30, half credit on 0.30-0.40, ignore the rest.
# Division by 100 is the client's 10000 images expressed as a percent.
SUCCESS_BINS = ("0.00-0.20", "0.20-0.30", "0.30-0.40")

_RE_USABLE = re.compile(r"(\d+) usable fragments")
_RE_WROTE = re.compile(r"wrote (\d+) bins / (\d+) fragments -> .+ \(max (\d+), median (\d+)\)")
_RE_NOTHING = re.compile(r"nothing to cluster")
_RE_RECONS = re.compile(r"(\d+) reconstructions in ")
_RE_BIN = re.compile(r"(?m)^\s*(\d+\.\d+-\d+\.\d+)\s+(\d+)\s+\d+\.\d+\s+[0-9.]+")
_RE_NEAREST = re.compile(
    r"distinct client-0 images leaked\s*:\s*(\d+)/(\d+)\s+\(([0-9.]+)%")
_RE_WEIGHTED = re.compile(
    r"precision-weighted\):\s*([0-9.]+)/(\d+)\s+\(([0-9.]+)%\)")

COLUMNS = (
    ["through_round", "n_files", "n_usable", "n_bins", "n_clustered",
     "max_cluster", "median_cluster", "n_recons"]
    + [f"lp_{k}" for k in BIN_KEYS]
    + ["success", "nearest_pct", "weighted_pct", "seconds"]
)

_child: subprocess.Popen | None = None
_qa_parked = False
_cleaned = False


def _free_gib() -> float:
    return shutil.disk_usage(ROOT).free / (1024 ** 3)


def _under_work(path: Path) -> Path:
    """Refuse to delete anything outside the scratch tree."""
    resolved = path.resolve()
    if WORK.resolve() not in resolved.parents and resolved != WORK.resolve():
        raise RuntimeError(f"refusing to delete {resolved}")
    return resolved


def _rm_tree(path: Path) -> None:
    resolved = _under_work(path)
    if resolved.is_symlink():
        resolved.unlink()
    elif resolved.exists():
        shutil.rmtree(resolved)


def _cleanup_work() -> None:
    _rm_tree(CLUSTER_DIR)
    _rm_tree(RECON_DIR)
    _rm_tree(LINK_DIR)


def _park_qa() -> None:
    global _qa_parked
    if QA_DIR.exists() and not QA_BAK.exists():
        shutil.move(str(QA_DIR), str(QA_BAK))
        print(f"[rounds] parked {QA_DIR.name} -> {QA_BAK.name}", flush=True)
    elif QA_DIR.exists() and QA_BAK.exists():
        shutil.rmtree(QA_DIR)
    _qa_parked = True


def _drop_sweep_qa() -> None:
    """Remove montages this sweep just wrote. The parked original stays."""
    if _qa_parked and QA_DIR.exists() and QA_BAK.exists():
        shutil.rmtree(QA_DIR)
    elif _qa_parked and QA_DIR.exists() and not QA_BAK.exists():
        shutil.rmtree(QA_DIR)


def _restore_qa() -> None:
    global _cleaned
    if _cleaned:
        return
    _drop_sweep_qa()
    if QA_BAK.exists():
        if QA_DIR.exists():
            shutil.rmtree(QA_DIR)
        shutil.move(str(QA_BAK), str(QA_DIR))
        print(f"[rounds] restored {QA_DIR.name}", flush=True)
    _cleaned = True


def _rounds_on_disk() -> list[Path]:
    files = []
    for path in FRAG_SRC.glob("round_*.pt"):
        files.append((int(path.stem.split("_")[1]), path))
    files.sort()
    return [path for _, path in files]


def _link(files: list[Path]) -> None:
    _rm_tree(LINK_DIR)
    LINK_DIR.mkdir(parents=True)
    for src in files:
        os.symlink(src.resolve(), LINK_DIR / src.name)


def _run(script: str, env: dict, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    penv = {
        **env,
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONUNBUFFERED": "1",
        "SSL_METHOD": "byol",
    }
    print(f"[rounds] {script} -> {log.name}", flush=True)
    global _child
    with open(log, "w") as handle:
        proc = subprocess.Popen(
            [PY, "-u", script],
            env=penv,
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd=str(ROOT),
        )
        _child = proc
        try:
            rc = proc.wait()
        finally:
            _child = None
    if rc != 0:
        tail = log.read_text(errors="replace")[-2500:]
        raise RuntimeError(f"{script} exited {rc}\n{tail}")


def success(bins: dict[str, int]) -> float:
    full = bins.get("0.00-0.20", 0) + bins.get("0.20-0.30", 0)
    half = bins.get("0.30-0.40", 0) / 2
    return (full + half) / 100


def parse_bins(text: str) -> dict[str, int]:
    return {m.group(1): int(m.group(2)) for m in _RE_BIN.finditer(text)}


def _blank(through: int, n_files: int) -> dict:
    row = {c: 0 for c in COLUMNS}
    row["through_round"] = through
    row["n_files"] = n_files
    for key in ("success", "nearest_pct", "weighted_pct"):
        row[key] = 0.0
    return row


def _load_rows() -> list[dict]:
    if not CSV_PATH.exists():
        return []
    with open(CSV_PATH, newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = CSV_PATH.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: int(r["through_round"])):
            writer.writerow(row)
    tmp.replace(CSV_PATH)


def _score(through: int, files: list[Path]) -> dict:
    """Regroup, infer, compare one prefix. Caller deletes the scratch copies."""
    row = _blank(through, len(files))
    if not files:
        print(f"[rounds] through r{through}: no fragment file, success=0", flush=True)
        return row

    free = _free_gib()
    if free < MIN_FREE_GIB:
        raise RuntimeError(f"only {free:.0f} GiB free; need {MIN_FREE_GIB:.0f} before clustering")

    _link(files)
    regroup_log = OUT / f"regroup_r{through:03d}.log"
    infer_log = OUT / f"infer_r{through:03d}.log"
    compare_log = OUT / f"compare_r{through:03d}.log"

    base = os.environ.copy()
    _run("regroup_fragments.py", {
        **base,
        "FRAG_DIR": str(LINK_DIR),
        "CKPT": str(EVAL_CKPT),
        "OUT_DIR": str(CLUSTER_DIR),
    }, regroup_log)
    regroup_text = regroup_log.read_text(errors="replace")
    usable = _RE_USABLE.search(regroup_text)
    row["n_usable"] = int(usable.group(1)) if usable else 0
    wrote = _RE_WROTE.search(regroup_text)
    if wrote:
        row["n_bins"] = int(wrote.group(1))
        row["n_clustered"] = int(wrote.group(2))
        row["max_cluster"] = int(wrote.group(3))
        row["median_cluster"] = int(wrote.group(4))
    if _RE_NOTHING.search(regroup_text):
        print(f"[rounds] through r{through}: nothing clustered, success=0", flush=True)
        return row
    if not wrote:
        raise RuntimeError(f"regroup log has no cluster summary\n{regroup_text[-1500:]}")

    _run("infer_fragments.py", {
        **base,
        "FRAG_DIR": str(CLUSTER_DIR),
        "MIN_VIEWS": MIN_VIEWS,
        "RECON_SUBDIR": str(RECON_DIR),
        "CKPT": str(RECON_CKPT),
    }, infer_log)
    infer_text = infer_log.read_text(errors="replace")
    got = _RE_RECONS.search(infer_text)
    n_png = len(list(RECON_DIR.glob("*.png"))) if RECON_DIR.is_dir() else 0
    row["n_recons"] = int(got.group(1)) if got else n_png
    if n_png == 0:
        print(f"[rounds] through r{through}: 0 reconstructions, success=0", flush=True)
        return row

    _run("compare_cid0.py", {
        **base,
        "RECON_DIR": str(RECON_DIR),
        "CLUSTER_DIR": str(CLUSTER_DIR),
    }, compare_log)
    text = compare_log.read_text(errors="replace")
    bins = parse_bins(text)
    missing = [k for k in SUCCESS_BINS if k not in bins]
    if missing:
        raise RuntimeError(f"compare_cid0 log missing bins {missing}")
    near, weight = _RE_NEAREST.search(text), _RE_WEIGHTED.search(text)
    if not near or not weight:
        raise RuntimeError("compare_cid0 log has no leak summary")
    for key in BIN_KEYS:
        row[f"lp_{key}"] = bins.get(key, 0)
    row["success"] = f"{success(bins):.4f}"
    row["nearest_pct"] = near.group(3)
    row["weighted_pct"] = weight.group(3)
    return row


def _on_signal(signum, _frame):
    print(f"[rounds] signal {signum}, stopping", flush=True)
    child = _child
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill()
    _cleanup_work()
    _restore_qa()
    raise SystemExit(128 + signum)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PID_PATH.write_text(str(os.getpid()) + "\n")
    if not FRAG_SRC.is_dir():
        raise SystemExit(f"missing {FRAG_SRC}")
    if not EVAL_CKPT.is_file() or not RECON_CKPT.is_file():
        raise SystemExit("missing eval model or inverter checkpoint")

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    _park_qa()
    try:
        files = _rounds_on_disk()
        print(f"[rounds] {len(files)} fragment files in {FRAG_SRC.name}: "
              f"{files[0].name} .. {files[-1].name}", flush=True)
        print("[rounds] success = (n[0.00-0.20] + n[0.20-0.30] + n[0.30-0.40] / 2) / 100",
              flush=True)
        rows = _load_rows()
        done = {int(r["through_round"]) for r in rows}
        for k in range(1, LAST_ROUND + 1):
            if k in done:
                print(f"[rounds] through r{k}: already in {CSV_PATH.name}", flush=True)
                continue
            use = [p for p in files if int(p.stem.split("_")[1]) <= k]
            t0 = time.time()
            _cleanup_work()
            row = _score(k, use)
            row["success"] = f"{float(row['success']):.4f}"
            row["seconds"] = round(time.time() - t0)
            rows.append(row)
            _write_csv(rows)
            _cleanup_work()
            _drop_sweep_qa()
            print(f"[rounds] through r{k}: files={row['n_files']} "
                  f"bins={row['n_bins']} recons={row['n_recons']} "
                  f"success={row['success']} ({row['seconds']}s)", flush=True)
        print(f"[rounds] wrote {CSV_PATH}", flush=True)
    finally:
        _cleanup_work()
        _restore_qa()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Sweep DP_size and collect the privacy/utility trade-off.

Produces the two axes of the claim "standard DP either destroys the model or does
not stop extraction":

  * utility  -- knn_acc, read from the server's per-round eval
  * leak     -- LEAK_NOISE = sigma / median(m_fired), the in-run reconstruction
                noise proxy the server already computes (server.py ~line 525).
                LEAK_NOISE >= 1 means a fragment is drowned in noise; << 1 means
                it reconstructs cleanly.

WHY THE size GRID LOOKS ABSURD
-----------------------------
It is not a typo. LOKI's trap makes the transmitted update norm ~8.3e5 while the
honest part is ~9 (measured: DP_CALIBRATE=1), so the L2 sensitivity a flat clip
must bound is ~1e5x the signal being protected. Client-level local DP therefore
prices the noise that actually kills the leak (sigma ~ 5e-4) at size ~ 1e21. The
sweep spans the transition wherever it happens to sit; that it sits at 1e17..1e22
IS the finding.

Anchors worth keeping: size=1e2 is a real guarantee (model annihilated), size=1e24
is effectively no noise (leak intact).

USAGE
    python dp_size_sweep.py              # run the full grid, resumable, both GPUs
    python dp_size_sweep.py --dry-run    # print the plan + predicted sigma, run nothing
    python dp_size_sweep.py --single-gpu # everything sequentially on the main GPU

Each point is a full NUM_ROUNDS run (~2.4 h at 85 s/round), so the default grid is
roughly a day of GPU. Completed points are skipped on re-invocation.

DUAL-GPU LAYOUT
---------------
One point per card, running concurrently -- NOT one run split across both, which
measured slower than packing 3 clients onto the 5090. The 4060 Ti runs one client
at a time (its 16G cannot hold 3), so it is the slower card and takes only the
first point of each block of CHUNK while the 5090 works through the rest. See
side_indices(): a 14-point grid puts points 1 and 7 on the 4060 Ti.

PER-POINT ARTEFACTS
-------------------
Every fixed output path is suffixed with RUN_TAG (= _size_<size>), which is what
lets two runs share the filesystem and lands the artefacts on their final names:

  fragments_size_<size>/       (~48 GB per point)   server.py loki_fragments_dir
  eval_model_size_<size>.pth   (~1 GB per point)    server.py EVAL_MODEL_PATH
  local_weights_size_<size>/   (transient)          client.py LOCAL_WEIGHTS

The eval model is the PEAK-knn round's, not the final one (server.py gates the
save on a knn improvement), so it is the usable encoder for downstream
clustering. Set LOKI_SAVE_FRAGS = False in FedEMA_run.py to sweep without the
~48 GB/point fragment dumps, then rerun the interesting points with it on.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Spans the trade-off at ~2.3 h/point: 1e2 = real guarantee, model annihilated;
# 1e18 = transition; 1e20 = model mostly recovered; 1e24+ = ~no noise, ceiling.
# Purpose of this run: USABLE per-point artefacts (peak-checkpointed encoder +
# fragments) for offline reconstruction, on CIFAR-100. Earlier CIFAR-10 runs are
# archived in dp_sweep_out/run1_8pt/ and dp_sweep_out/run2_cifar10/.
Trap_size_grid = [128,256,512,1024,2048,4096]#[1e2,1e18, 1e20,1e22, 1e24, 1e26,1e30]
DP_CLIP = 1e6      # public constant, above every observed |update| (~8.3e5) so the
                   # clip stays inactive and does not confound the noise axis
DP_DELTA = 1e-5
OUT = Path("trap_sweep_out")
COLLAPSE_KNN = 0.1   # % knn at/below which a run counts as collapsed (CIFAR-100)

# ── Dual-GPU scheduling ──────────────────────────────────────────────────────
# Splitting the CLIENTS of one run across both cards measured slower than packing
# 3 onto the 5090, so instead each card runs a WHOLE point, concurrently. The 16G
# 4060 Ti takes one client at a time (1.0 gpu/client) or it OOMs on VRAM; that
# makes it slower per point, so it gets 1 point per block of CHUNK while the 5090
# works through the other CHUNK-1. CPUs: 3x5 + 1x5 = 20 of 24 cores.
MAIN_GPU, MAIN_CLIENT_GPUS, MAIN_CPUS = "0", "0.3", "5"   # RTX 5090, 32G
SIDE_GPU, SIDE_CLIENT_GPUS, SIDE_CPUS = "1", "1.0", "5"   # RTX 4060 Ti, 16G
CHUNK = 6

# The knn regex MUST accept nan/inf and MUST capture the round. At small size the
# noise diverges training outright (size=1e2 => sigma=1.9e6 NaNs the model), and the
# server prints "knn_acc=nan%"; a [0-9.]+ pattern would silently skip those and let
# a stale earlier value stand in. (The LOKI m_fired line is intentionally NOT parsed:
# under DP it is measured off the noised update and collapses to a sigma-independent
# constant -- the real leak axis is offline reconstruction over the saved fragments.)
_RE_KNN = re.compile(r"r(\d+) eval: knn_acc=(nan|-?inf|[0-9.]+)%")
_RE_ROUND = re.compile(r"\[ROUND (\d+)\]")


def _num(s: str):
    """None for nan/inf: a diverged run has no metric, and must not be plotted as 0."""
    v = float(s)
    return None if (v != v or v in (float("inf"), float("-inf"))) else v


def parse(text: str) -> dict:
    """PEAK knn and collapse onset -- deliberately NOT the final-round value.

    Under DP noise BYOL representation-collapses: a sudden, sustained fall to ~10%
    knn whose onset comes earlier the more noise there is. The LAST eval is therefore
    post-collapse at every size, so the round-100 number would report ~10% for the
    whole grid and hide the real trade-off (this is the bug the first run's sweep.csv
    walked into). The run's usable utility is its PEAK -- which is exactly the encoder
    server.py now best-checkpoints -- and the collapse onset is a second axis worth
    recording on its own.
    """
    evals = [(int(r), _num(a)) for r, a in _RE_KNN.findall(text)]
    rounds = _RE_ROUND.findall(text)
    healthy = [(r, v) for r, v in evals if v is not None]
    peak_round, peak = max(healthy, key=lambda t: t[1]) if healthy else (None, None)
    # First eval at/below the collapse floor OR nan, occurring AFTER the peak.
    onset = next((r for r, v in evals
                  if peak_round is not None and r > peak_round
                  and (v is None or v <= COLLAPSE_KNN)), None)
    return {
        "rounds_done": int(rounds[-1]) if rounds else 0,
        "peak_knn": peak,
        "peak_round": peak_round,
        "collapse_onset": onset,
        # peak_round == 0 means it never beat random-init features, i.e. never learned
        # (the size=1e2 case); distinct from learning-then-collapsing.
        "never_learned": peak_round == 0 if peak_round is not None else True,
    }


def _num_rounds() -> int:
    return int(os.getenv("NUM_ROUNDS", "100"))


def side_indices(n: int) -> set:
    """Grid positions that run on the second GPU: the FIRST of every full block of
    CHUNK, so the side run starts alongside the block's first main-GPU run.

    A short trailing block is left entirely on the main GPU. Offloading out of one
    would put the slower card on the critical path with too few main-GPU runs left
    to hide it. n=14 => {0, 6}, i.e. points 1 and 7.
    """
    return {i for i in range(0, n, CHUNK) if n - i >= CHUNK}


def _done_row(size: float, tag: str, log: Path):
    """Row for a point that needs no run, else None."""
    if not log.exists():
        return None
    prev = parse(log.read_text())
    if prev["rounds_done"] >= _num_rounds():
        print(f"[sweep] size={tag}: already complete, skipping")
        return {"size": size, **prev, "secs": 0}
    print(f"[sweep] size={tag}: incomplete ({prev['rounds_done']} rounds), rerunning")
    return None


def launch(size: float, gpu: str, client_gpus: str, cpus: str):
    """Start a point as a background process. Returns a handle for finish(), or a
    finished row dict if the point was already complete."""
    tag = str(int(size))
    log = OUT / f"size_{tag}.log"

    done = _done_row(size, tag, log)
    if done is not None:
        return done

    # RUN_TAG suffixes every fixed output path (fragments/, eval_model.pth,
    # local_weights/) in FedEMA_run.py, server.py and client.py. Without it the two
    # concurrent runs would interleave round_NNN.pt into one dir and overwrite each
    # other's peak encoder. It also lands the artefacts on their final names, so
    # there is no post-run rename step.
    run_tag = f"_size_{tag}"
    frags = Path(f"fragments{run_tag}")
    if frags.is_dir():
        shutil.rmtree(frags)   # stale rounds from an interrupted earlier attempt

    env = {
        **os.environ,
        # Load-bearing: FedEMA_run.py gates every clip+noise path on DP_MODE, which
        # defaults to "0". Setting DP_size alone is silently inert -- the sweep would
        # run N identical no-DP baselines and report them as an size axis.
        "DP_MODE": "0",
        "TRAP_SIZE": tag,
        "DP_CLIP": repr(DP_CLIP),
        "DP_DELTA": repr(DP_DELTA),
        "DP_CALIBRATE": "0",
        "RUN_TAG": run_tag,
        "CUDA_VISIBLE_DEVICES": gpu,
        "CLIENT_NUM_GPUS": client_gpus,
        "CLIENT_NUM_CPUS": cpus,
    }
    f = open(log, "w")
    proc = subprocess.Popen([sys.executable, "FedEMA_run.py"], env=env,
                            stdout=f, stderr=subprocess.STDOUT)
    print(f"[sweep] size={tag} gpu={gpu} "
          f"({client_gpus} gpu/client) -> {log}")
    return {"size": size, "tag": tag, "log": log,
            "proc": proc, "f": f, "t0": time.time()}


def finish(h: dict) -> dict:
    """Wait for a launch() handle and parse its log into a row."""
    if "proc" not in h:
        return h                      # already-complete point, nothing to wait on
    h["proc"].wait()
    h["f"].close()
    secs = time.time() - h["t0"]
    res = parse(h["log"].read_text())
    print(f"[sweep] size={h['tag']}: peak_knn={res['peak_knn']}% @r{res['peak_round']} "
          f"collapse@r{res['collapse_onset']} ({secs / 3600:.2f} h)")
    return {"size": h["size"], **res, "secs": round(secs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan, run nothing")
    ap.add_argument("--single-gpu", action="store_true",
                    help="run every point sequentially on the main GPU")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    csv_path = OUT / "sweep.csv"
    n = len(Trap_size_grid)
    side = set() if args.single_gpu else side_indices(n)

    if args.dry_run:
        rows = []
        for i, size in enumerate(Trap_size_grid):
            gpu = SIDE_GPU if i in side else MAIN_GPU
            print(f"[plan] size={int(size)} gpu={gpu} -> fragments_size_{int(size)}/")
            rows.append({"size": size, "rounds_done": 0,
                         "peak_knn": None, "peak_round": None,
                         "collapse_onset": None, "never_learned": None, "secs": 0})
        _summary(rows)
        return

    results: dict = {}
    # One block = CHUNK consecutive points. Its side-GPU point (if any) is launched
    # first and runs concurrently with the block's main-GPU points, then is joined
    # before the next block so only one process ever touches the second card.
    for start in range(0, n, CHUNK):
        block = list(range(start, min(start + CHUNK, n)))
        side_i = block[0] if block[0] in side else None
        side_h = (launch(Trap_size_grid[side_i], SIDE_GPU, SIDE_CLIENT_GPUS, SIDE_CPUS)
                  if side_i is not None else None)
        try:
            for i in block:
                if i == side_i:
                    continue
                results[i] = finish(
                    launch(Trap_size_grid[i], MAIN_GPU, MAIN_CLIENT_GPUS, MAIN_CPUS))
                _checkpoint(csv_path, results)
        except BaseException:
            # Ctrl-C or a main-GPU crash: kill the side run rather than orphan a
            # process holding the second card.
            if side_h is not None and "proc" in side_h:
                side_h["proc"].terminate()
            raise
        finally:
            if side_h is not None:
                results[side_i] = finish(side_h)
        _checkpoint(csv_path, results)

    _summary([results[i] for i in sorted(results)])
    print(f"\nwrote {csv_path}")


def _checkpoint(csv_path: Path, results: dict) -> None:
    """Rewrite the CSV in grid order after every completed point (the grid is ~a day)."""
    rows = [results[i] for i in sorted(results)]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _summary(rows: list) -> None:
    print(f"\n{'size':>9} {'peak_knn':>9} {'@round':>7} {'collapse':>9}  note")
    for r in rows:
        pk = f"{r['peak_knn']:.1f}%" if r.get("peak_knn") is not None else "-"
        pr = str(r.get("peak_round")) if r.get("peak_round") is not None else "-"
        co = f"r{r['collapse_onset']}" if r.get("collapse_onset") is not None else "-"
        note = "never learned (>= random only)" if r.get("never_learned") else ""
        print(f"{int(r['size']):>9} {pk:>9} {pr:>7} {co:>9}  {note}")


if __name__ == "__main__":
    main()

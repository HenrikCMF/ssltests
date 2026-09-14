#!/usr/bin/env python
"""Sweep LOKI_CSF at fixed DP_EPS=1e22 and collect the leak/utility trade-off.

CSF (Eq. 10) is the convolutional scaling factor that fights the fp32 precision
floor in the FedAVG weight delta (CSF^2 scaling). High CSF is what lets the leak
survive at fc_size=40k + small SSL gradients; low CSF collapses extraction. This
grid is orders of magnitude around the current default (1e5):

    1e3, 1e4, 1e5, 1e6, 1e7, 1e8

eps is pinned at 1e22 (the "model recovered, leak intact" point of dp_eps_sweep.py)
so the CSF axis is not confounded with a changing noise scale. DP_CLIP stays the
public constant 1e6 used across the other sweeps: if a high-CSF point inflates
|update| above C, that clip activation IS part of CSF's influence under DP.

USAGE
    python csf_sweep.py              # run the full grid, resumable, both GPUs
    python csf_sweep.py --dry-run    # print the plan, run nothing
    python csf_sweep.py --single-gpu # everything sequentially on the main GPU

Each point is a full NUM_ROUNDS run (~2.4 h at 85 s/round). Completed points are
skipped on re-invocation.

DUAL-GPU LAYOUT
---------------
One point per card, running concurrently -- NOT one run split across both, which
measured slower than packing 3 clients onto the 5090. The 4060 Ti runs one client
at a time (its 16G cannot hold 3), so it is the slower card and takes only the
first point of each block of CHUNK while the 5090 works through the rest. See
side_indices(): a 6-point grid puts point 1 on the 4060 Ti.

PER-POINT ARTEFACTS
-------------------
Every fixed output path is suffixed with RUN_TAG (= _csf_<tag>, tag is 1e3/1e4/…),
which is what lets two runs share the filesystem and lands the artefacts on
their final names:

  fragments_csf_1e5/         (~48 GB per point)   server.py loki_fragments_dir
  eval_model_csf_1e5.pth     (~1 GB per point)    server.py EVAL_MODEL_PATH
  local_weights_csf_1e5/     (transient)          client.py LOCAL_WEIGHTS

The eval model is the PEAK-knn round's, not the final one (server.py gates the
save on a knn improvement), so it is the usable encoder for downstream
clustering / extraction tests.
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

from dp_accounting import noise_std_for_eps

# Orders of magnitude around the FedEMA_run.py default (1e5). 1e3 is expected to
# sit on the fp32 floor (leak collapsed); 1e5 is the known-good leak; 1e8 probes
# whether inflating CSF further helps extraction or just blows |update| into the
# clip.
CSF_GRID = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
DP_EPS = 1e22      # pinned: model-recovered / leak-intact point of the eps sweep
DP_CLIP = 1e6      # public constant, above the CSF=1e5 |update| (~8.3e5)
DP_DELTA = 1e-5
OUT = Path("csf_sweep_out")
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

# The knn regex MUST accept nan/inf and MUST capture the round. A [0-9.]+ pattern
# would silently skip those and let a stale earlier value stand in. (The LOKI
# m_fired line is intentionally NOT parsed: under DP it is measured off the noised
# update and collapses to a sigma-independent constant -- the real leak axis is
# offline reconstruction over the saved fragments.)
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
    post-collapse at every noisy point, so the round-100 number would hide the real
    trade-off. The run's usable utility is its PEAK -- which is exactly the encoder
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
        "never_learned": peak_round == 0 if peak_round is not None else True,
    }


def _num_rounds() -> int:
    return int(os.getenv("NUM_ROUNDS", "100"))


def _csf_tag(csf: float) -> str:
    """1000 -> '1e3', 1e5 -> '1e5'. Compact scientific notation, no '+' or padding."""
    return f"{csf:.0e}".replace("e+0", "e").replace("e+", "e")


def side_indices(n: int) -> set:
    """Grid positions that run on the second GPU: the FIRST of every full block of
    CHUNK, so the side run starts alongside the block's first main-GPU run.

    A short trailing block is left entirely on the main GPU. Offloading out of one
    would put the slower card on the critical path with too few main-GPU runs left
    to hide it. n=6 => {0}, i.e. point 1.
    """
    return {i for i in range(0, n, CHUNK) if n - i >= CHUNK}


def _done_row(csf: float, tag: str, log: Path):
    """Row for a point that needs no run, else None."""
    if not log.exists():
        return None
    prev = parse(log.read_text())
    if prev["rounds_done"] >= _num_rounds():
        print(f"[sweep] csf={tag}: already complete, skipping")
        return {"csf": csf, "eps": DP_EPS, **prev, "secs": 0}
    print(f"[sweep] csf={tag}: incomplete ({prev['rounds_done']} rounds), rerunning")
    return None


def launch(csf: float, gpu: str, client_gpus: str, cpus: str):
    """Start a point as a background process. Returns a handle for finish(), or a
    finished row dict if the point was already complete."""
    tag = _csf_tag(csf)
    log = OUT / f"csf_{tag}.log"

    done = _done_row(csf, tag, log)
    if done is not None:
        return done

    # RUN_TAG suffixes every fixed output path (fragments/, eval_model.pth,
    # local_weights/) in FedEMA_run.py, server.py and client.py. Without it the two
    # concurrent runs would interleave round_NNN.pt into one dir and overwrite each
    # other's peak encoder. It also lands the artefacts on their final names, so
    # there is no post-run rename step.
    run_tag = f"_csf_{tag}"
    frags = Path(f"fragments{run_tag}")
    if frags.is_dir():
        shutil.rmtree(frags)   # stale rounds from an interrupted earlier attempt

    env = {
        **os.environ,
        # Load-bearing: FedEMA_run.py gates every clip+noise path on DP_MODE, which
        # defaults to "0". Setting DP_EPS alone is silently inert.
        "DP_MODE": "1",
        "DP_EPS": repr(DP_EPS),
        "DP_CLIP": repr(DP_CLIP),
        "DP_DELTA": repr(DP_DELTA),
        "DP_CALIBRATE": "0",
        "LOKI_CSF": repr(csf),
        "RUN_TAG": run_tag,
        "CUDA_VISIBLE_DEVICES": gpu,
        "CLIENT_NUM_GPUS": client_gpus,
        "CLIENT_NUM_CPUS": cpus,
    }
    f = open(log, "w")
    proc = subprocess.Popen([sys.executable, "FedEMA_run.py"], env=env,
                            stdout=f, stderr=subprocess.STDOUT)
    sigma = noise_std_for_eps(DP_EPS, DP_DELTA, rounds=_num_rounds(), clip=DP_CLIP)
    print(f"[sweep] csf={tag} eps={DP_EPS:.0e} sigma={sigma:.4g} gpu={gpu} "
          f"({client_gpus} gpu/client) -> {log}")
    return {"csf": csf, "eps": DP_EPS, "tag": tag, "log": log,
            "proc": proc, "f": f, "t0": time.time()}


def finish(h: dict) -> dict:
    """Wait for a launch() handle and parse its log into a row."""
    if "proc" not in h:
        return h                      # already-complete point, nothing to wait on
    h["proc"].wait()
    h["f"].close()
    secs = time.time() - h["t0"]
    res = parse(h["log"].read_text())
    print(f"[sweep] csf={h['tag']}: peak_knn={res['peak_knn']}% @r{res['peak_round']} "
          f"collapse@r{res['collapse_onset']} ({secs / 3600:.2f} h)")
    return {"csf": h["csf"], "eps": h["eps"], **res, "secs": round(secs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan, run nothing")
    ap.add_argument("--single-gpu", action="store_true",
                    help="run every point sequentially on the main GPU")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    csv_path = OUT / "sweep.csv"
    n = len(CSF_GRID)
    side = set() if args.single_gpu else side_indices(n)

    if args.dry_run:
        rows = []
        sigma = noise_std_for_eps(DP_EPS, DP_DELTA, rounds=_num_rounds(), clip=DP_CLIP)
        for i, csf in enumerate(CSF_GRID):
            gpu = SIDE_GPU if i in side else MAIN_GPU
            tag = _csf_tag(csf)
            print(f"[plan] csf={tag} eps={DP_EPS:.0e} sigma={sigma:.4g} gpu={gpu} "
                  f"-> fragments_csf_{tag}/  eval_model_csf_{tag}.pth")
            rows.append({"csf": csf, "eps": DP_EPS, "rounds_done": 0,
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
        side_h = (launch(CSF_GRID[side_i], SIDE_GPU, SIDE_CLIENT_GPUS, SIDE_CPUS)
                  if side_i is not None else None)
        try:
            for i in block:
                if i == side_i:
                    continue
                results[i] = finish(
                    launch(CSF_GRID[i], MAIN_GPU, MAIN_CLIENT_GPUS, MAIN_CPUS))
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
    print(f"\n{'csf':>12} {'eps':>9} {'peak_knn':>9} {'@round':>7} {'collapse':>9}  note")
    for r in rows:
        pk = f"{r['peak_knn']:.1f}%" if r.get("peak_knn") is not None else "-"
        pr = str(r.get("peak_round")) if r.get("peak_round") is not None else "-"
        co = f"r{r['collapse_onset']}" if r.get("collapse_onset") is not None else "-"
        note = "never learned (>= random only)" if r.get("never_learned") else ""
        print(f"{_csf_tag(r['csf']):>12} {r['eps']:>9.0e} {pk:>9} {pr:>7} {co:>9}  {note}")


if __name__ == "__main__":
    main()

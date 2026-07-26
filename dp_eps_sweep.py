#!/usr/bin/env python
"""Sweep DP_EPS and collect the privacy/utility trade-off.

Produces the two axes of the claim "standard DP either destroys the model or does
not stop extraction":

  * utility  -- knn_acc, read from the server's per-round eval
  * leak     -- LEAK_NOISE = sigma / median(m_fired), the in-run reconstruction
                noise proxy the server already computes (server.py ~line 525).
                LEAK_NOISE >= 1 means a fragment is drowned in noise; << 1 means
                it reconstructs cleanly.

WHY THE EPS GRID LOOKS ABSURD
-----------------------------
It is not a typo. LOKI's trap makes the transmitted update norm ~8.3e5 while the
honest part is ~9 (measured: DP_CALIBRATE=1), so the L2 sensitivity a flat clip
must bound is ~1e5x the signal being protected. Client-level local DP therefore
prices the noise that actually kills the leak (sigma ~ 5e-4) at eps ~ 1e21. The
sweep spans the transition wherever it happens to sit; that it sits at 1e17..1e22
IS the finding.

Anchors worth keeping: eps=1e2 is a real guarantee (model annihilated), eps=1e24
is effectively no noise (leak intact).

USAGE
    python dp_eps_sweep.py            # run the full grid, resumable
    python dp_eps_sweep.py --dry-run  # print the plan + predicted sigma, run nothing

Each point is a full NUM_ROUNDS run (~2.4 h at 85 s/round), so the default grid is
roughly a day of GPU. Completed points are skipped on re-invocation.

PER-POINT ARTEFACTS
-------------------
Both are written to FIXED paths on every run, so this driver renames them after each
point or the sweep would end holding only the last one:

  fragments/       -> fragments_eps_<eps>/       (~43 GB per point, ~340 GB total)
  eval_model.pth   -> eval_model_eps_<eps>.pth   (~1 GB per point)

The eval model is the FINAL round's: server.py re-saves that same path every eval
round. Set LOKI_SAVE_FRAGS = False in FedEMA_run.py to sweep on the LEAK_NOISE proxy
alone (eval models are unaffected and still archived), then rerun the two or three
interesting points with it on to get real reconstructions.
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

# Four-point subset of the full 8-point run (archived in dp_sweep_out/run1_8pt/),
# chosen to span the trade-off at manageable runtime (~2.3 h/point): 1e2 = real
# guarantee, model annihilated; 1e18 = transition; 1e20 = model mostly recovered;
# 1e24 = ~no noise, ceiling. This rerun's purpose is USABLE per-point artefacts
# (peak-checkpointed encoder + fragments) for offline reconstruction, which the
# first run could not provide (it saved collapsed final-round models).
EPS_GRID = [1e22, 1e26]
DP_CLIP = 1e6      # public constant, above every observed |update| (~8.3e5) so the
                   # clip stays inactive and does not confound the noise axis
DP_DELTA = 1e-5
OUT = Path("dp_sweep_out")
FRAG_DIR = Path("fragments")
EVAL_MODEL = Path("eval_model.pth")

# The knn regex MUST accept nan/inf and MUST capture the round. At small eps the
# noise diverges training outright (eps=1e2 => sigma=1.9e6 NaNs the model), and the
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
    post-collapse at every eps, so the round-100 number would report ~10% for the
    whole grid and hide the real trade-off (this is the bug the first run's sweep.csv
    walked into). The run's usable utility is its PEAK -- which is exactly the encoder
    server.py now best-checkpoints -- and the collapse onset is a second axis worth
    recording on its own.
    """
    evals = [(int(r), _num(a)) for r, a in _RE_KNN.findall(text)]
    rounds = _RE_ROUND.findall(text)
    healthy = [(r, v) for r, v in evals if v is not None]
    peak_round, peak = max(healthy, key=lambda t: t[1]) if healthy else (None, None)
    # First eval at/below random (10% on CIFAR-10) OR nan, occurring AFTER the peak.
    onset = next((r for r, v in evals
                  if peak_round is not None and r > peak_round
                  and (v is None or v <= 10.5)), None)
    return {
        "rounds_done": int(rounds[-1]) if rounds else 0,
        "peak_knn": peak,
        "peak_round": peak_round,
        "collapse_onset": onset,
        # peak_round == 0 means it never beat random-init features, i.e. never learned
        # (the eps=1e2 case); distinct from learning-then-collapsing.
        "never_learned": peak_round == 0 if peak_round is not None else True,
    }


def run_one(eps: float, dry: bool) -> dict:
    sigma = noise_std_for_eps(eps, DP_DELTA, rounds=int(os.getenv("NUM_ROUNDS", "100")),
                              clip=DP_CLIP)
    tag = f"{eps:.0e}"
    log = OUT / f"eps_{tag}.log"
    row = {"eps": eps, "sigma": sigma}

    if dry:
        row.update(rounds_done=0, peak_knn=None, peak_round=None,
                   collapse_onset=None, never_learned=None, secs=0)
        return row

    if log.exists():
        prev = parse(log.read_text())
        if prev["rounds_done"] >= int(os.getenv("NUM_ROUNDS", "100")):
            print(f"[sweep] eps={tag}: already complete, skipping")
            row.update(prev, secs=0)
            return row
        print(f"[sweep] eps={tag}: incomplete ({prev['rounds_done']} rounds), rerunning")

    env = {
        **os.environ,
        "DP_SCHEME": "eps",
        "DP_EPS": repr(eps),
        "DP_CLIP": repr(DP_CLIP),
        "DP_DELTA": repr(DP_DELTA),
        "DP_CALIBRATE": "0",
    }
    print(f"[sweep] eps={tag} sigma={sigma:.4g} -> {log}")
    t0 = time.time()
    with open(log, "w") as f:
        subprocess.run([sys.executable, "FedEMA_run.py"], env=env,
                       stdout=f, stderr=subprocess.STDOUT, check=False)
    secs = time.time() - t0

    # Archive both per-run artefacts before the next point overwrites them: the
    # fragments dir and the eval model are written to FIXED paths every run
    # (server.py: loki_fragments_dir defaults to "fragments"; eval_model.pth is
    # hardcoded), so without this the sweep would end holding only the last point's.
    if FRAG_DIR.is_dir() and any(FRAG_DIR.iterdir()):
        dest = Path(f"fragments_eps_{tag}")
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(FRAG_DIR), str(dest))
        FRAG_DIR.mkdir(exist_ok=True)
        print(f"[sweep] eps={tag}: fragments -> {dest}")

    # Named to match the existing eval_model_<variant>.pth convention. This is the
    # PEAK-knn model now (server.py gates the save on a knn improvement), so it is
    # the usable encoder for downstream clustering -- not the collapsed final round.
    if EVAL_MODEL.exists():
        dest = Path(f"eval_model_eps_{tag}.pth")
        if dest.exists():
            dest.unlink()
        shutil.move(str(EVAL_MODEL), str(dest))
        print(f"[sweep] eps={tag}: eval model (peak) -> {dest}")

    res = parse(log.read_text())
    row.update(res, secs=round(secs))
    print(f"[sweep] eps={tag}: peak_knn={res['peak_knn']}% @r{res['peak_round']} "
          f"collapse@r{res['collapse_onset']} ({secs / 3600:.2f} h)")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and predicted sigma, run nothing")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    csv_path = OUT / "sweep.csv"
    rows = []
    for eps in EPS_GRID:
        rows.append(run_one(eps, args.dry_run))
        if not args.dry_run:      # checkpoint after each point; the grid is ~a day
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

    print(f"\n{'eps':>9} {'sigma':>10} {'peak_knn':>9} {'@round':>7} {'collapse':>9}  note")
    for r in rows:
        pk = f"{r['peak_knn']:.1f}%" if r.get("peak_knn") is not None else "-"
        pr = str(r.get("peak_round")) if r.get("peak_round") is not None else "-"
        co = f"r{r['collapse_onset']}" if r.get("collapse_onset") is not None else "-"
        note = "never learned (>= random only)" if r.get("never_learned") else ""
        print(f"{r['eps']:>9.0e} {r['sigma']:>10.3g} {pk:>9} {pr:>7} {co:>9}  {note}")
    if not args.dry_run:
        print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()

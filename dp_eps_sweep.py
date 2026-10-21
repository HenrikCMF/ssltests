#!/usr/bin/env python
"""Sweep DP_EPS for the basic FedBYOL + LOKI combo and score every finished point.

The combo is whatever FedEMA_run.py already is, pinned here so a dirty shell
cannot drift it: SSL_METHOD=byol, LOKI_MODE=individual, LOKI_CSF=1e8. 1e5 is
the CSF the earlier ε-sweep (and the attached accuracy/extraction plot) was
measured at. §11 of privacy_findings.md: 1e8 is the same attack with the
Gaussian floor cleared, so extraction should rise at a lower ε while kNN
stays flat.

USAGE
    python dp_eps_sweep.py              # full grid, resumable, both GPUs
    python dp_eps_sweep.py --dry-run    # print the plan and predicted sigma
    python dp_eps_sweep.py --single-gpu # every point sequentially on the 5090

Each point is a full NUM_ROUNDS run (~2.4 h at 85 s/round). When that run's log
shows every round, the same process then, on the 5090:

    regroup_fragments.py   fragments + the peak eval model -> clusters
    infer_fragments.py     clusters -> reconstructions (MIN_VIEWS=8)
    compare_cid0.py        reconstructions -> LPIPS bin montages
    result_evaluation.py   linear probe on the peak eval model

Completed training is not repeated. A point whose post json is missing is
scored again without retraining. Cluster copies are removed after the LPIPS
bins are saved; the raw fragments and the eval model stay.

DUAL-GPU LAYOUT
---------------
One point per card, concurrently. The 4060 Ti runs one client at a time (its
16G cannot hold 3), so it takes only the first point of each full block of
CHUNK while the 5090 works through the rest. Scoring always uses the 5090:
the kNN similarity block does not fit on the 4060 Ti once a point has
millions of fragments.

PER-POINT ARTEFACTS
-------------------
  fragments_eps_<eps>_csf1e8/                 raw leak (~48 GB)     kept
  eval_model_eps_<eps>_csf1e8.pth       peak-knn encoder      kept
  recons_eps_<eps>_csf1e8/              infer_fragments pngs  kept
  dp_sweep_out/lpips_<eps>_csf1e8/      compare_cid0 montages kept
  dp_sweep_out/post_<eps>_csf1e8.json   linear eval + leak    kept
  dp_sweep_out/eps_<eps>_csf1e8.log     training log
  local_weights_eps_<eps>_csf1e8/       transient, deleted by FedEMA_run.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from dp_accounting import noise_std_for_eps

# Basic FedBYOL + LOKI, at the CSF §11.4 says moves the extraction cliff.
# FedEMA_run.py's default is still 1e5 (the old plot). This sweep overrides it.
LOKI_CSF = 1e8
SSL_METHOD = "byol"
LOKI_MODE = "individual"
# Deployed BYOL inverter (infer_fragments' own default is the older noise ckpt).
RECON_CKPT = "reconstruction_out/best_leak_down32_lpips.pt"
MIN_VIEWS = "8"          # same gate as the earlier end-to-end arms
EVAL_DATASET = "cifar10"  # FedEMA_run.py DATASET; compare_cid0.py is hardcoded to it

# 1e2 = real guarantee, model annihilated; 1e18 = transition; 1e20 = model
# mostly recovered; 1e24+ = ~no noise, extraction ceiling.
EPS_GRID = [1e2, 1e18, 1e20, 1e22, 1e24, 1e26, 1e30]
DP_CLIP = 1e6      # public constant, above every observed |update| (~8.3e5) so the
                   # clip stays inactive and does not confound the noise axis
DP_DELTA = 1e-5
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "dp_sweep_out"
COLLAPSE_KNN = 0.1   # % knn at/below which a run counts as collapsed
QA_DIR = ROOT / "cid0_compare"                 # compare_cid0.py writes here, not via env
QA_BAK = ROOT / "cid0_compare_pre_dp_sweep"    # original montages, restored at the end

# ── Dual-GPU scheduling ──────────────────────────────────────────────────────
# Splitting the CLIENTS of one run across both cards measured slower than packing
# 3 onto the 5090, so instead each card runs a WHOLE point, concurrently. The 16G
# 4060 Ti takes one client at a time (1.0 gpu/client) or it OOMs on VRAM; that
# makes it slower per point, so it gets 1 point per block of CHUNK while the 5090
# works through the other CHUNK-1. CPUs: 3x5 + 1x5 = 20 of 24 cores.
MAIN_GPU, MAIN_CLIENT_GPUS, MAIN_CPUS = "0", "0.3", "5"   # RTX 5090, 32G
SIDE_GPU, SIDE_CLIENT_GPUS, SIDE_CPUS = "1", "1.0", "5"   # RTX 4060 Ti, 16G
CHUNK = 6
# One fragment directory is ~48 GB. Refuse to start a point below this so a
# full disk fails in the driver instead of halfway through a round file.
MIN_FREE_GIB = 70

FIELDS = [
    "eps", "sigma", "csf", "rounds_done", "peak_knn", "peak_round",
    "collapse_onset", "never_learned", "secs",
    "lin_eval", "lin_eval_std", "nearest_pct", "weighted_pct",
    "n_bins", "n_recons", "post_secs",
]

# The knn regex MUST accept nan/inf and MUST capture the round. At small eps the
# noise diverges training outright (eps=1e2 => sigma=1.9e6 NaNs the model), and the
# server prints "knn_acc=nan%"; a [0-9.]+ pattern would silently skip those and let
# a stale earlier value stand in.
_RE_KNN = re.compile(r"r(\d+) eval: knn_acc=(nan|-?inf|[0-9.]+)%")
_RE_ROUND = re.compile(r"\[ROUND (\d+)\]")
_RE_WROTE = re.compile(r"wrote (\d+) bins / (\d+) fragments")
_RE_NEAREST = re.compile(
    r"distinct client-0 images leaked\s*:\s*(\d+)/(\d+)\s+\(([0-9.]+)%")
_RE_WEIGHTED = re.compile(
    r"precision-weighted\):\s*([0-9.]+)/(\d+)\s+\(([0-9.]+)%\)")
_RE_BIN = re.compile(
    r"(?m)^\s*(\d+\.\d+-\d+\.\d+)\s+(\d+)\s+\d+\.\d+\s+[0-9.]+")
_RE_LIN = re.compile(r"Linear eval top-1 \(raw feats\)\s*:\s*([0-9.]+)%")
_RE_LIN_STD = re.compile(
    r"Linear eval top-1 \(standardized feats\)\s*:\s*([0-9.]+)%")


def _num(s: str):
    """None for nan/inf: a diverged run has no metric, and must not be plotted as 0."""
    v = float(s)
    return None if (v != v or v in (float("inf"), float("-inf"))) else v


def parse(text: str) -> dict:
    """PEAK knn and collapse onset -- deliberately NOT the final-round value.

    Under DP noise BYOL representation-collapses: a sudden, sustained fall toward
    chance whose onset comes earlier the more noise there is. The LAST eval is
    therefore post-collapse at every noisy point, so the round-100 number would
    hide the real trade-off. The run's usable utility is its PEAK -- which is
    exactly the encoder server.py best-checkpoints. Linear eval of that encoder
    (result_evaluation.py) is the accuracy column of the curve.
    """
    evals = [(int(r), _num(a)) for r, a in _RE_KNN.findall(text)]
    rounds = _RE_ROUND.findall(text)
    healthy = [(r, v) for r, v in evals if v is not None]
    peak_round, peak = max(healthy, key=lambda t: t[1]) if healthy else (None, None)
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


def _eps_tag(eps: float) -> str:
    return f"{eps:.0e}"


def _csf_tag(csf: float) -> str:
    """1e8 -> '1e8'. Python's .0e format writes 1e+08."""
    return f"{csf:.0e}".replace("e+0", "e").replace("e+", "e")


def _paths(eps: float) -> dict:
    tag = f"{_eps_tag(eps)}_csf{_csf_tag(LOKI_CSF)}"
    run_tag = f"_eps_{tag}"
    return {
        "tag": tag,
        "run_tag": run_tag,
        "log": OUT / f"eps_{tag}.log",
        "frags": ROOT / f"fragments{run_tag}",
        "eval": ROOT / f"eval_model{run_tag}.pth",
        "cluster": ROOT / f"fragments_clustered{run_tag}",
        "recons": ROOT / f"recons{run_tag}",
        "lpips": OUT / f"lpips_{tag}",
        "post": OUT / f"post_{tag}.json",
        "regroup_log": OUT / f"regroup_{tag}.log",
        "infer_log": OUT / f"infer_{tag}.log",
        "compare_log": OUT / f"compare_{tag}.log",
        "eval_log": OUT / f"lineval_{tag}.log",
    }


def _blank_post() -> dict:
    return {
        "lin_eval": None, "lin_eval_std": None,
        "nearest_pct": None, "weighted_pct": None,
        "n_bins": None, "n_recons": None, "post_secs": 0,
    }


def side_indices(n: int) -> set:
    """Grid positions that run on the second GPU: the FIRST of every full block of
    CHUNK, so the side run starts alongside the block's first main-GPU run.

    A short trailing block is left entirely on the main GPU. Offloading out of one
    would put the slower card on the critical path with too few main-GPU runs left
    to hide it. n=7, CHUNK=6 => {0}, i.e. the 1e2 point.
    """
    return {i for i in range(0, n, CHUNK) if n - i >= CHUNK}


def _free_gib() -> float:
    return shutil.disk_usage(ROOT).free / (1024 ** 3)


def _run(script: str, env: dict, log: Path, extra: list | None = None) -> None:
    """Run one post step on the 5090. The 4060 Ti is reserved for its training point."""
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", script, *(extra or [])]
    penv = {**env, "CUDA_VISIBLE_DEVICES": MAIN_GPU, "PYTHONUNBUFFERED": "1"}
    print(f"[sweep] {' '.join(cmd)} -> {log.relative_to(ROOT)}", flush=True)
    with open(log, "w") as f:
        proc = subprocess.run(cmd, env=penv, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
    if proc.returncode != 0:
        tail = log.read_text(errors="replace")[-2500:]
        raise RuntimeError(f"{script} exited {proc.returncode}\n{tail}")


def _regroup_done(p: dict) -> bool:
    if not p["regroup_log"].exists():
        return False
    text = p["regroup_log"].read_text(errors="replace")
    if "nothing to cluster" in text:
        return True
    return "wrote " in text and p["cluster"].is_dir()


def _infer_done(p: dict) -> bool:
    return p["infer_log"].exists() and "done." in p["infer_log"].read_text(errors="replace")


def _lineval_done(p: dict) -> bool:
    return p["eval_log"].exists() and _RE_LIN.search(p["eval_log"].read_text(errors="replace"))


def _park_qa() -> None:
    """compare_cid0.py always writes cid0_compare/. Keep the pre-sweep montages."""
    if QA_DIR.exists() and not QA_BAK.exists():
        shutil.move(str(QA_DIR), str(QA_BAK))
        print(f"[sweep] parked {QA_DIR.name} -> {QA_BAK.name}", flush=True)
    elif QA_DIR.exists():
        shutil.rmtree(QA_DIR)


def _save_lpips(p: dict) -> None:
    if p["lpips"].exists():
        shutil.rmtree(p["lpips"])
    shutil.copytree(QA_DIR, p["lpips"])
    n = len(list(p["lpips"].glob("match_lpbin_*.png")))
    print(f"[sweep] eps={p['tag']}: {n} LPIPS bins -> {p['lpips'].relative_to(ROOT)}", flush=True)


def _parse_compare(text: str) -> dict:
    near = _RE_NEAREST.search(text)
    weight = _RE_WEIGHTED.search(text)
    if not near or not weight:
        raise RuntimeError("compare_cid0 log has no leak summary")
    return {
        "nearest_pct": float(near.group(3)),
        "weighted_pct": float(weight.group(3)),
        "lpips_bins": {m.group(1): int(m.group(2)) for m in _RE_BIN.finditer(text)},
    }


def _parse_lineval(text: str) -> dict:
    raw, std = _RE_LIN.search(text), _RE_LIN_STD.search(text)
    if not raw or not std:
        raise RuntimeError("result_evaluation log has no top-1 line")
    return {"lin_eval": float(raw.group(1)), "lin_eval_std": float(std.group(1))}


def _write_post(p: dict, metrics: dict) -> None:
    payload = {
        **metrics,
        "eps_tag": p["tag"],
        "csf": LOKI_CSF,
        "fragments": str(p["frags"].relative_to(ROOT)),
        "eval_model": str(p["eval"].relative_to(ROOT)),
        "recons": str(p["recons"].relative_to(ROOT)),
        "lpips": str(p["lpips"].relative_to(ROOT)),
    }
    p["post"].write_text(json.dumps(payload, indent=2) + "\n")


def _load_post(path: Path) -> dict:
    data = json.loads(path.read_text())
    return {k: data.get(k) for k in _blank_post()}


def _drop_clusters(p: dict) -> None:
    """The cluster tree is a second copy of the raw fragments (~48 GB). The raw
    directory, the reconstructions and the LPIPS bins are the kept artefacts."""
    if p["cluster"].exists():
        print(f"[sweep] eps={p['tag']}: removing {p['cluster'].name}", flush=True)
        shutil.rmtree(p["cluster"])


def postprocess(eps: float) -> dict:
    """Score one finished point. Idempotent: a step whose log already completed is skipped."""
    p = _paths(eps)
    t0 = time.time()
    metrics = _blank_post()
    bins: dict = {}

    if not p["eval"].exists():
        print(f"[sweep] eps={p['tag']}: no eval model (training never beat the init); extraction 0",
              flush=True)
        metrics.update(nearest_pct=0.0, weighted_pct=0.0, n_bins=0, n_recons=0,
                       post_secs=round(time.time() - t0))
        _write_post(p, metrics)
        return metrics

    if not _regroup_done(p):
        if p["cluster"].exists():
            shutil.rmtree(p["cluster"])
        _run("regroup_fragments.py", {
            **os.environ,
            "FRAG_DIR": str(p["frags"]),
            "CKPT": str(p["eval"]),
            "OUT_DIR": str(p["cluster"]),
            "SSL_METHOD": SSL_METHOD,
        }, p["regroup_log"])
    regroup_text = p["regroup_log"].read_text(errors="replace")
    wrote = _RE_WROTE.search(regroup_text)
    metrics["n_bins"] = int(wrote.group(1)) if wrote else 0

    if "nothing to cluster" in regroup_text:
        metrics.update(n_recons=0, nearest_pct=0.0, weighted_pct=0.0)
        p["compare_log"].write_text("nothing to cluster; extraction 0\n")
    else:
        if not _infer_done(p):
            if p["recons"].exists():
                shutil.rmtree(p["recons"])
            _run("infer_fragments.py", {
                **os.environ,
                "FRAG_DIR": str(p["cluster"]),
                "MIN_VIEWS": MIN_VIEWS,
                "RECON_SUBDIR": str(p["recons"]),
                "CKPT": RECON_CKPT,
                "SSL_METHOD": SSL_METHOD,
            }, p["infer_log"])
        n_png = len(list(p["recons"].glob("*.png"))) if p["recons"].is_dir() else 0
        metrics["n_recons"] = n_png
        if n_png == 0:
            metrics.update(nearest_pct=0.0, weighted_pct=0.0)
            p["compare_log"].write_text("no reconstructions; extraction 0\n")
        else:
            if not (p["compare_log"].exists() and p["lpips"].is_dir() and _RE_NEAREST.search(
                    p["compare_log"].read_text(errors="replace"))):
                _park_qa()
                _run("compare_cid0.py", {
                    **os.environ,
                    "RECON_DIR": str(p["recons"]),
                    "CLUSTER_DIR": str(p["cluster"]),
                }, p["compare_log"])
                _save_lpips(p)
            parsed = _parse_compare(p["compare_log"].read_text(errors="replace"))
            bins = parsed.pop("lpips_bins")
            metrics.update(parsed)

    if not _lineval_done(p):
        _run("result_evaluation.py", dict(os.environ), p["eval_log"],
             ["--ckpt", str(p["eval"]), "--dataset", EVAL_DATASET])
    metrics.update(_parse_lineval(p["eval_log"].read_text(errors="replace")))
    metrics["post_secs"] = round(time.time() - t0)
    _write_post(p, {**metrics, "lpips_bins": bins})
    _drop_clusters(p)
    print(f"[sweep] eps={p['tag']}: lin_eval={metrics['lin_eval']}% "
          f"nearest={metrics['nearest_pct']}% weighted={metrics['weighted_pct']}% "
          f"({metrics['post_secs'] / 3600:.2f} h scoring)", flush=True)
    return metrics


def launch(eps: float, gpu: str, client_gpus: str, cpus: str):
    """Start a point as a background process, or a handle if training already finished."""
    sigma = noise_std_for_eps(eps, DP_DELTA, rounds=_num_rounds(), clip=DP_CLIP)
    p = _paths(eps)
    base = {"eps": eps, "sigma": sigma, "tag": p["tag"], "paths": p}

    if p["log"].exists():
        prev = parse(p["log"].read_text(errors="replace"))
        if prev["rounds_done"] >= _num_rounds():
            print(f"[sweep] eps={p['tag']}: training already complete", flush=True)
            return {**base, "res": prev}
        print(f"[sweep] eps={p['tag']}: incomplete ({prev['rounds_done']} rounds), rerunning",
              flush=True)

    if p["frags"].is_dir():
        shutil.rmtree(p["frags"])   # stale rounds from an interrupted earlier attempt
    free = _free_gib()
    if free < MIN_FREE_GIB:
        raise RuntimeError(f"only {free:.0f} GiB free; a point needs ~48 GiB of fragments")

    # RUN_TAG suffixes fragments/, eval_model.pth and local_weights/. Without it
    # the two concurrent runs would interleave round_NNN.pt into one directory
    # and overwrite each other's peak encoder.
    env = {
        **os.environ,
        # FedEMA_run.py gates every clip+noise path on DP_MODE, which defaults
        # to "0". Setting DP_EPS alone is silently inert.
        "DP_MODE": "1",
        "DP_EPS": repr(eps),
        "DP_CLIP": repr(DP_CLIP),
        "DP_DELTA": repr(DP_DELTA),
        "DP_CALIBRATE": "0",
        "LOKI_CSF": repr(LOKI_CSF),
        "SSL_METHOD": SSL_METHOD,
        "LOKI_MODE": LOKI_MODE,
        "RUN_TAG": p["run_tag"],
        "CUDA_VISIBLE_DEVICES": gpu,
        "CLIENT_NUM_GPUS": client_gpus,
        "CLIENT_NUM_CPUS": cpus,
        "PYTHONUNBUFFERED": "1",
    }
    f = open(p["log"], "w")
    # Own process group so an interrupt can take the Flower/Ray children with it
    # instead of leaving a client actor on the card.
    proc = subprocess.Popen(
        [sys.executable, "-u", "FedEMA_run.py"],
        env=env, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT),
        start_new_session=True)
    print(f"[sweep] eps={p['tag']} csf={LOKI_CSF:.0e} sigma={sigma:.4g} gpu={gpu} "
          f"({client_gpus} gpu/client) -> {p['log'].relative_to(ROOT)}", flush=True)
    return {**base, "proc": proc, "f": f, "t0": time.time()}


def finish(h: dict) -> dict:
    """Wait for training if needed, then score the point when the run is complete."""
    p = h["paths"]
    rc = 0
    if "proc" in h:
        rc = h["proc"].wait()
        h["f"].close()
        secs = time.time() - h["t0"]
        res = parse(p["log"].read_text(errors="replace"))
        print(f"[sweep] eps={h['tag']}: peak_knn={res['peak_knn']}% @r{res['peak_round']} "
              f"collapse@r{res['collapse_onset']} rc={rc} ({secs / 3600:.2f} h)", flush=True)
    else:
        secs = 0
        res = h["res"]

    row = {"eps": h["eps"], "sigma": h["sigma"], "csf": LOKI_CSF, **res,
           "secs": round(secs), **_blank_post()}
    if res["rounds_done"] >= _num_rounds() and rc == 0:
        if p["post"].exists():
            print(f"[sweep] eps={h['tag']}: scoring already done", flush=True)
            row.update(_load_post(p["post"]))
            _drop_clusters(p)
        else:
            row.update(postprocess(h["eps"]))
    else:
        print(f"[sweep] eps={h['tag']}: training incomplete "
              f"(rc={rc}, rounds={res['rounds_done']}), not scoring", flush=True)
    return row


def _restore_qa() -> None:
    if not QA_BAK.exists():
        return
    if QA_DIR.exists():
        shutil.rmtree(QA_DIR)
    shutil.move(str(QA_BAK), str(QA_DIR))
    print(f"[sweep] restored {QA_DIR.name}", flush=True)


def main() -> None:
    os.chdir(ROOT)
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and predicted sigma, run nothing")
    ap.add_argument("--single-gpu", action="store_true",
                    help="run every point sequentially on the main GPU")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    csv_path = OUT / "sweep.csv"
    n = len(EPS_GRID)
    side = set() if args.single_gpu else side_indices(n)

    if args.dry_run:
        rows = []
        for i, eps in enumerate(EPS_GRID):
            gpu = SIDE_GPU if i in side else MAIN_GPU
            sigma = noise_std_for_eps(eps, DP_DELTA, rounds=_num_rounds(), clip=DP_CLIP)
            p = _paths(eps)
            print(f"[plan] eps={p['tag']} csf={LOKI_CSF:.0e} sigma={sigma:.4g} gpu={gpu} "
                  f"-> {p['frags'].name}/  {p['eval'].name}  {p['lpips'].relative_to(ROOT)}/")
            rows.append({"eps": eps, "sigma": sigma, "csf": LOKI_CSF, "rounds_done": 0,
                         "peak_knn": None, "peak_round": None, "collapse_onset": None,
                         "never_learned": None, "secs": 0, **_blank_post()})
        _summary(rows)
        return

    ray = Path(sys.executable).with_name("ray")
    if ray.exists():
        subprocess.run([str(ray), "stop", "--force"], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    results: dict = {}
    # One block = CHUNK consecutive points. Its side-GPU point (if any) is launched
    # first and runs concurrently with the block's main-GPU points, then is joined
    # before the next block so only one process ever touches the second card.
    try:
        for start in range(0, n, CHUNK):
            block = list(range(start, min(start + CHUNK, n)))
            side_i = block[0] if block[0] in side else None
            side_h = (launch(EPS_GRID[side_i], SIDE_GPU, SIDE_CLIENT_GPUS, SIDE_CPUS)
                      if side_i is not None else None)
            try:
                for i in block:
                    if i == side_i:
                        continue
                    results[i] = finish(
                        launch(EPS_GRID[i], MAIN_GPU, MAIN_CLIENT_GPUS, MAIN_CPUS))
                    _checkpoint(csv_path, results)
            except KeyboardInterrupt:
                proc = side_h.get("proc") if side_h is not None else None
                if proc is not None and proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                raise
            finally:
                if side_h is not None and side_i not in results:
                    results[side_i] = finish(side_h)
                    _checkpoint(csv_path, results)
    finally:
        _restore_qa()

    _summary([results[i] for i in sorted(results)])
    print(f"\nwrote {csv_path}")


def _checkpoint(csv_path: Path, results: dict) -> None:
    """Rewrite the CSV in grid order after every completed point (the grid is ~a day)."""
    rows = [{k: results[i].get(k) for k in FIELDS} for i in sorted(results)]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def _summary(rows: list) -> None:
    print(f"\n{'eps':>9} {'sigma':>10} {'lin_eval':>9} {'knn':>8} {'nearest':>8} {'weighted':>9}")
    for r in rows:
        def pct(key):
            v = r.get(key)
            return f"{v:.1f}%" if isinstance(v, (int, float)) else "-"
        print(f"{r['eps']:>9.0e} {r['sigma']:>10.3g} {pct('lin_eval'):>9} "
              f"{pct('peak_knn'):>8} {pct('nearest_pct'):>8} {pct('weighted_pct'):>9}")


if __name__ == "__main__":
    main()

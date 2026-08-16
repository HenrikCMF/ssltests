#!/usr/bin/env bash
# run_arm.sh <arm_name> <frag_dir> [min_views]
#
# One arm of the round-cutoff / fixed-criterion experiments through the UNMODIFIED
# extraction pipeline: regroup (embed + DBSCAN re-identification) -> infer (the deployed
# inverter on each cluster). Scoring is done separately by score_recons.py so the CIFAR
# pool and the LPIPS network are built once for all arms.
#
# SIGN_CANON=0 (control) on every arm: the eps runs predate per-round cutoff persistence,
# so canonicalisation would fall back to fitting z0 per round and SUCCEED on clean rounds
# while FAILING (R^2 < 0.5) on noisy ones -- an arm-dependent behaviour that would
# confound exactly the comparison being made. Canon is worth ~+1 point uniformly (Table 5).
set -euo pipefail
ARM=$1; FRAGS=$2; MV=${3:-8}
CLU="clusters_${ARM}"
REC="recons_${ARM}"

echo "########## REGROUP ${ARM} (${FRAGS}) K_NN=${K_NN:-99} ##########"
SIGN_CANON=0 FRAG_DIR="${FRAGS}" OUT_DIR="${CLU}" python -u regroup_fragments.py

echo "########## INFER ${ARM} (MIN_VIEWS=${MV}) ##########"
FRAG_DIR="${CLU}" MIN_VIEWS="${MV}" RECON_SUBDIR="${REC}" python -u infer_fragments.py \
  | grep -Ev "^  [0-9]+/[0-9]+$"

echo "########## DONE ${ARM} ##########"

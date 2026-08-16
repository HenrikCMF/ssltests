#!/usr/bin/env bash
# queue_arms.sh <gpu> <wait_for_pid> <arm:fragdir> [arm:fragdir ...]
#
# Wait for an in-flight regroup on this GPU to finish, then run the queued arms one at a
# time. One arm per GPU at a time is a HOST-RAM limit, not a VRAM one: collect_usable holds
# ~3M (path, index) tuples plus the fragment cache, which is ~15-21 GB RSS per process
# against 60 GB total, so a third concurrent arm risks the OOM killer taking all of them.
set -uo pipefail
GPU=$1; shift
WAIT_PID=$1; shift

if [ "${WAIT_PID}" != "none" ]; then
  echo "[gpu${GPU}] waiting for pid ${WAIT_PID}..."
  while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 30; done
  echo "[gpu${GPU}] pid ${WAIT_PID} finished"
fi

for spec in "$@"; do
  arm="${spec%%:*}"; frags="${spec#*:}"
  echo "[gpu${GPU}] === ${arm} (${frags}) start $(date +%H:%M:%S) ==="
  CUDA_VISIBLE_DEVICES="${GPU}" ./run_arm.sh "${arm}" "${frags}" 8 > "arm_logs/${arm}.log" 2>&1
  rc=$?
  echo "[gpu${GPU}] === ${arm} exit=${rc} $(date +%H:%M:%S) ==="
  grep -E "usable fragments|clusters \||wrote |full bins" "arm_logs/${arm}.log" || true
  # The cluster dir is ~10-20 GB of ~12 KB files and is only needed until its
  # reconstructions exist; drop it so six arms do not sit on disk at once.
  if [ -d "reconstruction_out/recons_${arm}" ] && \
     [ "$(ls -A "reconstruction_out/recons_${arm}" 2>/dev/null | wc -l)" -gt 0 ]; then
    rm -rf "clusters_${arm}"
    echo "[gpu${GPU}] removed clusters_${arm}"
  fi
done
echo "[gpu${GPU}] QUEUE DONE"

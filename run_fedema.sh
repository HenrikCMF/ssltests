#!/usr/bin/env bash

#SBATCH --job-name=fedema
#SBATCH --output=fedema.out
#SBATCH --error=fedema.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G

# 4 GPUs: with client_resources={"num_gpus": 1} in FedEMA_run.py, Flower/Ray runs
# one client per card, so 4 of the 5 clients train in parallel (the 5th queues) and
# none has to share a card with the server-driver process (KNN eval + LOKI). At
# gres=:1 that sharing was the OOM -- one client (~30GB) + the driver (~14GB) filled
# the single 48GB card. cpus-per-task=32 == 4 clients x num_cpus:8.
#
# Per-card the L40s (48GB) is still the best option: the A100 node (aicentre-a100,
# nv-ai-04) IS reachable -- it mounts /home and reads torch.sif fine; the real gate
# was QOS, needs --qos=unprivileged, NOT a storage domain -- but those cards are the
# A100-SXM4-40GB (40GB < 48GB here), so they'd OOM sooner on a single-GPU run.

sif="torch.sif"

# Stream stdout/stderr live instead of block-buffering them: without this, print()
# output sits in an 8KB buffer and fedema.out stays empty until the process exits,
# making a long run impossible to monitor.
export SINGULARITYENV_PYTHONUNBUFFERED=1

# Cap BLAS/OMP threads per process. This node has many cores; uncapped, each client
# actor spawns ~1 math thread per core -> thousands of threads thrashing, which is
# what stalls/deadlocks data loading. Match the cpus we request per client.
export SINGULARITYENV_OMP_NUM_THREADS=8
export SINGULARITYENV_MKL_NUM_THREADS=8
export SINGULARITYENV_OPENBLAS_NUM_THREADS=8

# Let the CUDA allocator grow/shrink segments instead of pre-reserving fixed blocks.
# The last OOM died needing 2.29 GiB while 2.26 GiB sat reserved-but-unallocated
# (fragmented) in-process -- expandable_segments reclaims exactly that. This matters
# here because the server's KNN-eval + LOKI reconstruction runs on the same card as
# the client actor (gres=l40s:1), so the 48GB fills and fragmentation is the margin.
export SINGULARITYENV_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --nv binds the host NVIDIA driver into the container (PyTorch ships its own CUDA
# runtime via the bundled nvidia-*-cu12 wheels). Run from the project directory so
# Singularity bind-mounts $PWD: ./data is read, fragments/ and local_weights/ are
# written back to the host.
singularity run --nv "$sif"

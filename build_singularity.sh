#!/usr/bin/env bash

#SBATCH --job-name=build_torch
#SBATCH --output=build_torch.out
#SBATCH --error=build_torch.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

export SINGULARITY_TMPDIR=$HOME/.singularity/tmp
export SINGULARITY_CACHEDIR=$HOME/.singularity/cache
mkdir -p $SINGULARITY_CACHEDIR $SINGULARITY_TMPDIR

# The path to the definition file
input_def="torch.def"

# The resulting container image
output_sif="torch.sif"

# Build fresh (remove a stale image so the build does not prompt/abort).
rm -f "$output_sif"
singularity build --fakeroot "$output_sif" "$input_def"
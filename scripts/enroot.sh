#!/bin/bash
#SBATCH --partition=batch
#SBATCH --account=coreai_resiliency_osiris
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --job-name=enroot

##SBATCH --exclusive

set -Eeoux pipefail

function detect_gpus_per_node {
    if command -v sinfo &>/dev/null; then
        sinfo -eO "Gres" \
            | grep -v "^GRES" \
            | grep "gpu:" \
            | sed 's/([^)]*)//g' \
            | awk -F: '{print $NF}' \
            | grep -E '^[0-9]+$' \
            | sort -un \
            | head -1
    fi
}

if [ -n "${SLURM_JOB_ID:-}" ] ; then
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
else
    SCRIPT_PATH=$(realpath "$0")
fi
SCRIPT_DIR=$(dirname ${SCRIPT_PATH})
MEGATRON_ROOT=${SCRIPT_DIR}/../../..
DOCKER_IMAGE_BASE="nvcr.io/nvidia/pytorch:26.04-py3"
# First 10 hex chars of sha256(prepare.sh) — must match _prepare_sh_hash() in
# ci/scripts/run_job.py so both paths agree on the filename.
PREPARE_SH_HASH=$(sha256sum "${SCRIPT_DIR}/prepare.sh" | cut -c1-10)
ENROOT_IMAGE_BASE=${ENROOT_IMAGE_BASE:-${MEGATRON_ROOT}/../pytorch2604.sqsh}
ENROOT_IMAGE_OUT=${ENROOT_IMAGE_OUT:-${MEGATRON_ROOT}/../native.${PREPARE_SH_HASH}.sqsh}
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
GPUS_PER_NODE=${GPUS_PER_NODE:-$(detect_gpus_per_node)}

# Common srun flags. Defaults match run_job.py; override via env vars in CI.
# The SBATCH headers above only apply when submitted via sbatch — when this script
# is run directly (CI, devbox), srun uses the cluster's user defaults, which on lyris
# requires --account to be specified explicitly.
SLURM_ACCOUNT=${SLURM_ACCOUNT:-coreai_resiliency_osiris}
SLURM_TIME=${SLURM_TIME:-00:60:00}
SLURM_PARTITION=${SLURM_PARTITION:-interactive}
slurm_common_args=(
    "--account=${SLURM_ACCOUNT}"
    "--time=${SLURM_TIME}"
    "--mem=0"
)
if [[ -n "${SLURM_PARTITION:-}" ]]; then
    slurm_common_args+=("--partition=${SLURM_PARTITION}")
fi

# GPU request for the build/run srun. The image build itself doesn't need GPUs,
# but some schedulers require the request; prepare.sh tolerates their absence.
gpu_args=""
if [[ -n "${GPUS_PER_NODE:-}" && "${GPUS_PER_NODE}" -gt 0 ]]; then
    gpu_args="--gpus-per-node=${GPUS_PER_NODE}"
fi

function build_image_base {
	if [[ -e ${ENROOT_IMAGE_BASE} ]]; then
		echo "using ENROOT_IMAGE_BASE=${ENROOT_IMAGE_BASE}"
	else
		rm -f "${ENROOT_IMAGE_BASE}.tmp"
		srun -l "${slurm_common_args[@]}" enroot import --output ${ENROOT_IMAGE_BASE}.tmp docker://${DOCKER_IMAGE_BASE}
		mv -f "${ENROOT_IMAGE_BASE}.tmp" "${ENROOT_IMAGE_BASE}"
	fi
	$*
}


function build {
	build_image_base
	rm -f "${ENROOT_IMAGE_OUT}.tmp"
	srun -l \
		"${slurm_common_args[@]}" \
		${gpu_args} \
		--container-mounts "${HOME}:${HOME},/lustre:/lustre" \
		--container-image "${ENROOT_IMAGE_BASE}" \
		--container-save "${ENROOT_IMAGE_OUT}.tmp" \
		--container-writable \
		bash -c "bash ${SCRIPT_DIR}/prepare.sh ubuntu2404 megatron test"
	mv -f "${ENROOT_IMAGE_OUT}.tmp" "${ENROOT_IMAGE_OUT}"
	$*
}

function run {
	srun -l \
		"${slurm_common_args[@]}" \
		${gpu_args} \
		--container-mounts "${HOME}:${HOME},/lustre:/lustre" \
		--container-image "${ENROOT_IMAGE_OUT}" \
		bash ${SCRIPT_DIR}/run_1.sh 
	$*
}

if [ "$#" -eq 0 ]; then
    build
else
    $*
fi
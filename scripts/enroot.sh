#!/bin/bash
#SBATCH --partition=batch
#SBATCH --account=coreai_resiliency_osiris
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --time=00:20:00
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
DOCKER_IMAGE_BASE="nvcr.io/nvidia/pytorch:26.01-py3"
# First 10 hex chars of sha256(prepare.sh) — must match _prepare_sh_hash() in
# megatron/native/ci/scripts/run_job.py so both paths agree on the filename.
PREPARE_SH_HASH=$(sha256sum "${SCRIPT_DIR}/prepare.sh" | cut -c1-10)
ENROOT_IMAGE_BASE=${ENROOT_IMAGE_BASE:-${MEGATRON_ROOT}/../base.sqsh}
ENROOT_IMAGE_OUT=${ENROOT_IMAGE_OUT:-${MEGATRON_ROOT}/../native.${PREPARE_SH_HASH}.sqsh}
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
GPUS_PER_NODE=${GPUS_PER_NODE:-$(detect_gpus_per_node)}

# Common srun flags. Defaults match run_job.py; override via env vars in CI.
# The SBATCH headers above only apply when submitted via sbatch — when this script
# is run directly (CI, devbox), srun uses the cluster's user defaults, which on lyris
# requires --account to be specified explicitly.
SLURM_ACCOUNT=${SLURM_ACCOUNT:-coreai_resiliency_osiris}
SLURM_TIME=${SLURM_TIME:-00:20:00}
slurm_common_args=(
    "--account=${SLURM_ACCOUNT}"
    "--time=${SLURM_TIME}"
    "--mem=0"
)
if [[ -n "${SLURM_PARTITION:-}" ]]; then
    slurm_common_args+=("--partition=${SLURM_PARTITION}")
fi

function build_image_base {
	if [[ -e ${ENROOT_IMAGE_BASE} ]]; then
		echo "using ENROOT_IMAGE_BASE=${ENROOT_IMAGE_BASE}"
	else
		base_image_file=${MEGATRON_ROOT}/../$(basename ${ENROOT_IMAGE_BASE})
		srun -l "${slurm_common_args[@]}" enroot import --output ${base_image_file} docker://${DOCKER_IMAGE_BASE}
		ln -s ${base_image_file} "${ENROOT_IMAGE_BASE}"
	fi
	$*
}


function build {
	build_image_base
	# Honor SLURM_GPU_FLAGS_ENABLED=0 the same way run_job.py does — some clusters
	# (e.g. lyris) reject --gpus-per-node with "Invalid generic resource (gres)".
	# Image build doesn't actually need GPUs; prepare.sh handles nvcc/nvidia-smi absence.
	local gpu_args=""
	local gpu_flags_enabled="${SLURM_GPU_FLAGS_ENABLED:-1}"
	if [[ "${gpu_flags_enabled}" != "0" && "${gpu_flags_enabled}" != "false" && "${gpu_flags_enabled}" != "False" \
	      && -n "${GPUS_PER_NODE:-}" && "${GPUS_PER_NODE}" -gt 0 ]]; then
		gpu_args="--gpus-per-node=${GPUS_PER_NODE}"
	fi
	srun -l \
		"${slurm_common_args[@]}" \
		${gpu_args} \
		--container-mounts "${HOME}:${HOME},${HOME}:/root,/lustre:/lustre" \
		--container-image "${ENROOT_IMAGE_BASE}" \
		--container-save "${ENROOT_IMAGE_OUT}" \
		--container-writable \
		bash -c "bash ${SCRIPT_DIR}/prepare.sh ubuntu2404 megatron test"
	$*
}

function run {
	srun -l \
		"${slurm_common_args[@]}" \
		${gpu_args} \
		--container-mounts "${HOME}:${HOME},${HOME}:/root,/lustre:/lustre" \
		--container-image "${ENROOT_IMAGE_OUT}" \
		bash ${SCRIPT_DIR}/run_1.sh 
	$*
}

if [ "$#" -eq 0 ]; then
    build
else
    $*
fi
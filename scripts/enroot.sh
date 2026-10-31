#!/bin/bash
#SBATCH --partition=batch
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
MEGATRON_ROOT=$(bash ${SCRIPT_DIR}/prepare.sh megatron_root)
source <(grep -E '^PYTORCH_IMAGE=' "${SCRIPT_DIR}/prepare.sh")
DOCKER_IMAGE_BASE=${PYTORCH_IMAGE}
PYTORCH_RELEASE=${PYTORCH_IMAGE##*:}
PYTORCH_RELEASE=${PYTORCH_RELEASE%%-*}
# First 10 hex chars of sha256(prepare.sh) — must match _prepare_sh_hash() in
# ci/scripts/slurm_job.py so both paths agree on the filename.
PREPARE_SH_HASH=$(sha256sum "${SCRIPT_DIR}/prepare.sh" | cut -c1-10)
ENROOT_IMAGE_BASE=${ENROOT_IMAGE_BASE:-${MEGATRON_ROOT}/../pytorch${PYTORCH_RELEASE//./}.sqsh}
ENROOT_IMAGE_OUT=${ENROOT_IMAGE_OUT:-${MEGATRON_ROOT}/../native.${PREPARE_SH_HASH}.sqsh}
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
GPUS_PER_NODE=${GPUS_PER_NODE:-$(detect_gpus_per_node)}

# Common srun flags. Defaults match slurm_job.py; override via env vars in CI.
# Direct runs bypass the SBATCH headers above, so srun takes the account explicitly.
SLURM_ACCOUNT=${SLURM_ACCOUNT:?set SLURM_ACCOUNT to the Slurm account to charge}
SLURM_TIME=${SLURM_TIME:-00:60:00}
slurm_common_args=(
    "--account=${SLURM_ACCOUNT}"
    "--time=${SLURM_TIME}"
    "--mem=0"
)

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

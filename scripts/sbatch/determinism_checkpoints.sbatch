#!/bin/bash
# Generate an A/A checkpoint pair for the pico Mamba-hybrid MoE model on the
# PyTorch 26.07 stack using only a Megatron-LM checkout. Submit this file twice
# with distinct RUN_TAG values:
#
#   export DET_WORK_DIR=/lustre/<project>/<user>/determinism
#   export MEGATRON_ROOT=/lustre/<project>/<user>/Megatron-LM
#   export ENROOT_IMAGE=/lustre/<project>/<user>/pytorch2607-megatron.sqsh
#   sbatch --segment=8 --export=ALL,RUN_TAG=pico-cg-a scripts/sbatch/determinism_checkpoints.sbatch
#   sbatch --segment=8 --export=ALL,RUN_TAG=pico-cg-b scripts/sbatch/determinism_checkpoints.sbatch
#
# Add --nodes=2 to both submissions for the two-node case. The decomposed
# weighted-squared-ReLU path is exact at both one and two nodes with Mamba,
# attention, and MoE-router CUDA graphs enabled. Enabling the fused path
# reproduces the observed run-to-run drift.
#
# For module isolation, set CUDA_GRAPH_MODULES to a space-separated subset, for
# example CUDA_GRAPH_MODULES="mamba attn moe_router". Set CUDA_GRAPH_IMPL=none
# for the eager control run. SEQ_LENGTH can be changed to distinguish
# shape-dependent behavior; the CI pico shape is the default 1024.
#
# The deterministic default uses Megatron's decomposed weighted-squared-ReLU
# path. Set FUSED_WEIGHTED_SQUARED_RELU=1 on both submissions to isolate the
# run-varying fused backward observed on GB200 with the 26.07 image.
#
# The image must contain Megatron-LM's Python dependencies, but the checkout
# itself is bind-mounted from MEGATRON_ROOT. Each submission gets independent
# checkpoint, dataset, Triton, TorchInductor, and extension caches.

#SBATCH --account=coreai_resiliency_osiris
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --signal=TERM@120
#SBATCH --job-name=pico-cudagraph-det

set -Eeuo pipefail

: "${DET_WORK_DIR:?set DET_WORK_DIR to a shared Lustre directory}"
: "${MEGATRON_ROOT:?set MEGATRON_ROOT to a Megatron-LM checkout}"
: "${ENROOT_IMAGE:?set ENROOT_IMAGE to an image containing Megatron-LM dependencies}"
: "${RUN_TAG:=det-cudagraph-${SLURM_JOB_ID}}"
: "${MEGATRON_COMMIT:=aac9b2db7f0d203940f6c0a7f2852a094c18c12e}"
: "${GPUS_PER_NODE:=4}"
: "${SAVE_INTERVAL:=250}"
: "${CHECKPOINTS:=3}"
: "${MASTER_PORT:=$((20000 + SLURM_JOB_ID % 20000))}"
: "${CUDA_GRAPH_IMPL:=local}"
: "${CUDA_GRAPH_MODULES:=mamba attn moe_router}"
: "${FUSED_WEIGHTED_SQUARED_RELU:=0}"
: "${SEQ_LENGTH:=1024}"
: "${TENSOR_MODEL_PARALLEL_SIZE:=${GPUS_PER_NODE}}"
: "${EXPERT_MODEL_PARALLEL_SIZE:=${GPUS_PER_NODE}}"

MEGATRON_ROOT=$(realpath "${MEGATRON_ROOT}")
ENROOT_IMAGE=$(realpath "${ENROOT_IMAGE}")
mkdir -p "${DET_WORK_DIR}"
DET_WORK_DIR=$(realpath "${DET_WORK_DIR}")

[[ -f "${MEGATRON_ROOT}/pretrain_mamba.py" ]] || {
    echo "pretrain_mamba.py not found under MEGATRON_ROOT=${MEGATRON_ROOT}" >&2
    exit 2
}
[[ -f "${ENROOT_IMAGE}" ]] || {
    echo "ENROOT_IMAGE does not exist: ${ENROOT_IMAGE}" >&2
    exit 2
}
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
    echo "RUN_TAG must contain only letters, digits, '.', '_' and '-'" >&2
    exit 2
}
[[ "${SAVE_INTERVAL}" =~ ^[1-9][0-9]*$ && "${CHECKPOINTS}" =~ ^[1-9][0-9]*$ \
    && "${SEQ_LENGTH}" =~ ^[1-9][0-9]*$ \
    && "${TENSOR_MODEL_PARALLEL_SIZE}" =~ ^[1-9][0-9]*$ \
    && "${EXPERT_MODEL_PARALLEL_SIZE}" =~ ^[1-9][0-9]*$ ]] || {
    echo "save, sequence, and parallelism settings must be positive integers" >&2
    exit 2
}
((SEQ_LENGTH % 128 == 0)) || {
    echo "SEQ_LENGTH must be a multiple of the Mamba scan chunk size (128)" >&2
    exit 2
}
[[ "${CUDA_GRAPH_IMPL}" == local || "${CUDA_GRAPH_IMPL}" == none ]] || {
    echo "CUDA_GRAPH_IMPL must be 'local' or 'none'" >&2
    exit 2
}
[[ "${FUSED_WEIGHTED_SQUARED_RELU}" == 0 || "${FUSED_WEIGHTED_SQUARED_RELU}" == 1 ]] || {
    echo "FUSED_WEIGHTED_SQUARED_RELU must be 0 or 1" >&2
    exit 2
}
read -r -a cuda_graph_modules <<<"${CUDA_GRAPH_MODULES}"
for module in "${cuda_graph_modules[@]}"; do
    case "${module}" in
        mamba|attn|moe_router) ;;
        *)
            echo "unsupported CUDA graph module: ${module}" >&2
            exit 2
            ;;
    esac
done
graph_options=(--cuda-graph-impl "${CUDA_GRAPH_IMPL}")
if [[ "${CUDA_GRAPH_IMPL}" == local ]]; then
    ((${#cuda_graph_modules[@]})) || {
        echo "CUDA_GRAPH_MODULES must not be empty for local capture" >&2
        exit 2
    }
    graph_options+=(--cuda-graph-modules "${cuda_graph_modules[@]}")
fi
activation_options=()
if ((FUSED_WEIGHTED_SQUARED_RELU)); then
    activation_options+=(--use-fused-weighted-squared-relu)
fi

actual_commit=$(git -C "${MEGATRON_ROOT}" rev-parse HEAD)
[[ "${actual_commit}" == "${MEGATRON_COMMIT}" ]] || {
    echo "Megatron-LM checkout is ${actual_commit}; expected ${MEGATRON_COMMIT}" >&2
    exit 2
}

RUN_NAME="${RUN_TAG}-mlm-${SLURM_NNODES}n"
RUN_DIR="${DET_WORK_DIR}/${RUN_NAME}"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
DATA_CACHE_DIR="${RUN_DIR}/data-cache"
TENSORBOARD_DIR="${RUN_DIR}/tensorboard"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p \
    "${CHECKPOINT_DIR}" \
    "${DATA_CACHE_DIR}" \
    "${TENSORBOARD_DIR}" \
    "${LOG_DIR}" \
    "${RUN_DIR}/triton-cache" \
    "${RUN_DIR}/torchinductor-cache" \
    "${RUN_DIR}/torch-extensions"

if find "${CHECKPOINT_DIR}" -mindepth 1 -print -quit | grep -q .; then
    echo "checkpoint directory is not empty; choose a new RUN_TAG: ${CHECKPOINT_DIR}" >&2
    exit 2
fi

MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
TRAIN_ITERS=$((SAVE_INTERVAL * CHECKPOINTS))
WORLD_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))
((WORLD_SIZE % TENSOR_MODEL_PARALLEL_SIZE == 0)) || {
    echo "TENSOR_MODEL_PARALLEL_SIZE must divide WORLD_SIZE=${WORLD_SIZE}" >&2
    exit 2
}
((WORLD_SIZE % EXPERT_MODEL_PARALLEL_SIZE == 0)) || {
    echo "EXPERT_MODEL_PARALLEL_SIZE must divide WORLD_SIZE=${WORLD_SIZE}" >&2
    exit 2
}
TRAIN_SAMPLES=$((TRAIN_ITERS * WORLD_SIZE * 2))
if (( TRAIN_SAMPLES < 51200 )); then
    TRAIN_SAMPLES=51200
fi

export MASTER_ADDR MASTER_PORT
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MAMBA_DETERMINISTIC=1
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_NVLS_ENABLE=0
export NCCL_TUNER_PLUGIN=none
export NCCL_IB_TIMEOUT=24
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_WORKER_START=fork
export TRITON_CACHE_DIR=/workspace/run/triton-cache
export TRITON_CACHE_PATH=/workspace/run/triton-cache
export TORCHINDUCTOR_CACHE_DIR=/workspace/run/torchinductor-cache
export TORCH_EXTENSIONS_DIR=/workspace/run/torch-extensions
unset TRITON_CACHE_AUTOTUNING

options=(
    --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}"
    --expert-model-parallel-size "${EXPERT_MODEL_PARALLEL_SIZE}"
    --expert-tensor-parallel-size 1
    --pipeline-model-parallel-size 1
    --global-batch-size "${WORLD_SIZE}"
    --micro-batch-size 1
    --seq-length "${SEQ_LENGTH}"
    --max-position-embeddings "${SEQ_LENGTH}"
    --data-cache-path /workspace/run/data-cache
    --tensorboard-dir /workspace/run/tensorboard
    --log-interval "${SAVE_INTERVAL}"
    --eval-interval "$((TRAIN_ITERS * 2))"
    --exit-interval "${TRAIN_ITERS}"
    --train-samples "${TRAIN_SAMPLES}"
    --lr-decay-samples 38000
    --lr-decay-style WSD
    --lr-warmup-samples 2000
    --lr-wsd-decay-samples 40000
    --lr-wsd-decay-style minus_sqrt
    --is-hybrid-model
    --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec
    --num-layers 3
    --hidden-size 2048
    --ffn-hidden-size 1344
    --mamba-num-heads 64
    --hybrid-override-pattern 'M*E'
    --num-attention-heads 32
    --kv-channels 128
    --mtp-num-layers 2
    --mtp-hybrid-override-pattern '*E'
    --mtp-loss-scaling-factor 0.3
    --calculate-per-token-loss
    --first-last-layers-bf16
    --num-layers-at-start-in-bf16 0
    --num-layers-at-end-in-bf16 2
    --position-embedding-type none
    --untie-embeddings-and-output-weights
    --init-method-std 0.014
    --bf16
    --use-mcore-models
    --sequence-parallel
    --no-gradient-accumulation-fusion
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
    --override-opt_param-scheduler
    --num-experts 32
    --moe-shared-expert-intermediate-size 2688
    --moe-latent-size 1024
    --moe-token-dispatcher-type alltoall
    --moe-router-score-function sigmoid
    --moe-router-topk 11
    --moe-router-topk-scaling-factor 5.0
    --moe-router-enable-expert-bias
    --moe-router-dtype fp32
    --moe-router-load-balancing-type seq_aux_loss
    --moe-aux-loss-coeff 1e-4
    --moe-grouped-gemm
    --moe-permute-fusion
    --attention-backend unfused
    "${activation_options[@]}"
    --squared-relu
    --te-rng-tracker
    --ddp-num-buckets 10
    "${graph_options[@]}"
    --deterministic-mode
    --mock-data
    --tokenizer-type NullTokenizer
    --vocab-size 256000
    --lr 4.5e-4
    --min-lr 4.5e-6
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --disable-bias-linear
    --normalization RMSNorm
    --log-num-zeros-in-grad
    --log-params-norm
    --log-throughput
    --log-progress
    --logging-level 20
    --straggler-minmax-count 16
    --timing-log-option minmax
    --no-mmap-bin-files
    --no-create-attention-mask-in-dataloader
    --distributed-timeout-minutes 30
    --num-workers 1
    --disable-gloo-process-groups
    --ckpt-format torch_dist
    --ckpt-assume-constant-structure
    --ckpt-fully-parallel-save
    --ckpt-fully-parallel-save-process-group dp
    --save /workspace/run/checkpoints
    --save-interval "${SAVE_INTERVAL}"
    --save-retain-interval "${SAVE_INTERVAL}"
)

container_env=(
    MASTER_ADDR MASTER_PORT
    CUDA_DEVICE_MAX_CONNECTIONS CUBLAS_WORKSPACE_CONFIG MAMBA_DETERMINISTIC
    NCCL_ALGO NCCL_PROTO NCCL_NVLS_ENABLE NCCL_TUNER_PLUGIN NCCL_IB_TIMEOUT
    NVTE_ALLOW_NONDETERMINISTIC_ALGO NVTE_BWD_LAYERNORM_SM_MARGIN
    NVTE_FWD_LAYERNORM_SM_MARGIN PYTHONHASHSEED PYTORCH_CUDA_ALLOC_CONF
    TORCHINDUCTOR_COMPILE_THREADS TORCHINDUCTOR_WORKER_START
    TRITON_CACHE_DIR TRITON_CACHE_PATH TORCHINDUCTOR_CACHE_DIR
    TORCH_EXTENSIONS_DIR
)
container_env_csv=$(IFS=,; echo "${container_env[*]}")

echo "[det-repro] run=${RUN_NAME} nodes=${SLURM_NNODES} gpus=${WORLD_SIZE}"
echo "[det-repro] seq=${SEQ_LENGTH} tp=${TENSOR_MODEL_PARALLEL_SIZE} ep=${EXPERT_MODEL_PARALLEL_SIZE}"
echo "[det-repro] checkpoints=${CHECKPOINT_DIR} iterations=${TRAIN_ITERS}"
echo "[det-repro] megatron=${actual_commit}"

srun \
    --nodes="${SLURM_NNODES}" \
    --ntasks="${WORLD_SIZE}" \
    --ntasks-per-node="${GPUS_PER_NODE}" \
    --gpus-per-node="${GPUS_PER_NODE}" \
    --kill-on-bad-exit=1 \
    --label \
    --container-image="${ENROOT_IMAGE}" \
    --container-mounts="${MEGATRON_ROOT}:/workspace/megatron-lm,${RUN_DIR}:/workspace/run,/lustre:/lustre" \
    --container-env="${container_env_csv}" \
    --output="${LOG_DIR}/rank-%t.log" \
    --open-mode=truncate \
    bash -lc '
        export RANK="${SLURM_PROCID}"
        export WORLD_SIZE="${SLURM_NTASKS}"
        export LOCAL_RANK="${SLURM_LOCALID}"
        cd /workspace/megatron-lm
        exec python3 -u pretrain_mamba.py "$@"
    ' bash "${options[@]}"

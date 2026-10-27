#!/bin/bash
SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname ${SCRIPT_PATH})
MEGATRON_ROOT=$(${SCRIPT_DIR}/megatron_root.sh)
NATIVE_BUILD_DIR=${SCRIPT_DIR}/../build
REPO_ROOT=$(realpath ${SCRIPT_DIR}/..)

set -Eeoux pipefail

export TORCHINDUCTOR_COMPILE_THREADS=1
if [[ $(basename $0) == "run_1.sh" ]]; then
WORLD_SIZE=${WORLD_SIZE:-1}
fi
WORLD_SIZE=${WORLD_SIZE:-2}
GLOBAL_BATCH=${WORLD_SIZE}
export QUANTIZATION_TYPE_DEBUG=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Megatron asserts this on pre-Blackwell GPUs whenever CUDA graphs run with
# expandable_segments (core/transformer/cuda_graphs.py) - and this script
# enables both.
export NCCL_GRAPH_REGISTER=0
# AssertionError: Using tensor model parallelism or context parallelism require setting the environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1
export CUDA_DEVICE_MAX_CONNECTIONS=1
# FlashAttention 4's Hopper backward pass stalls, so Hopper trains with FlashAttention 2.
if [[ $(nvidia-smi --query-gpu=compute_cap --format=csv,noheader --id=0) == 9.0 ]]; then
	export NVTE_FLASH_ATTN_V4=${NVTE_FLASH_ATTN_V4:-0}
fi
export BUILDENV_DISABLE_MARCH_NATIVE=ON

MY_RUN_DIR="${REPO_ROOT}/ci/ci-runs"
RUN_DIR="${REPO_ROOT}/ci/ci-runs"
MEGATRON_LM_DIR="${MEGATRON_ROOT}"
DATACACHE_DIR="${MY_RUN_DIR}/data-cache/"
export NATIVE_CKPT_SERVER_LOG="${MY_RUN_DIR}/native_ckpt_server.log"
export NATIVE_CKPT_SERVER_HANDLER_THREADS=4
export NATIVE_CKPT_SERVER_MAX_ACTIVE_SNAPSHOTS=0
export NATIVE_CKPT_SERVER_TCP_ACCEPTORS=1
export NATIVE_CKPT_SERVER_TCP_REUSE_PORT=0
export NATIVE_CKPT_SHM_RING_BUFFER_NUM_QUEUES=2
export NATIVE_CKPT_SHM_RING_BUFFER_SIZE=128MiB
export NATIVE_CKPT_SHM_RING_BUFFER_SLOTS=4
export NATIVE_CKPT_SHM_RING_BUFFER_WRITE_TIMEOUT_SECONDS=300

BLEND_PATH="${REPO_ROOT}/../blend_files/superblend1-exposure-filtered.json"


CHECKPOINT_DIR="${MY_RUN_DIR}/checkpoints/phase1_rerun/"
TENSORBOARD_DIR="${MY_RUN_DIR}/tensorboard/phase1_rerun/"
export TRITON_CACHE_DIR="${MY_RUN_DIR}/triton-cache/"

mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${DATACACHE_DIR}
mkdir -p ${TENSORBOARD_DIR}
mkdir -p ${TRITON_CACHE_DIR}

#export TRITON_CACHE_DIR="/home/tmp/triton_cache"
#export TRITON_CACHE_MANAGER="megatron.core.ssm.triton_cache_manager:ParallelFileCacheManager"

SEQ_LEN=4096
TRAIN_SAMPLES=5000
LR_WARMUP_SAMPLES=24414063
LR_DECAY_SAMPLES=3048706055
LR_WSD_DECAY_SAMPLES=610351563


OPTIONS=(
	#--enable-ft-package
	--result-rejected-tracker-filename ${CHECKPOINT_DIR}/result_rejected_tracker.txt

	--first-last-layers-bf16
	--num-layers-at-start-in-bf16 0
	--num-layers-at-end-in-bf16 2
	#--fp4-format e2m1
	#--fp4-recipe nvfp4

	--mtp-num-layers 2
	--mtp-hybrid-override-pattern EE
	--calculate-per-token-loss
	--mtp-loss-scaling-factor 0.3

	--enable-cuda-graph
	--cuda-graph-scope mamba attn moe_router
	--te-rng-tracker
	--ddp-num-buckets 10
	--high-priority-stream-groups ep
	--manual-gc
	--manual-gc-interval 1

	--moe-latent-size 1024
	--moe-permute-fusion
	--cross-entropy-loss-fusion
	--cross-entropy-fusion-impl native
	--use-fused-weighted-squared-relu
	--no-gradient-accumulation-fusion

	--moe-token-dispatcher-type alltoall
	--moe-router-score-function sigmoid
	--moe-grouped-gemm
	--num-experts 32
	--moe-router-topk 11
	--moe-aux-loss-coeff 1e-4
	--moe-router-topk-scaling-factor 5.0
	--moe-router-enable-expert-bias
	--moe-router-dtype fp32
	--moe-router-load-balancing-type seq_aux_loss
	--moe-shared-expert-intermediate-size 2688

	--attention-backend flash
	--num-workers 1
	--disable-gloo-process-groups
	--ckpt-format torch_dist
	--ckpt-fully-parallel-save
	--ckpt-fully-parallel-load
	--ckpt-assume-constant-structure
#	--async-save
#	--use-persistent-ckpt-worker

	--squared-relu
	--no-mmap-bin-files
	--distributed-timeout-minutes 10
	--exit-duration-in-mins 5750
	--no-create-attention-mask-in-dataloader

	--overlap-grad-reduce
	--overlap-param-gather
	--sequence-parallel
	--tensor-model-parallel-size ${WORLD_SIZE}
	--expert-model-parallel-size ${WORLD_SIZE}
	--expert-tensor-parallel-size 1
	--pipeline-model-parallel-size 1
	--use-distributed-optimizer
	--override-opt_param-scheduler

	--mamba-num-heads 64
	--is-hybrid-model
	--untie-embeddings-and-output-weights
	--init-method-std 0.014
	--position-embedding-type none
	--num-layers 3
	--hidden-size 2048
	--num-attention-heads 32
	--group-query-attention
	--num-query-groups 2
	--hybrid-override-pattern "M*E"
	--ffn-hidden-size 1344
	--kv-channels 128
	--seq-length ${SEQ_LEN}
	--max-position-embeddings ${SEQ_LEN}
	--train-samples ${TRAIN_SAMPLES}
	--lr-decay-style WSD
	--lr-warmup-samples ${LR_WARMUP_SAMPLES}
	--lr-decay-samples ${LR_DECAY_SAMPLES}
	--lr-wsd-decay-style minus_sqrt
	--lr-wsd-decay-samples ${LR_WSD_DECAY_SAMPLES}
	--save ${CHECKPOINT_DIR}
	--load ${CHECKPOINT_DIR}
	--mock-data
	#--per-split-data-args-path ${BLEND_PATH}
	--data-cache-path ${DATACACHE_DIR}
	--tiktoken-pattern v2
	--tokenizer-type NullTokenizer
	--vocab-size 131072
	--micro-batch-size 1
	--global-batch-size ${GLOBAL_BATCH}
	--lr 4.5e-4
	--min-lr 4.5e-6
	--weight-decay 0.1
	--clip-grad 1.0
	--attention-dropout 0.0
	--hidden-dropout 0.0
	--disable-bias-linear
	--normalization RMSNorm
	--adam-beta1 0.9
	--adam-beta2 0.95
	--log-interval 1
	--log-params-norm
	--log-num-zeros-in-grad
	--log-throughput
	--log-progress
	--logging-level 20
	--disable-straggler-on-startup
	--straggler-minmax-count 16
	--check-weight-hash-across-dp-replicas-interval 20000
	--ddp-pad-buckets-for-high-nccl-busbw
	--timing-log-option minmax
	--save-interval 1
	--save-retain-interval 1
	--exit-interval 2
	--eval-interval 1000
	--eval-iters 14
	--bf16
	--use-mcore-models
	--spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec
	--tensorboard-dir "${TENSORBOARD_DIR}"
)

MEGATRON_LAUNCHER=${MEGATRON_LAUNCHER:-"torchrun --nproc_per_node=${WORLD_SIZE}"}

if [[ $WORLD_SIZE -gt 1 ]]; then
	PYTHONPATH=${MEGATRON_ROOT}:${REPO_ROOT}/.. ${MEGATRON_LAUNCHER} ${MEGATRON_ROOT}/pretrain_mamba.py ${OPTIONS[@]} 2>&1 | tee ${MY_RUN_DIR}/log.txt
else
	PYTHONPATH=${MEGATRON_ROOT}:${REPO_ROOT}/.. MASTER_ADDR=localhost MASTER_PORT=48000 python3 ${MEGATRON_ROOT}/pretrain_mamba.py ${OPTIONS[@]}  2>&1 | tee ${MY_RUN_DIR}/log.txt
fi

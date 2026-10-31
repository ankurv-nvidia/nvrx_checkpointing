#!/bin/bash

#SBATCH --partition=batch
#SBATCH --account=coreai_resiliency_osiris
#SBATCH --ntasks=64
#SBATCH --time=00:30:00
#SBATCH --mem=0
#SBATCH --job-name=test_16_nodes
#SBATCH --gpus=64


##SBATCH --exclusive
##SBATCH --reservation=oos_triage_nemotron_super_v3_poor_performers
##SBATCH --ntasks-per-node=4

# squeue --format="%.18i %.9P %.38j %.8u %.2t %.10M %.10l %.6D %R %.20S" -A coreai_resiliency_osiris
export NCCL_IB_TIMEOUT=24
export UCX_MEM_MMAP_HOOK_MODE=none
export UCX_MEM_CUDA_HOOK_MODE=none
export UCX_MEM_MALLOC_HOOKS=n
export UCX_ERROR_SIGNALS=

export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export TORCHINDUCTOR_WORKER_START=fork

export QUANTIZATION_TYPE_DEBUG=1
# AssertionError: expandable_segments:True may not be safe when using CUDA Graphs with some specific parallel settings. The training may crash with illegal memory access.
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# AssertionError: Using tensor model parallelism or context parallelism require setting the environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1
export CUDA_DEVICE_MAX_CONNECTIONS=1

RUN_TAG=${1:-${SLURM_JOB_ID}}

#IMAGE="/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/nemotron6/images/high_stride/nemo_megatron_perf_optimization:mcore-moe-pytorch25.06-te2.10.0.dev0-cudnn9.11.0.92-main0807-hybridep-nvfp4-arm.sqsh"

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
if [ -n "${SLURM_JOB_ID:-}" ] ; then
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
    ENV_LOG_FILENAME=${NAME}_${SLURM_JOB_ID}_${DATETIME}.env.log
else
    SCRIPT_PATH=$(realpath "$0")
    ENV_LOG_FILENAME=${NAME}_${DATETIME}.env.log
fi

SCRIPT_DIR=$(dirname ${SCRIPT_PATH})
MEGATRON_ROOT=$(bash ${SCRIPT_DIR}/../prepare.sh megatron_root)
ENROOT_IMAGE_BASE=${ENROOT_IMAGE_BASE:-${MEGATRON_ROOT}/../nemotron6-base.sqsh}
ENROOT_IMAGE_OUT=${ENROOT_IMAGE_OUT:-${MEGATRON_ROOT}/../nemotron6-native.sqsh}
RUN_DIR="${MEGATRON_ROOT}/../run_${RUN_TAG}"

IMAGE=${ENROOT_IMAGE_OUT}

MEGATRON_LM_DIR="${MEGATRON_ROOT}"
DATACACHE_DIR="${MEGATRON_ROOT}/../data-cache/"


TOKENIZER_MODEL_PATH="${MEGATRON_ROOT}/../tokenizers/multiMixV8.gpt4o_nc_sd.500000.128k.vocab.json"
BLEND_PATH="${MEGATRON_ROOT}/../blend_files/superblend1-exposure-filtered.json"

LOGS_DIR="${RUN_DIR}/logs"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
TENSORBOARD_DIR="${RUN_DIR}/tensorboard"
#INIT_CHECKPOINT_DIR="${RUN_DIR}/starting_ckpt/"
INIT_CHECKPOINT_DIR=${CHECKPOINT_DIR}

mkdir -p ${LOGS_DIR}
mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${DATACACHE_DIR}
mkdir -p ${TENSORBOARD_DIR}

export TRITON_CACHE_DIR="${RUN_DIR}/../triton-cache/"
#export TRITON_CACHE_DIR="/home/tmp/triton_cache"
#export TRITON_CACHE_MANAGER="megatron.core.ssm.triton_cache_manager:ParallelFileCacheManager"
mkdir -p ${TRITON_CACHE_DIR}


#REPO_DIR=${CURRENT_DIR}

################################################################
### Log environment
################################################################
echo "<< START PATHS >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "IMAGE=${IMAGE}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
#echo "OUTPUT_ROOT=${OUTPUT_ROOT}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "MEGATRON_LM_DIR=${MEGATRON_LM_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "RUN_DIR=${RUN_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "LOGS_DIR=${LOGS_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "DATACACHE_DIR=${DATACACHE_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "TENSORBOARD_DIR=${TENSORBOARD_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "SCRIPT_DIR=${SCRIPT_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
#echo "REPO_DIR=${REPO_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "<< END PATHS >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "nn" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

echo "<< START GIT >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "GIT LOG" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
git -C ${MEGATRON_LM_DIR} log --oneline -1 |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "nn" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "GIT STATUS" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
git -C ${MEGATRON_LM_DIR} status --porcelain --branch |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "nn" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "GIT DIFF" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
git -C ${MEGATRON_LM_DIR} diff |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "<< END GIT >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "nn" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

echo "<< START ENV >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
env |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "<< END ENV >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}


#--result-rejected-tracker-filename ${RESULT_REJECTED_TRACKER_FILENAME}
#--iterations-to-skip ${ITERATIONS_TO_SKIP}
#--rerun-mode validate_results
#
#--enable-experimental
#--moe-shared-expert-overlap

	# Can not use with FP4

	# MXFP8
	#--moe-router-padding-for-fp8
	#--fp8-format e4m3
	#--fp8-recipe mxfp8
	#--fp8-param-gather
	#--reuse-grad-buf-for-mxfp8-param-ag

	# Additional options
	#--recompute-modules layernorm moe_act
	#
	#--recompute-granularity selective
	#--recompute-modules moe
	#
	#--tp-comm-overlap
	#

SEQ_LEN=4096
# TRAIN_SAMPLES=3051757813
# LR_WARMUP_SAMPLES=24414063
# LR_DECAY_SAMPLES=3048706055
# LR_WSD_DECAY_SAMPLES=610351563
TRAIN_SAMPLES=30517
LR_WARMUP_SAMPLES=244
LR_DECAY_SAMPLES=30487
LR_WSD_DECAY_SAMPLES=6103


# --keep-mtp-spec-in-bf16
# --keep-mamba-stack-attention-linear-in-bf16
# --keep-mamba-out-proj-in-mxfp8
# --keep-moe-latent-projections-in-bf16
	# --moe-shared-expert-compute-before-router
	# --log-memory-interval 200
# --hybrid-override-pattern MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME
#	 --num-layers 88

INTERVAL=1

OPTIONS=(
	--first-last-layers-bf16
	--num-layers-at-start-in-bf16 0
	--num-layers-at-end-in-bf16 14
#	--fp4-format e2m1
#	--fp4-recipe nvfp4

	--mtp-num-layers 2
	--mtp-hybrid-override-pattern "*E"
	--calculate-per-token-loss
	--mtp-loss-scaling-factor 0.3

	--enable-cuda-graph
	--cuda-graph-scope mamba attn moe_router
	--te-rng-tracker
	--ddp-num-buckets 10
	--high-priority-stream-groups ep
	--manual-gc
	--manual-gc-interval ${INTERVAL}

	--moe-latent-size 1024
	--moe-permute-fusion
	--cross-entropy-loss-fusion
	--cross-entropy-fusion-impl native
	--use-fused-weighted-squared-relu

	--moe-token-dispatcher-type alltoall
	--moe-router-score-function sigmoid
	--moe-grouped-gemm
	--num-experts 512
	--moe-router-topk 22
	--moe-aux-loss-coeff 1e-4
	--moe-router-topk-scaling-factor 5.0
	--moe-router-enable-expert-bias
	--moe-router-dtype fp32
	--moe-router-load-balancing-type seq_aux_loss
	--moe-shared-expert-intermediate-size 5376

	--attention-backend flash
	--num-workers 1
	--disable-gloo-process-groups
	--ckpt-format torch_dist
	--ckpt-fully-parallel-save
	--ckpt-fully-parallel-load
	--ckpt-assume-constant-structure
	--async-save
	--use-persistent-ckpt-worker

	--squared-relu
	--no-mmap-bin-files
	--distributed-timeout-minutes 60
	--exit-duration-in-mins 5750
	--no-create-attention-mask-in-dataloader

	--overlap-grad-reduce
	--overlap-param-gather
	--sequence-parallel
	--tensor-model-parallel-size 2
	--expert-model-parallel-size 64
	--expert-tensor-parallel-size 1
	--pipeline-model-parallel-size 1
	--use-distributed-optimizer
	--override-opt_param-scheduler

	--mamba-num-heads 128
	--is-hybrid-model
	--untie-embeddings-and-output-weights
	--init-method-std 0.014
	--position-embedding-type none
	--num-layers 44
	--hidden-size 4096
	--num-attention-heads 32
	--group-query-attention
	--num-query-groups 2
	--hybrid-override-pattern MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEME
	--ffn-hidden-size 2688
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
	--load ${INIT_CHECKPOINT_DIR}
	--mock-data
	--data-cache-path ${DATACACHE_DIR}
	--tiktoken-pattern v2
	--tokenizer-type TikTokenizer
	--tokenizer-model ${TOKENIZER_MODEL_PATH}
	--micro-batch-size 1
	--global-batch-size 3072
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
	--log-interval 10
	--log-params-norm
	--log-num-zeros-in-grad
	--log-throughput
	--log-progress
	--logging-level 10
	--log-straggler
	--disable-straggler-on-startup
	--straggler-minmax-count 16
	--check-weight-hash-across-dp-replicas-interval 20000
	--ddp-pad-buckets-for-high-nccl-busbw
	--timing-log-option minmax
	--save-interval ${INTERVAL}
	--save-retain-interval ${INTERVAL}
	--exit-interval 4
	--eval-interval 1000
	--eval-iters 14
	--bf16
	--use-mcore-models
	--spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec
	--tensorboard-dir ${TENSORBOARD_DIR}
)

RUN_CMD="numactl --localalloc --physcpubind=$([ "${SLURM_LOCALID:-0}" -le 1 ] && echo 0-63 || echo 72-135) python3 -u ${MEGATRON_LM_DIR}/pretrain_mamba.py ${OPTIONS[@]}"
# run_cmd="numactl --cpunodebind=$((SLURM_LOCALID/4)) --membind=$((SLURM_LOCALID/4)) python3 -u ${REPO_DIR}/pretrain_mamba.py ${options}"

srun -l \
     --container-image=${IMAGE} \
     --container-mounts="${HOME}:${HOME},/lustre:/lustre" \
     --container-env=NCCL_IB_TIMEOUT,UCX_MEM_MMAP_HOOK_MODE,UCX_MEM_CUDA_HOOK_MODE,UCX_MEM_MALLOC_HOOKS,UCX_ERROR_SIGNALS \
     --output="${LOGS_DIR}/%x_%j_${DATETIME}.log" \
     sh -c "${RUN_CMD}"


#!/bin/bash

#SBATCH --qos=batch_large_long
#SBATCH --reservation=sla_res_nemotron_super_v3
#SBATCH --partition=batch_long
#SBATCH --account=llmservice_fm_text
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=768
#SBATCH --time=96:00:00
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --segment=16
#SBATCH --mem=0
#SBATCH --job-name=12B_hybrid_moe_25T_phase1
#SBATCH --comment='{"APS": {"auto_resume_mode": "requeue", "nvrx": "enabled"}}'


export NCCL_IB_TIMEOUT=24
export UCX_MEM_MMAP_HOOK_MODE=none
export UCX_MEM_CUDA_HOOK_MODE=none
export UCX_MEM_MALLOC_HOOKS=n
export UCX_ERROR_SIGNALS=

export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export TORCHINDUCTOR_WORKER_START=fork

export QUANTIZATION_TYPE_DEBUG=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


NAME="12B_hybrid_moe_25T_phase1"
RUN_DIR="/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/nemotron6/12b_hybrid_moe/"
IMAGE="${RUN_DIR}/../images/high_stride/nemo_megatron_perf_optimization:mcore-moe-pytorch25.06-te2.10.0.dev0-cudnn9.11.0.92-main0807-hybridep-nvfp4-nvrxupdated-arm.sqsh"
MEGATRON_LM_DIR="${RUN_DIR}/../code/"
DATACACHE_DIR="${RUN_DIR}/../data-cache/"

TOKENIZER_MODEL_PATH="${RUN_DIR}/../tokenizers/multiMixV8.gpt4o_nc_sd.500000.128k.vocab.json"
BLEND_PATH="${RUN_DIR}/../blend_files/superblend1-exposure-filtered.json"


LOGS_DIR="${RUN_DIR}/logs/phase1_rerun/"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints/phase1_rerun/"
TENSORBOARD_DIR="${RUN_DIR}/tensorboard/phase1_rerun/"

mkdir -p ${LOGS_DIR}
mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${DATACACHE_DIR}
mkdir -p ${TENSORBOARD_DIR}

export TRITON_CACHE_DIR="${RUN_DIR}/../triton-cache/"
#export TRITON_CACHE_DIR="/home/tmp/triton_cache"
#export TRITON_CACHE_MANAGER="megatron.core.ssm.triton_cache_manager:ParallelFileCacheManager"


DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
if [ -n "${SLURM_JOB_ID:-}" ] ; then
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
    ENV_LOG_FILENAME=${NAME}_${SLURM_JOB_ID}_${DATETIME}.env.log
else
    SCRIPT_PATH=$(realpath "$0")
    ENV_LOG_FILENAME=${NAME}_${DATETIME}.env.log
fi

SCRIPT_DIR=$(dirname ${SCRIPT_PATH})
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
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

echo "<< START GIT >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "GIT LOG" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
git -C ${MEGATRON_LM_DIR} log --oneline -1 |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "GIT STATUS" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
git -C ${MEGATRON_LM_DIR} status --porcelain --branch |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "GIT DIFF" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
git -C ${MEGATRON_LM_DIR} diff |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "<< END GIT >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

echo "<< START ENV >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
env |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "<< END ENV >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

ITERATIONS_TO_SKIP="337071"
#--rerun-mode validate_results \
#
#--enable-experimental \
#--moe-shared-expert-overlap \

	# Can not use with FP4

	# MXFP8
	#--moe-router-padding-for-fp8 \
	#--fp8-format e4m3 \
	#--fp8-recipe mxfp8 \
	#--fp8-param-gather \
	#--reuse-grad-buf-for-mxfp8-param-ag \

	# Additional options
	#--recompute-modules layernorm moe_act \
	#
	#--recompute-granularity selective \
	#--recompute-modules moe \
	#
	#--tp-comm-overlap \
	#--log-straggler \
	#
	#--ckpt-fully-parallel-load-process-group ep_dp \
	#--ckpt-fully-parallel-load-exchange-algo gather_rounds \

SEQ_LEN=8192
TRAIN_SAMPLES=3051757813
LR_WARMUP_SAMPLES=24414063
LR_DECAY_SAMPLES=3048706055
LR_WSD_DECAY_SAMPLES=610351563

OPTIONS=" \
	--enable-ft-package \
	--result-rejected-tracker-filename ${CHECKPOINT_DIR}/result_rejected_tracker.txt \
	--iterations-to-skip ${ITERATIONS_TO_SKIP} \
	\
	--keep-mtp-spec-in-bf16 \
	--keep-mamba-stack-attention-linear-in-bf16 \
	--keep-mamba-out-proj-in-mxfp8 \
	--keep-moe-latent-projections-in-bf16 \
	--first-last-layers-bf16 \
	--num-layers-at-start-in-bf16 0 \
	--num-layers-at-end-in-bf16 14 \
	--fp4-format e2m1 \
	--fp4-recipe nvfp4 \
	\
	--mtp-spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
	--mtp-num-layers 2 \
	--mtp-hybrid-override-pattern \"*E\" \
	--calculate-per-token-loss \
	--mtp-loss-scaling-factor 0.3 \
	\
	--enable-cuda-graph \
	--cuda-graph-scope mamba attn moe_router \
	--te-rng-tracker \
	--ddp-num-buckets 10 \
	--high-priority-stream-groups ep \
	--manual-gc \
	--manual-gc-interval 10 \
	--moe-shared-expert-compute-before-router \
	\
	--moe-latent-size 1024 \
	--moe-permute-fusion \
	--cross-entropy-loss-fusion \
	--cross-entropy-fusion-impl native \
	--use-fused-weighted-squared-relu \
	\
	--moe-token-dispatcher-type alltoall \
	--moe-router-score-function sigmoid \
	--moe-grouped-gemm \
	--num-experts 512 \
	--moe-router-topk 22 \
	--moe-aux-loss-coeff 1e-4 \
	--moe-router-topk-scaling-factor 5.0 \
	--moe-router-enable-expert-bias \
	--moe-router-dtype fp32 \
	--moe-router-load-balancing-type seq_aux_loss \
	--moe-shared-expert-intermediate-size 5376 \
	\
	--attention-backend flash \
	--num-workers 1 \
	--disable-gloo-process-groups \
	--ckpt-format torch_dist \
	--ckpt-fully-parallel-save \
	--ckpt-fully-parallel-load \
	--ckpt-assume-constant-structure \
	--async-save \
	--use-persistent-ckpt-worker \
        \
	--squared-relu \
        --no-mmap-bin-files \
	--distributed-timeout-minutes 10 \
	--exit-duration-in-mins 5750 \
	--no-create-attention-mask-in-dataloader \
        \
        --overlap-grad-reduce \
        --overlap-param-gather \
        --sequence-parallel \
        --tensor-model-parallel-size 2 \
        --expert-model-parallel-size 64 \
	--expert-tensor-parallel-size 1 \
	--pipeline-model-parallel-size 1 \
	--use-distributed-optimizer \
	--override-opt_param-scheduler \
        \
	--mamba-num-heads 128 \
	--is-hybrid-model \
	--untie-embeddings-and-output-weights \
        --init-method-std 0.014 \
        --position-embedding-type none \
        --num-layers 88 \
        --hidden-size 4096 \
        --num-attention-heads 32 \
        --group-query-attention \
        --num-query-groups 2 \
        --hybrid-override-pattern MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME \
	--ffn-hidden-size 2688 \
	--kv-channels 128 \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${SEQ_LEN} \
        --train-samples ${TRAIN_SAMPLES} \
        --lr-decay-style WSD \
        --lr-warmup-samples ${LR_WARMUP_SAMPLES} \
        --lr-decay-samples ${LR_DECAY_SAMPLES} \
	--lr-wsd-decay-style minus_sqrt \
	--lr-wsd-decay-samples ${LR_WSD_DECAY_SAMPLES} \
	--save ${CHECKPOINT_DIR} \
        --load ${CHECKPOINT_DIR} \
	--per-split-data-args-path ${BLEND_PATH} \
	--data-cache-path ${DATACACHE_DIR} \
        --tiktoken-pattern v2 \
	--tokenizer-type TikTokenizer \
        --tokenizer-model ${TOKENIZER_MODEL_PATH} \
        --micro-batch-size 1 \
        --global-batch-size 3072 \
        --lr 4.5e-4 \
        --min-lr 4.5e-6 \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --disable-bias-linear \
        --normalization RMSNorm \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --log-interval 100 \
	--log-params-norm \
	--log-num-zeros-in-grad \
	--log-throughput \
	--log-progress \
	--log-energy \
	--log-memory-interval 500 \
	--logging-level 20 \
	--disable-straggler-on-startup \
	--straggler-minmax-count 16 \
	--check-weight-hash-across-dp-replicas-interval 20000 \
	--ddp-pad-buckets-for-high-nccl-busbw \
	--timing-log-option minmax \
        --save-interval 250 \
	--save-retain-interval 1000 \
        --eval-interval 1000 \
        --eval-iters 14 \
        --bf16 \
        --use-mcore-models \
        --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
        --tensorboard-dir ${TENSORBOARD_DIR}"

#wrong?
#RUN_CMD="numactl --localalloc --physcpubind=$([ "${SLURM_LOCALID:-0}" -le 1 ] && echo 0-63 || echo 72-135) python -u ${MEGATRON_LM_DIR}/pretrain_mamba.py ${OPTIONS}"

#RUN_CMD="numactl --localalloc --physcpubind=\$([ \"\${SLURM_LOCALID:-0}\" -le 1 ] && echo 0-63 || echo 72-135) python -u ${MEGATRON_LM_DIR}/pretrain_mamba.py ${OPTIONS}"
#RUN_CMD="numactl --cpunodebind=\$((SLURM_LOCALID/2)) --membind=\$((SLURM_LOCALID/2)) python -u ${MEGATRON_LM_DIR}/pretrain_mamba.py ${OPTIONS}"

export RDZV_HOST=$(hostname)
export RDZV_PORT=29400
export NVRX_GPUS_PER_NUMA=2
LAUNCHER_CMD="ft_launcher"
LAUNCHER_ARGS=" \
    --nnodes $SLURM_JOB_NUM_NODES \
    --nproc-per-node 4 \
    --rdzv-id ${SLURM_JOB_ID} \
    --rdzv-backend c10d \
    --rdzv-endpoint ${RDZV_HOST}:${RDZV_PORT} \
    --rdzv-conf join_timeout=300 \
    --max-restarts 3 \
    --ft-restart-policy any-failed \
    --ft-rdzv-impl barrier \
    --ft-numa-bind-strict \
    --ft-max-no-progress-restarts 3 \
    --ft-initial-rank-heartbeat-timeout null \
    --ft-rank-heartbeat-timeout null \
    --ft-rank-section-timeouts setup:1200,step:180,checkpointing:420 \
    --ft-rank-out-of-section-timeout 720 \
"

RUN_CMD="${LAUNCHER_CMD} ${LAUNCHER_ARGS} ${MEGATRON_LM_DIR}/pretrain_mamba.py ${OPTIONS}"

srun -l --mpi=none --ntasks=$SLURM_JOB_NUM_NODES \
     --ntasks-per-node=1 \
     \
     --container-image=${IMAGE} \
     --container-mounts="/lustre:/lustre" \
     --container-env=NCCL_IB_TIMEOUT,UCX_MEM_MMAP_HOOK_MODE,UCX_MEM_CUDA_HOOK_MODE,UCX_MEM_MALLOC_HOOKS,UCX_ERROR_SIGNALS \
     --output="${LOGS_DIR}/%x_%j_${DATETIME}.log" \
     sh -c "trap \"echo [\$(date '+%Y-%m-%d %H:%M:%S,%3N')] [ALWAYS] [RankMonServer.Restarter] [NestedRestarter] name=[InJob] state=aborted\" EXIT; eval \"${RUN_CMD}\""

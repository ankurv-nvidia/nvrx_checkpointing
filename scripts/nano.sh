#!/bin/bash

#SBATCH --account=coreai_resiliency_osiris
#SBATCH --partition=batch
#SBATCH --time=01:00:00

#SBATCH --job-name=nemotron-moe

#SBATCH --output=slurm-logs/%x_%j_%N.out
#SBATCH --error=slurm-logs/%x_%j_%N.err

#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --overcommit
#SBATCH --comment=metrics
# NOTE: --nodes and --segment are *not* set here.  submit.sh derives both
# from $NODES (ladder rung) and passes them on the sbatch command line so
# one launch.sh can serve every rung without duplication.  Per the MARS
# GB200 doc, the scheduler defaults <16-node multiple-of-2 jobs to
# --segment=2, which would split our 4-node and 8-node rungs across
# multiple NVLink blocks — submit.sh forces --segment=NODES on rungs ≤16
# so the whole allocation lands in one block (EP within NVLink domain).
# Rungs >16 (32/64/128) cap at --segment=16 (= block size); the
# scheduler then distributes the job across 2/4/8 blocks, but EP=32
# still fits in one block so the EP collective stays on NVLink.

# Env-var contract (set on the sbatch invocation, all have defaults):
#
#   NODES            — ladder rung; one of 1, 2, 4, 8, 16, 32, 64, 128.
#                      Drives --segment (NVLink-block packing); TP and
#                      EP defaults come from MODEL_SIZE below.  Default
#                      2 (8 GPUs).
#   CKPT_STRATEGY    — checkpoint backend to benchmark.  One of:
#                        async-nvrx       (default; torch_dist + NVRX async
#                                          worker; GPU→CPU via CUDA IPC)
#                        async-nvrx-shm   (async-nvrx + --async-ckpt-use-cpu-
#                                          shm — copies tensors to /dev/shm
#                                          per-tensor before worker handoff;
#                                          avoids MNNVL fabric IPC handle
#                                          exhaustion at scale)
#                        async-mcore      (torch_dist + Megatron's built-in
#                                          DCP async)
#                        async-mcore-shm  (async-mcore + --async-ckpt-use-cpu-
#                                          shm)
#                        sync-torch-dist  (torch_dist, no async — blocking
#                                          DCP save on the training thread)
#                        fsdp-dtensor     (--ckpt-format fsdp_dtensor; FSDP
#                                          native sharded save)
#   MXFP8=0/1        — Precision recipe selector.  Default 0 (pure bf16,
#                      matching Deepak's nemotron3 reference scripts).
#                      Set 1 to enable mxfp8 (--fp8-recipe mxfp8 +
#                      --fp8-format e4m3 + --fp8-param-gather + the
#                      MoE-specific quantization knobs) when
#                      benchmarking fp8 checkpoint cost.  Blackwell
#                      only.  Flip per-run on the submit.sh command
#                      line (e.g. MXFP8=1 NODES=8 MODEL_SIZE=nano ...).
#   HYBRIDEP=1       — flex+hybridep MoE token dispatcher.  Set 0 to fall
#                      back to alltoall (use if the container doesn't have
#                      the hybridep dep installed).
#   PRODUCTION_BATCH=0/1 — Batch-size regime selector.  Default 0
#                      (testing scale: GBS=WORLD_SIZE, MBS=1, keeps
#                      iters short on 8–128 node experiments).
#                      Set 1 to match Deepak's nemotron3 scripts
#                      (GBS=3072, ultra MBS=2) — only useful at
#                      production scale (384+ nodes); at smaller
#                      rungs the per-iter microbatch count balloons
#                      and iters take many seconds each.
#   PERFETTO=0/1     — emit per-process Perfetto JSON for the async-ckpt
#                      pipeline.  Most informative under
#                      CKPT_STRATEGY=async-nvrx; for other strategies only
#                      the outer Megatron spans populate (the NVRX-specific
#                      sub-spans never fire).
#   WARMUP_PINNED_POOL=0/1 — fire a noop warmup save right after first
#                      optimizer step to hide cudaHostAlloc on the
#                      first-real-save critical path.  Only meaningful
#                      with CKPT_STRATEGY=async-nvrx.
#   WARMUP_PROBE_GATHER=0/1 — independent of WARMUP_PINNED_POOL: when
#                      the warmup save fires, insert a tiny-payload
#                      dist.gather_object immediately before it and
#                      time it in the perfetto trace.  Isolates one-
#                      shot collective / process-group warm-up cost
#                      from payload-shaped Branch-B gather cost.
#                      Requires WARMUP_PINNED_POOL=1|plan-only and
#                      PERFETTO=1 to be useful (no warmup → no probe
#                      site; no tracer → no recorded timing).
#   STATE_DICT_DUMP=0/1 — on the first checkpoint, rank 0 prints a
#                      per-top-level-key tensor/byte breakdown of the
#                      state dict and torch.save's it to
#                      $STATE_DICT_DUMP_PATH (default: $LOG_DIR/
#                      state_dict_rank0_ckpt0.pt).  Debug aid for
#                      "why does assemble state dict take so long" —
#                      subsequent checkpoints pass through unchanged.
#   GC_CYCLE_PROBE=0/1 — bracket save_checkpoint with the cyclic GC in
#                      DEBUG_SAVEALL mode and, on rank 0, print which
#                      tensors (device/dtype/shape/bytes) end up trapped
#                      in reference cycles plus the CUDA bytes that
#                      gc.collect() reclaims.  Diagnoses whether the
#                      fp8 dequant temporaries need gc.collect() at all
#                      or are held by a plain lingering reference.
#                      GC_CYCLE_PROBE_MAX (default 3) caps probed saves;
#                      GC_CYCLE_PROBE_TOPN (default 20) caps the listing.
#
# Topology / model overrides (rarely changed; default derived from NODES):
#   TP, EP, PP, ETP, NUM_EXPERTS, GLOBAL_BATCH_SIZE, MICRO_BATCH_SIZE
#
# Note: nsys profiling is not wired up here — analysis uses Perfetto and
# the log-based iter_time_hist scripts.

: "${NODES:=2}"
: "${CKPT_STRATEGY:=async-nvrx}"
: "${PERFETTO:=0}"
: "${WARMUP_PINNED_POOL:=0}"
# MXFP8 / HYBRIDEP defaults are set after the MODEL_SIZE case below
# because they only apply to the MoE presets.

# ---------------------------------------------------------------------------
# Per-rung --segment.  Pins how many nodes one --segment block must
# contain so the scheduler doesn't split the allocation across NVLink
# blocks (block size = 16 nodes = 64 GPUs).  For NODES ≤ 16 we pin
# SEGMENT=NODES so the whole job lands in one block, keeping EP
# collectives on NVLink.  For NODES > 16 we cap at SEGMENT=16; the
# scheduler distributes the job across multiple blocks, but EP groups
# (≤64 GPUs for all current MODEL_SIZE presets) still fit in one block.
# Cross-block traffic is the DP/EDP all-reduce, which goes over RoCE/IB
# at any rung >4 nodes anyway.
#
# TP, EP, PP, ETP defaults are set per MODEL_SIZE below (since the
# right parallelism depends on the model, not the node count).  DP is
# implicit: DP = NODES * GPUS_PER_NODE / TP / PP.  EDP (expert-DP
# replication) = world_size / (EP * ETP * PP).  Each preset has a
# minimum-rung floor below which world_size < EP*ETP*PP and
# init_model_parallel will fail — pass EP=<lower> explicitly to run
# smaller, or switch to MODEL_SIZE=smoke for plumbing-only sanity.
# ---------------------------------------------------------------------------
GPUS_PER_NODE=4
case ${NODES} in
    1)   SEGMENT=1  ;;
    2)   SEGMENT=2  ;;
    4)   SEGMENT=4  ;;
    8)   SEGMENT=8  ;;
    16)  SEGMENT=16 ;;
    32)  SEGMENT=16 ;;
    64)  SEGMENT=16 ;;
    128) SEGMENT=16 ;;
    *)
        echo "ERROR: unsupported NODES=${NODES}.  Supported rungs: 1,2,4,8,16,32,64,128." >&2
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# Model size preset.  Five values, split into dense vs MoE via IS_MOE.
# Each preset fixes TP, EP, PP, ETP defaults (see table below); user
# env-vars still win via `:=`.  TP and EP cannot be honored on rungs
# where world_size < EP * ETP * PP — that's the minimum-rung floor.
#
#   preset  IS_MOE  TP  EP   min nodes  production reference
#   ──────  ──────  ──  ──   ─────────  ──────────────────────────────
#   smoke   0       4   1     1         (dense Mamba sanity gate)
#   nano    1       2   32    8         bespoke; no production log
#   super   1       2   64   16         12b_hybrid_moe/logs/phase1
#   ultra   1       8   64   16         55b_hybrid_moe/logs/phase2_20t
#   test    1       8   64   16         (14L testbed at ultra dims)
#
#   smoke   — dense Mamba model from the cw-dfw recipe (Nemotron5 8B
#             class, 52 layers, hidden=4096, no experts).  Fits a single
#             GB200 node at TP=4.  Validates the launcher / container /
#             NCCL / ckpt plumbing without engaging the MoE dispatcher,
#             EP sharding, MXFP8, or the precision-aware optimizer.
#   nano    — fixed-EP scaling experiment.  EP=32 plateau lets the
#             checkpoint-cost scaling curve be read against DP alone.
#   super,  — production-shaped MoE recipes.  TP and EP match the
#   ultra     phase logs above; on-cluster runs at lower rungs may OOM
#             well before the parallelism limits — be ready for that
#             if you're benchmarking below the production scale.
#
# Each preset fixes: NUM_LAYERS, HIDDEN_SIZE, NUM_ATTENTION_HEADS,
# FFN_HIDDEN_SIZE, MAMBA_NUM_HEADS (empty → flag omitted, framework picks
# from hidden_size), HYBRID_OVERRIDE_PATTERN.  MoE presets additionally
# fix NUM_EXPERTS, MOE_SHARED_EXPERT_INTERMEDIATE_SIZE, and MOE_LATENT_SIZE
# (empty → flag omitted).  Override any individual dim by setting the
# matching env var.
#
# Default is `smoke` so a one-line `NODES=1 bash submit.sh` actually
# fits 4 GPUs and finishes.  Pick nano/super/ultra explicitly for the
# MoE benchmarks at the matching node count.
# ---------------------------------------------------------------------------
: "${MODEL_SIZE:=smoke}"
case ${MODEL_SIZE} in
    smoke)
        # Plain dense transformer (RoPE, GQA, no Mamba, no MoE).  Sized
        # to fit one GB200 node at TP=4 PP=1 DP=1.  Validates the
        # cluster / container / NCCL / ckpt plumbing without exercising
        # the MoE / Mamba paths that nano/super/ultra benchmark.  Smoke
        # ckpts skip optim + RNG (--no-save-optim, --no-save-rng) so the
        # save cost is *not* representative of what we'll benchmark on
        # the real presets — smoke is a "does it run" gate, not a "how
        # fast" measurement.
        IS_MOE=0
        TP=4
        EP=1
        ETP=1
        PP=1
        : "${NUM_LAYERS:=32}"
        : "${HIDDEN_SIZE:=4096}"
        : "${NUM_ATTENTION_HEADS:=32}"
        : "${FFN_HIDDEN_SIZE:=21504}"
        : "${MAMBA_NUM_HEADS:=}"
        : "${GLOBAL_BATCH_SIZE:=16}"
        : "${MICRO_BATCH_SIZE:=1}"
        ;;
    nano)
        IS_MOE=1
        # Matches Deepak's nemotron3/nano.sh: TP=2, EP=32, alltoall
        # dispatcher (HYBRIDEP=0).  GBS/MBS are set per PRODUCTION_BATCH
        # below — Deepak's GBS=3072 / MBS=1 is sized for 384 nodes and
        # is far too large for our 8–128 node scaling experiments.
        : "${TP:=2}"
        : "${EP:=32}"
        : "${PP:=1}"
        : "${ETP:=1}"
        : "${HYBRIDEP:=0}"  # alltoall per nano.sh; overrides the
                            # IS_MOE-default HYBRIDEP=1 below.
        : "${NUM_LAYERS:=52}"
        : "${HIDDEN_SIZE:=2688}"
        : "${NUM_ATTENTION_HEADS:=32}"
        : "${FFN_HIDDEN_SIZE:=1856}"
        : "${MAMBA_NUM_HEADS:=64}"
        : "${NUM_EXPERTS:=128}"
        : "${MOE_SHARED_EXPERT_INTERMEDIATE_SIZE:=3712}"
        : "${MOE_LATENT_SIZE:=}"  # N/A on Nano
        : "${HYBRID_OVERRIDE_PATTERN:=MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME}"
        ;;
    super)
        IS_MOE=1
        # Matches Deepak's nemotron3/super.sh: TP=2, EP=64, flex
        # dispatcher (HYBRIDEP=1 → flex+hybridep backend per the 12B
        # phase1 production log).  Pattern verbatim from super.sh
        # including the trailing /*E/*E.  GBS/MBS gated by
        # PRODUCTION_BATCH below.
        : "${TP:=2}"
        : "${EP:=64}"
        : "${PP:=1}"
        : "${ETP:=1}"
        : "${NUM_LAYERS:=88}"
        : "${HIDDEN_SIZE:=4096}"
        : "${NUM_ATTENTION_HEADS:=32}"
        : "${FFN_HIDDEN_SIZE:=2688}"
        : "${MAMBA_NUM_HEADS:=128}"
        : "${NUM_EXPERTS:=512}"
        : "${MOE_SHARED_EXPERT_INTERMEDIATE_SIZE:=5376}"
        : "${MOE_LATENT_SIZE:=1024}"
        : "${HYBRID_OVERRIDE_PATTERN:=MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME/*E/*E}"
        ;;
    ultra)
        IS_MOE=1
        # Matches Deepak's nemotron3/ultra.sh: TP=8, EP=64, flex
        # dispatcher (HYBRIDEP=1 → flex+hybridep backend per the 55B
        # phase2_20t production log).  Pattern verbatim from ultra.sh
        # including the trailing /*E/*E.  GBS/MBS gated by
        # PRODUCTION_BATCH below (ultra's production MBS=2 only
        # applies when PRODUCTION_BATCH=1).
        : "${TP:=8}"
        : "${EP:=64}"
        : "${PP:=1}"
        : "${ETP:=1}"
        : "${NUM_LAYERS:=108}"
        : "${HIDDEN_SIZE:=8192}"
        : "${NUM_ATTENTION_HEADS:=64}"
        : "${FFN_HIDDEN_SIZE:=5120}"
        : "${MAMBA_NUM_HEADS:=256}"
        : "${NUM_EXPERTS:=512}"
        : "${MOE_SHARED_EXPERT_INTERMEDIATE_SIZE:=10240}"
        : "${MOE_LATENT_SIZE:=2048}"
        : "${HYBRID_OVERRIDE_PATTERN:=MEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME/*E/*E}"
        ;;
    test)
        # 14-layer testbed at Ultra MoE dims.  Does not pass
        # --mamba-num-heads (lets the framework compute it from
        # hidden_size).  TP/EP match ultra since the per-expert
        # tensor shapes are the same.
        IS_MOE=1
        : "${TP:=8}"
        : "${EP:=64}"
        : "${PP:=1}"
        : "${ETP:=1}"
        : "${NUM_LAYERS:=14}"
        : "${HIDDEN_SIZE:=8192}"
        : "${NUM_ATTENTION_HEADS:=64}"
        : "${FFN_HIDDEN_SIZE:=5120}"
        : "${MAMBA_NUM_HEADS:=}"
        : "${NUM_EXPERTS:=512}"
        : "${MOE_SHARED_EXPERT_INTERMEDIATE_SIZE:=10240}"
        : "${MOE_LATENT_SIZE:=2048}"
        : "${HYBRID_OVERRIDE_PATTERN:=MEMEMEM*EMEMEM}"
        ;;
    *)
        echo "ERROR: unknown MODEL_SIZE=${MODEL_SIZE}.  Supported: smoke, nano, super, ultra, test." >&2
        exit 1
        ;;
esac

# MXFP8 and HYBRIDEP only apply to MoE presets.  User-set values still
# win via `:=`.
#
# MXFP8 default is 0 (bf16) to match Deepak's nemotron3 reference
# recipes -- those run pure bf16 (ultra additionally enables FP4 weight
# storage via --fp4-format e2m1, which we don't currently support).
# Pass MXFP8=1 on the submit.sh command line to flip to the mxfp8
# recipe (--fp8-recipe mxfp8 --fp8-format e4m3 --fp8-param-gather etc.)
# when we want to benchmark fp8 checkpoint cost.
#
# HYBRIDEP default is 1 (flex+hybridep dispatcher, used by super/ultra
# in production).  Nano overrides this to 0 (alltoall) in its own arm
# above -- matches Deepak's nano.sh.
if [[ ${IS_MOE} -eq 1 ]]; then
    : "${MXFP8:=0}"
    : "${HYBRIDEP:=1}"
else
    : "${MXFP8:=0}"
    : "${HYBRIDEP:=0}"
fi

# alltoall dispatcher (HYBRIDEP=0) + mxfp8 (MXFP8=1) is a known-broken
# combo on this stack: TE's router.backward fails because the
# routing_map bool tensor (saved by forward via ctx.save_for_backward)
# is mutated in place by --moe-router-padding-for-quantization (which
# we add when MXFP8=1).  Symptom is "one of the variables needed for
# gradient computation has been modified by an inplace operation:
# [CUDABoolType [TOKENS, EXPERTS]] is at version 1; expected version 0"
# at iteration 1's backward pass.
#
# Nano defaults HYBRIDEP=0 to match Deepak's nano.sh, but that recipe runs
# bf16-only so the combo never
# arose in production.  Auto-flip to HYBRIDEP=1 (flex+hybridep) when
# MXFP8=1; emit a stderr notice so the deviation is visible in the
# step2 log.  User-set HYBRIDEP=0 + MXFP8=1 still gets the flip --
# the alternative is a confusing TE traceback on iter 1.
if [[ ${IS_MOE} -eq 1 && "${MXFP8}" -eq 1 && "${HYBRIDEP}" -eq 0 ]]; then
    echo "NOTICE: alltoall dispatcher (HYBRIDEP=0) is incompatible with mxfp8" >&2
    echo "        (TE router backward fails on routing_map in-place mutation)." >&2
    echo "        Auto-flipping HYBRIDEP=1 to use flex+hybridep dispatcher." >&2
    HYBRIDEP=1
fi

# MoE router top-k and its scaling factor per the Nemotron scaling ladder:
# Nano=6, Super/Ultra=22 (test reuses Ultra dims).  Super/Ultra production
# (12b/55b phase logs) use scaling factor 5.0; Nano has no production
# reference, so it keeps the prior 2.5.  User-set values still win via `:=`.
if [[ ${IS_MOE} -eq 1 ]]; then
    case "${MODEL_SIZE}" in
        nano)
            : "${MOE_ROUTER_TOPK:=6}"
            : "${MOE_ROUTER_TOPK_SCALING_FACTOR:=2.5}"
            ;;
        super|ultra|test)
            : "${MOE_ROUTER_TOPK:=22}"
            : "${MOE_ROUTER_TOPK_SCALING_FACTOR:=5.0}"
            ;;
    esac
fi

WORLD_SIZE=$(( NODES * GPUS_PER_NODE ))

# Batch-size defaults.  Two regimes, gated by PRODUCTION_BATCH:
#
#   PRODUCTION_BATCH=0 (default — testing scale)
#       GBS = WORLD_SIZE, MBS = 1.  Keeps each iteration short (~2
#       microbatches per DP per iter typical) so checkpoint-cost
#       experiments at 8–128 nodes finish meaningful numbers of iters
#       within reasonable time budgets.
#
#   PRODUCTION_BATCH=1 (Deepak's nemotron3 reference recipes)
#       GBS = 3072 for all MoE presets.  MBS=2 for ultra; MBS=1
#       otherwise.  Designed for 384+ node production runs.  At
#       smaller rungs the per-iter microbatch count balloons
#       proportionally (e.g. nano @ 8 nodes hits 192 microbatches
#       per DP per iter — ~85 s per iter).  Use only when you
#       actually want production-shape iteration cost.
#
# Either GLOBAL_BATCH_SIZE or MICRO_BATCH_SIZE passed on the submit.sh
# command line still wins via `:=`.
: "${PRODUCTION_BATCH:=0}"
if [[ ${IS_MOE} -eq 1 && "${PRODUCTION_BATCH}" -eq 1 ]]; then
    : "${GLOBAL_BATCH_SIZE:=3072}"
    if [[ "${MODEL_SIZE}" == "ultra" ]]; then
        : "${MICRO_BATCH_SIZE:=2}"
    fi
fi
: "${MICRO_BATCH_SIZE:=1}"
: "${GLOBAL_BATCH_SIZE:=${WORLD_SIZE}}"

set -e
set -o pipefail

. "${SLURM_SUBMIT_DIR}/clusters/oci-hsg/env.sh"

# HybridEP rank-per-NVLink-domain count == EP.  Must be exported into the
# container so the hybridep dispatcher sees it at MoE init time.
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=${EP}

DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')
SLURM_JOB_DIR=${REPO_ROOT}/data/${SLURM_JOB_NAME}_${SLURM_JOBID}_${DATETIME}

WORK_DIR=${SLURM_JOB_DIR}/output
LOG_DIR=${SLURM_JOB_DIR}/logs
DATACACHE_DIR=${SLURM_JOB_DIR}/datacache
CHECKPOINT_DIR=${SLURM_JOB_DIR}/checkpoints
TENSORBOARD_DIR=${SLURM_JOB_DIR}/tensorboard

mkdir -p "${WORK_DIR}" "${LOG_DIR}" "${DATACACHE_DIR}" "${CHECKPOINT_DIR}" "${TENSORBOARD_DIR}"

# ---------------------------------------------------------------------------
# Lustre striping for the checkpoint directory.  Not on the training-
# thread critical path, but keeps the layout reproducible across runs
# and pins it independent of whatever Lustre defaults are inherited.
# ---------------------------------------------------------------------------
: "${LUSTRE_STRIPE_LAYOUT:=-E 1G -c 1 -S 1M -E 128G -c 8 -S 4M -E -1 -c -1 -S 4M}"
if command -v lfs >/dev/null 2>&1; then
    if lfs setstripe ${LUSTRE_STRIPE_LAYOUT} "${CHECKPOINT_DIR}" 2>/dev/null; then
        echo "Lustre stripe set on ${CHECKPOINT_DIR}: ${LUSTRE_STRIPE_LAYOUT}"
    else
        echo "WARNING: lfs setstripe failed on ${CHECKPOINT_DIR} — directory may not be on Lustre, or has files with a conflicting layout.  Continuing with whatever layout was inherited."
    fi
else
    echo "WARNING: lfs not on PATH; skipping Lustre stripe setup."
fi

export TRITON_CACHE_DIR=${WORK_DIR}/triton_cache
# Do NOT set TRITON_CACHE_MANAGER to Megatron's
# `megatron.core.ssm.triton_cache_manager:ParallelFileCacheManager` —
# that patch asserts triton_version <= 3.1 and the container ships a
# newer Triton (its upstream cache manager has the parallel-friendly fix
# folded in already).

CHECKPOINT_INTERVAL=100
SEQ_LEN=8192

# ---------------------------------------------------------------------------
# Model / training options.  Two completely separate option blocks:
#
#   smoke (IS_MOE=0) — plain dense transformer (RoPE, GQA, no Mamba, no
#     MoE).  Recipe from a colleague's single-node debug config.  Adapted
#     for mock-data + NullTokenizer so it doesn't depend on a specific
#     filesystem mount.  Skips optim + RNG on save (--no-save-optim,
#     --no-save-rng) — smoke ckpts are cheaper than real ckpts; do not
#     read perf numbers off them.
#
#   nano/super/ultra/test (IS_MOE=1) — hybrid Mamba MoE with MXFP8,
#     HybridEP, precision-aware optimizer, fully-sharded FSDP.  The
#     actual benchmark surface.
# ---------------------------------------------------------------------------
if [[ ${IS_MOE} -eq 0 ]]; then
    options=" \
    --exit-duration-in-mins 12 \
    --distributed-timeout-minutes 10 \
    --no-create-attention-mask-in-dataloader \
    --no-mmap-bin-files \
    --train-samples 3200 \
    --lr-warmup-samples 500 \
    --lr-decay-samples 3200 \
    --lr-decay-style WSD \
    --lr-wsd-decay-style minus_sqrt \
    --lr-wsd-decay-samples 2000 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --use-mcore-models \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --init-method-std 0.014 \
    --bf16 \
    --sequence-parallel \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --ddp-num-buckets 6 \
    --attention-backend flash \
    --log-interval 1 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --log-throughput \
    --log-timers-to-tensorboard \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --tensorboard-log-interval 1 \
    --timing-log-option minmax \
    --eval-interval 10000 \
    --eval-iters 0 \
    --manual-gc \
    --manual-gc-interval 10 \
    --num-workers 1 \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --num-attention-heads ${NUM_ATTENTION_HEADS} \
    --group-query-attention \
    --num-query-groups 8 \
    --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
    --kv-channels 128 \
    --normalization RMSNorm \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --squared-relu \
    --position-embedding-type rope \
    --rotary-base 1000000 \
    --rotary-percent 1.0 \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${SEQ_LEN} \
    --lr 8e-4 \
    --min-lr 8e-6 \
    "
else
    # EXIT_INTERVAL bounds the run by iter-count, independent of LR
    # horizon.  We bump --train-samples to a value that always exceeds
    # EXIT_INTERVAL * GBS so Megatron doesn't cut the run short on the
    # train-iters ceiling (which Deepak's 40000 hit at iter 312 with
    # GBS=128).  LR plateaus at min_lr past lr-decay-samples — fine for
    # a checkpoint-overhead measurement.
    : "${EXIT_INTERVAL:=800}"
    TRAIN_SAMPLES=$(( EXIT_INTERVAL * GLOBAL_BATCH_SIZE * 2 ))
    options=" \
    --exit-duration-in-mins 5750 \
    --exit-interval ${EXIT_INTERVAL} \
    --distributed-timeout-minutes 120 \
    --no-create-attention-mask-in-dataloader \
    --no-mmap-bin-files \
    --train-samples ${TRAIN_SAMPLES} \
    --lr-warmup-samples 2000 \
    --lr-decay-samples 38000 \
    --lr-decay-style WSD \
    --lr-wsd-decay-style minus_sqrt \
    --lr-wsd-decay-samples 40000 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --use-mcore-models \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --transformer-impl transformer_engine \
    --init-method-std 0.0099 \
    --bf16 \
    --sequence-parallel \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --ddp-num-buckets 24 \
    --grad-reduce-in-bf16 \
    --log-interval 1 \
    --log-params-norm \
    --log-throughput \
    --log-timers-to-tensorboard \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --tensorboard-log-interval 1 \
    --eval-iters 7 \
    --eval-interval 2000 \
    --eval-global-batch-size 256 \
    --manual-gc \
    --manual-gc-interval 100 \
    --num-workers 2 \
    --disable-gloo-process-groups \
    --is-hybrid-model \
    --hybrid-layer-pattern ${HYBRID_OVERRIDE_PATTERN} \
    --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --num-attention-heads ${NUM_ATTENTION_HEADS} \
    --group-query-attention \
    --num-query-groups 8 \
    --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
    --kv-channels 128 \
    --normalization RMSNorm \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --squared-relu \
    --cross-entropy-loss-fusion \
    --cross-entropy-fusion-impl native \
    --te-rng-tracker \
    --no-gradient-accumulation-fusion \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${SEQ_LEN} \
    --lr 8.0e-4 \
    --min-lr 8.0e-6 \
    --position-embedding-type none \
    --expert-model-parallel-size ${EP} \
    --expert-tensor-parallel-size ${ETP} \
    --high-priority-stream-groups ep \
    --use-fused-weighted-squared-relu \
    --use-precision-aware-optimizer \
    --main-params-dtype fp32 \
    --main-grads-dtype bf16 \
    --exp-avg-dtype bf16 \
    --exp-avg-sq-dtype bf16 \
    --megatron-fsdp-main-params-dtype fp32 \
    --megatron-fsdp-main-grads-dtype bf16"
fi

# --mamba-num-heads is a Mamba-block knob (applies to dense and MoE).
# Empty preset value → flag omitted, framework picks from hidden_size.
if [[ -n "${MAMBA_NUM_HEADS}" ]]; then
    options="${options} --mamba-num-heads ${MAMBA_NUM_HEADS}"
fi

# NullTokenizer + a synthetic vocab — no tokenizer file needed under
# --mock-data.  Override TOKENIZER_MODEL in env.sh and flip the type if
# you ever switch to real data.
tokenizer_options=" \
    --tokenizer-type NullTokenizer \
    --vocab-size 131072"

data_options=" \
    --mock-data \
    --data-cache-path ${DATACACHE_DIR}"

# ---------------------------------------------------------------------------
# MoE + MXFP8 options.  Whole block is skipped for dense (smoke) presets.
# Routing: sigmoid + seq_aux_loss + top-k=5; dispatcher picks alltoall
# vs flex/hybridep based on HYBRIDEP.  MXFP8 (Blackwell-only) upcasts
# the checkpoint payload back to BF16 on save, so the ckpt size we
# benchmark matches the BF16 model regardless.
# ---------------------------------------------------------------------------
moe_options=""
mxfp8_options=""
if [[ ${IS_MOE} -eq 1 ]]; then
    moe_options=" \
    --num-experts ${NUM_EXPERTS} \
    --moe-router-topk ${MOE_ROUTER_TOPK} \
    --moe-router-score-function sigmoid \
    --moe-aux-loss-coeff 1e-4 \
    --moe-router-topk-scaling-factor ${MOE_ROUTER_TOPK_SCALING_FACTOR} \
    --moe-router-enable-expert-bias \
    --moe-router-dtype fp32 \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-shared-expert-intermediate-size ${MOE_SHARED_EXPERT_INTERMEDIATE_SIZE} \
    --moe-permute-fusion \
    --moe-router-fusion \
    --moe-router-force-load-balancing \
    --moe-grouped-gemm"

    # --moe-latent-size unset on Nano (ladder lists N/A); pass only when
    # the preset / env-var provides a value.
    if [[ -n "${MOE_LATENT_SIZE}" ]]; then
        moe_options="${moe_options} --moe-latent-size ${MOE_LATENT_SIZE}"
    fi

    if [[ ${HYBRIDEP} -eq 1 ]]; then
        moe_options="${moe_options} \
    --moe-token-dispatcher-type flex \
    --moe-flex-dispatcher-backend hybridep \
    --moe-hybridep-num-sms 32"
    else
        moe_options="${moe_options} --moe-token-dispatcher-type alltoall"
    fi

    if [[ ${MXFP8} -eq 1 ]]; then
        mxfp8_options=" \
    --fp8-recipe mxfp8 \
    --fp8-format e4m3 \
    --fp8-param-gather \
    --moe-router-padding-for-quantization \
    --reuse-grad-buf-for-mxfp8-param-ag"
    fi
fi

# ---------------------------------------------------------------------------
# Checkpoint strategy.  This is the dimension we're benchmarking across.
#
# Common flags (apply to every strategy):
#   --save / --load / --save-interval                              — where + cadence
#   --ckpt-fully-parallel-save / --ckpt-fully-parallel-load        — DP sharding
#   --ckpt-fully-parallel-save-process-group ep_dp                 — sharding scope
#
# Strategy-specific flags layer on top.  The four strategies all share
# torch_dist's underlying state-dict representation except fsdp-dtensor,
# which switches to FSDP's native sharded checkpointer.
#
# Note on instrumentation: the perfetto spans in
# launchers/ckpt_perfetto_trace.py patch NVRX module functions.  For
# strategies other than async-nvrx those patches sit idle (no-op), so
# PERFETTO=1 will still emit the outer Megatron spans
# (save_checkpoint_and_time, assemble state dict, the polling region) and
# the analyzer will surface the wall-clock blocking window — but the
# per-phase NVRX breakdown (D2H, persist, write buckets) only populates
# for async-nvrx.  That's expected; the comparison story is "wall-clock
# blocking + steady-state throughput", not "phase breakdown".
# ---------------------------------------------------------------------------
ckpt_options=" \
    --save ${CHECKPOINT_DIR} \
    --load ${CHECKPOINT_DIR} \
    --save-interval ${CHECKPOINT_INTERVAL} \
    --ckpt-fully-parallel-save \
    --ckpt-fully-parallel-load"

# Process group for fully-parallel ckpt sharding: `ep_dp` parallelizes
# across both expert and data ranks for MoE, but the `ep_dp` group only
# exists when EP is configured.  Dense (smoke) runs use the default `dp`
# and skip optim + RNG on save to keep the smoke ckpt tiny.
if [[ ${IS_MOE} -eq 1 ]]; then
    ckpt_options="${ckpt_options} \
    --ckpt-fully-parallel-save-process-group ep_dp \
    --ckpt-fully-parallel-load-process-group ep_dp"
else
    ckpt_options="${ckpt_options} --no-save-optim --no-save-rng"
fi

# --async-ckpt-use-cpu-shm requires --ckpt-assume-constant-structure
# (Megatron PR #4355): the SHM tensors are allocated once per save shape
# and reused across checkpoints; without the constant-structure flag the
# per-tensor /dev/shm allocations would dominate cost.  All four async
# strategies below already pass constant-structure, so the SHM variants
# just layer the extra flag on top.
case ${CKPT_STRATEGY} in
    async-nvrx)
        ckpt_options="${ckpt_options} \
    --ckpt-format torch_dist \
    --ckpt-assume-constant-structure \
    --async-save \
    --async-strategy nvrx \
    --use-persistent-ckpt-worker"
        ;;
    async-nvrx-shm)
        ckpt_options="${ckpt_options} \
    --ckpt-format torch_dist \
    --ckpt-assume-constant-structure \
    --async-save \
    --async-strategy nvrx \
    --use-persistent-ckpt-worker \
    --async-ckpt-use-cpu-shm"
        ;;
    async-mcore)
        ckpt_options="${ckpt_options} \
    --ckpt-format torch_dist \
    --ckpt-assume-constant-structure \
    --async-save \
    --async-strategy mcore"
        ;;
    async-mcore-shm)
        ckpt_options="${ckpt_options} \
    --ckpt-format torch_dist \
    --ckpt-assume-constant-structure \
    --async-save \
    --async-strategy mcore \
    --async-ckpt-use-cpu-shm"
        ;;
    sync-torch-dist)
        ckpt_options="${ckpt_options} --ckpt-format torch_dist"
        ;;
    fsdp-dtensor)
        ckpt_options="${ckpt_options} --ckpt-format fsdp_dtensor"
        ;;
    *)
        echo "ERROR: unknown CKPT_STRATEGY=${CKPT_STRATEGY}." >&2
        echo "  Supported: async-nvrx, async-nvrx-shm, async-mcore," >&2
        echo "             async-mcore-shm, sync-torch-dist, fsdp-dtensor" >&2
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# Perfetto + pinned-pool warmup.  Wrappers compose via runpy: the warmup
# wrapper (if WARMUP_PINNED_POOL=1) wraps the perfetto wrapper (if
# PERFETTO=1) wraps pretrain_mamba.py.
# ---------------------------------------------------------------------------
# Pick the right pretrain entry: pretrain_mamba.py for hybrid Mamba+MoE
# (it requires --is-hybrid-model + --spec mamba_layer_specs); pretrain_
# gpt.py for the plain-transformer smoke recipe.
if [[ ${IS_MOE} -eq 1 ]]; then
    PRETRAIN_ENTRY="${MEGATRON_DIR}/pretrain_mamba.py"
else
    PRETRAIN_ENTRY="${MEGATRON_DIR}/pretrain_gpt.py"
fi

# state_dict_dump.py must sit *inside* the perfetto wrapper so the
# perfetto "assemble state dict" span measures only generate_state_dict
# itself, not the torch.save that happens after.  Put it on the chain
# before the perfetto wrap.
if [[ ${STATE_DICT_DUMP:-0} -eq 1 ]]; then
    export STATE_DICT_DUMP=1
    : "${STATE_DICT_DUMP_PATH:=${LOG_DIR}/state_dict_rank0_ckpt0.json}"
    export STATE_DICT_DUMP_PATH
    PRETRAIN_ENTRY="${SLURM_SUBMIT_DIR}/launchers/state_dict_dump.py ${PRETRAIN_ENTRY}"
fi

if [[ ${PERFETTO} -eq 1 ]]; then
    export CKPT_PERFETTO_TRACE=1
    export CKPT_PERFETTO_OUT="${LOG_DIR}"
    : "${CKPT_PERFETTO_RANKS:=all}"
    export CKPT_PERFETTO_RANKS
    PRETRAIN_ENTRY="${SLURM_SUBMIT_DIR}/launchers/pretrain_mamba_perfetto.py ${PRETRAIN_ENTRY}"
fi

# gc_cycle_probe.py wraps save_checkpoint to detect whether the fp8
# dequant temporaries are held by reference cycles.  Goes outside the
# perfetto wrapper so its gc.collect() stays out of the inner save spans.
if [[ ${GC_CYCLE_PROBE:-0} -eq 1 ]]; then
    export GC_CYCLE_PROBE=1
    : "${GC_CYCLE_PROBE_MAX:=3}"
    : "${GC_CYCLE_PROBE_TOPN:=20}"
    : "${GC_CYCLE_PROBE_TRACE_DEPTH:=30}"
    : "${GC_CYCLE_PROBE_TRACE_VISITS:=1000}"
    export GC_CYCLE_PROBE_MAX GC_CYCLE_PROBE_TOPN \
        GC_CYCLE_PROBE_TRACE_DEPTH GC_CYCLE_PROBE_TRACE_VISITS
    PRETRAIN_ENTRY="${SLURM_SUBMIT_DIR}/launchers/gc_cycle_probe.py ${PRETRAIN_ENTRY}"
fi

case "${WARMUP_PINNED_POOL:-0}" in
    0|"")
        ;;
    1|true|full|plan-only|plan_only|planonly)
        export WARMUP_PINNED_POOL
        PRETRAIN_ENTRY="${SLURM_SUBMIT_DIR}/launchers/pinned_pool_warmup.py ${PRETRAIN_ENTRY}"
        ;;
    *)
        echo "WARN: unknown WARMUP_PINNED_POOL=${WARMUP_PINNED_POOL}; treating as 0." >&2
        ;;
esac

# GLOO_EAGER_INIT goes on the *outside* of the wrapper chain so its
# monkey-patch on torch.distributed.init_process_group is installed
# before any inner wrapper imports torch.distributed.  Requires PyTorch
# >= 2.3 (the wrapper falls back to a no-op pass-through otherwise and
# logs a warning).
: "${GLOO_EAGER_INIT:=0}"
export GLOO_EAGER_INIT
case "${GLOO_EAGER_INIT}" in
    1|true|yes)
        PRETRAIN_ENTRY="${SLURM_SUBMIT_DIR}/launchers/gloo_eager_init.py ${PRETRAIN_ENTRY}"
        ;;
esac

# WARMUP_PROBE_GATHER is read inside pinned_pool_warmup._probe_gather_object
# at warmup time; just export it so srun's child processes see it.
: "${WARMUP_PROBE_GATHER:=0}"
export WARMUP_PROBE_GATHER

run_cmd="${PRETRAIN_ENTRY} \
    ${options} \
    ${data_options} \
    ${tokenizer_options} \
    ${moe_options} \
    ${mxfp8_options} \
    ${ckpt_options}"

# ---------------------------------------------------------------------------
# Echo the resolved knobs so the slurm log carries the full reproducer.
# ---------------------------------------------------------------------------
echo "==== oci-hsg/launch.sh resolved config ===="
echo "  NODES=${NODES}  SEGMENT=${SEGMENT}  WORLD_SIZE=${WORLD_SIZE}"
echo "  MODEL_SIZE=${MODEL_SIZE}  IS_MOE=${IS_MOE}"
echo "  TP=${TP}  EP=${EP}  PP=${PP}  ETP=${ETP}"
echo "  NUM_LAYERS=${NUM_LAYERS}  HIDDEN_SIZE=${HIDDEN_SIZE}  FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE}"
echo "  NUM_EXPERTS=${NUM_EXPERTS:-<dense>}  MAMBA_NUM_HEADS=${MAMBA_NUM_HEADS:-<auto>}  MOE_LATENT_SIZE=${MOE_LATENT_SIZE:-<unset>}"
echo "  GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}  MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE}  PRODUCTION_BATCH=${PRODUCTION_BATCH}"
echo "  EXIT_INTERVAL=${EXIT_INTERVAL:-<dense:n/a>}  TRAIN_SAMPLES=${TRAIN_SAMPLES:-<dense:n/a>}"
echo "  CKPT_STRATEGY=${CKPT_STRATEGY}  MXFP8=${MXFP8}  HYBRIDEP=${HYBRIDEP}"
echo "  PERFETTO=${PERFETTO}  WARMUP_PINNED_POOL=${WARMUP_PINNED_POOL}  WARMUP_PROBE_GATHER=${WARMUP_PROBE_GATHER}  GLOO_EAGER_INIT=${GLOO_EAGER_INIT}  SKIP_EMPTY_CACHE_IN_SAVE=${SKIP_EMPTY_CACHE_IN_SAVE:-0}"
echo "  GC_CYCLE_PROBE=${GC_CYCLE_PROBE:-0}  STATE_DICT_DUMP=${STATE_DICT_DUMP:-0}"
echo "  NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=${NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN}"
echo "==========================================="

# ---------------------------------------------------------------------------
# In-job container prep is GONE — we now use a pre-built derived image
# (clusters/oci-hsg/build-image.sh produces images/nemotron-moe-nvrx.sqsh
# with our pinned NVRX baked in) so every node just loads the ready-to-
# go sqsh.  No more per-node pip-install dance, no more PyPI flake on
# multi-node submissions, no more concurrent-job races.  env.sh picks
# the derived image if it exists and warns if you forgot to build it.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Step: run training.  bindpcie sits between srun and the training
# command so each rank gets pinned to the NUMA node closest to its GPU
# (and its CPU memory to the same NUMA, so cudaHostAlloc / pinned-memory
# allocations for D2H land near the assigned GPU).
#
# Defaults: --cpu=node --mem=node --ib=off.  Override via BINDPCIE_ARGS
# to add e.g. --ib=single for IB-device-nearest-GPU binding once NCCL
# perf is in scope.
# ---------------------------------------------------------------------------
BINDPCIE="${SLURM_SUBMIT_DIR}/clusters/oci-hsg/bindpcie"
: "${BINDPCIE_ARGS:=}"

srun -l \
    --container-image "${CONTAINER_IMAGE}" \
    --container-name "node${SLURM_NODEID}" \
    --container-mounts "/lustre:/lustre" \
    --output="${LOG_DIR}/%x_%j_${DATETIME}_step2.log" \
    "${BINDPCIE}" ${BINDPCIE_ARGS} -- bash -c "python -u ${run_cmd}"

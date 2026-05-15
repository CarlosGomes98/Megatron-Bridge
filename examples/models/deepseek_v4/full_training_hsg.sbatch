#!/bin/bash
# One-iteration DeepSeek-V4-Flash full-config training smoke on HSG.
# This uses the real HF config, Adam, and the OCI C4 data directory. Set LOAD_CHECKPOINT to
# initialize from an already-converted Megatron-Core checkpoint instead of random init.

#SBATCH --job-name=coreai_dlalgo_llm-mbridge:dsv4-full-adam
#SBATCH --account=coreai_mlperf_training
#SBATCH --partition=batch
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fsw/portfolios/coreai/projects/coreai_mlperf_training/users/cgomes/dsv4/logs/dsv4_full_adam_%j.log
#SBATCH --exclusive

set -euo pipefail

CONTAINER_IMAGE=${CONTAINER_IMAGE:-/lustre/fsw/portfolios/coreai/users/weijiac/sqsh/nemo_26.04.sqsh}
WORKDIR=${WORKDIR:-/lustre/fsw/portfolios/coreai/projects/coreai_mlperf_training/users/cgomes/Megatron-Bridge}
MCORE_DIR=${MCORE_DIR:-/lustre/fsw/portfolios/coreai/users/chcui/repos/Megatron-LM-pr4518}
FHT_SITE=${FHT_SITE:-/lustre/fsw/portfolios/coreai/users/chcui/parity/dsv4_cudnn_dsa/fast_hadamard_site}
CUDNN_FE_SITE=${CUDNN_FE_SITE:-/lustre/fsw/portfolios/coreai/users/chcui/parity/dsv4_cudnn_dsa/cudnn_fe_site}
TRANSFORMERS_SITE=${TRANSFORMERS_SITE:-/lustre/fsw/portfolios/coreai/projects/coreai_mlperf_training/users/cgomes/dsv4_transformers_5_8_1_nodeps}
HF_CONFIG=${HF_CONFIG:-/lustre/fsw/portfolios/coreai/users/weijiac/models/deepseek-ai/DeepSeek-V4-Flash}
OUTDIR=${OUTDIR:-/lustre/fsw/portfolios/coreai/projects/coreai_mlperf_training/users/cgomes/dsv4/logs/dsv4_full_adam}
LOGDIR=${LOGDIR:-/lustre/fsw/portfolios/coreai/projects/coreai_mlperf_training/users/cgomes/dsv4/logs}
PY_DEPS=${PY_DEPS:-$OUTDIR/python_deps}
LOAD_CHECKPOINT=${LOAD_CHECKPOINT:-}
DATADIR=${DATADIR:-/lustre/fsw/portfolios/coreai/projects/coreai_mlperf_training/data/c4}
DATASET=${DATASET:-8b}
PREPROC_DATA_DIR=${PREPROC_DATA_DIR:-$DATADIR/$DATASET}
TOKENIZER_DIR=${TOKENIZER_DIR:-$PREPROC_DATA_DIR/tokenizer}
NPY_INDEX_DIR=${NPY_INDEX_DIR:-$OUTDIR/npy_index}
CSA_BACKEND=${CSA_BACKEND:-unfused}
MOE_TOKEN_DISPATCHER_TYPE=${MOE_TOKEN_DISPATCHER_TYPE:-alltoall}
ADAM_MAX_LR=${ADAM_MAX_LR:-1e-4}
ADAM_MIN_LR=${ADAM_MIN_LR:-1e-5}
ADAM_WEIGHT_DECAY=${ADAM_WEIGHT_DECAY:-0.01}
EXTRA_ARGS=${EXTRA_ARGS:-}
NCCL_PXN_DISABLE=${NCCL_PXN_DISABLE:-1}

SEQ_LENGTH=${SEQ_LENGTH:-128}
TRAIN_ITERS=${TRAIN_ITERS:-1}
EVAL_INTERVAL=${EVAL_INTERVAL:-$TRAIN_ITERS}
EVAL_ITERS=${EVAL_ITERS:-1}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
TP=${TP:-1}
PP=${PP:-4}
CP=${CP:-1}
EP=${EP:-8}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MASTER_PORT=${MASTER_PORT:-29571}

mkdir -p "$LOGDIR" "$OUTDIR" "$NPY_INDEX_DIR"

CONTAINER_MOUNTS="/lustre:/lustre,${PREPROC_DATA_DIR}:/preproc_data:ro,${TOKENIZER_DIR}:/tokenizer:ro,${NPY_INDEX_DIR}:/npy_index"

MASTER_ADDR=$(python3 - <<'PY'
import os
import re

s = os.environ.get("SLURM_NODELIST", "")
m = re.match(r"([\w-]+)\[(\d+)", s)
print(m.group(1) + m.group(2) if m else s.split(",")[0])
PY
)

echo "DeepSeek-V4 full Adam training smoke | job=$SLURM_JOB_ID"
echo "nodes=$SLURM_JOB_NUM_NODES nproc_per_node=$NPROC_PER_NODE master=$MASTER_ADDR:$MASTER_PORT"
echo "parallelism TP=$TP PP=$PP CP=$CP EP=$EP"
echo "seq_length=$SEQ_LENGTH train_iters=$TRAIN_ITERS gbs=$GLOBAL_BATCH_SIZE mbs=$MICRO_BATCH_SIZE"
echo "eval_interval=$EVAL_INTERVAL eval_iters=$EVAL_ITERS"
echo "workdir=$WORKDIR"
echo "mcore=$MCORE_DIR"
echo "fht_site=$FHT_SITE"
echo "cudnn_fe_site=$CUDNN_FE_SITE"
echo "hf_config=$HF_CONFIG"
echo "outdir=$OUTDIR"
echo "preproc_data_dir=$PREPROC_DATA_DIR"
echo "tokenizer_dir=$TOKENIZER_DIR"
echo "npy_index_dir=$NPY_INDEX_DIR"
echo "load_checkpoint=${LOAD_CHECKPOINT:-<none>}"
echo "transformers_site=$TRANSFORMERS_SITE"
echo "py_deps=$PY_DEPS"
echo "csa_backend=$CSA_BACKEND"
echo "moe_token_dispatcher_type=$MOE_TOKEN_DISPATCHER_TYPE"
echo "adam_lr=$ADAM_MAX_LR:$ADAM_MIN_LR adam_weight_decay=$ADAM_WEIGHT_DECAY"
echo "extra_args=$EXTRA_ARGS"
echo "nccl_pxn_disable=$NCCL_PXN_DISABLE"

srun --mpi=pmix \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts="$CONTAINER_MOUNTS" \
  --no-container-mount-home \
  bash -lc "
    set -euo pipefail
    export HF_HOME=/lustre/fsw/portfolios/coreai/users/chcui/hf_home
    export HF_HUB_OFFLINE=1
    export UV_CACHE_DIR=/tmp/uv_cache
    export NCCL_DEBUG=WARN
    export TORCH_NCCL_AVOID_RECORD_STREAMS=1
    export NCCL_NVLS_ENABLE=0
    export NCCL_PXN_DISABLE=$NCCL_PXN_DISABLE
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export NCCL_TIMEOUT=1800000
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    cd $WORKDIR
    rm -f /opt/venv/lib/python3.12/site-packages/__editable__*megatron*.pth \
          /opt/venv/lib/python3.12/site-packages/__editable__*megatron*.py
    mkdir -p $PY_DEPS
    export PYTHONPATH=$PY_DEPS:$TRANSFORMERS_SITE:$FHT_SITE:$CUDNN_FE_SITE:$WORKDIR/src:$MCORE_DIR:\${PYTHONPATH:-}
    source $WORKDIR/examples/models/deepseek_v4/hsg_runtime.sh
    ensure_transformers_site $TRANSFORMERS_SITE
    load_checkpoint_arg=()
    if [[ -n \"$LOAD_CHECKPOINT\" ]]; then
      load_checkpoint_arg=(--load-checkpoint \"$LOAD_CHECKPOINT\")
    fi
    uv run --no-sync python -m torch.distributed.run \
      --nproc_per_node=$NPROC_PER_NODE \
      --nnodes=$SLURM_JOB_NUM_NODES \
      --node_rank=\$SLURM_PROCID \
      --master_addr=$MASTER_ADDR \
      --master_port=$MASTER_PORT \
      examples/models/deepseek_v4/toy_training.py \
      --model-size full \
      --case adam \
      --hf-config-path $HF_CONFIG \
      --output-dir $OUTDIR \
      --data-dir /preproc_data \
      --dataset-name $DATASET \
      --tokenizer-model /tokenizer \
      --path-to-cache /npy_index \
      --seq-length $SEQ_LENGTH \
      --train-iters $TRAIN_ITERS \
      --eval-interval $EVAL_INTERVAL \
      --eval-iters $EVAL_ITERS \
      --micro-batch-size $MICRO_BATCH_SIZE \
      --global-batch-size $GLOBAL_BATCH_SIZE \
      --tensor-parallel-size $TP \
      --pipeline-parallel-size $PP \
      --context-parallel-size $CP \
      --expert-parallel-size $EP \
      --moe-token-dispatcher-type $MOE_TOKEN_DISPATCHER_TYPE \
      --csa-backend $CSA_BACKEND \
      --adam-max-lr $ADAM_MAX_LR \
      --adam-min-lr $ADAM_MIN_LR \
      --adam-weight-decay $ADAM_WEIGHT_DECAY \
      \"\${load_checkpoint_arg[@]}\" \
      $EXTRA_ARGS
  "

echo "Exit $?"

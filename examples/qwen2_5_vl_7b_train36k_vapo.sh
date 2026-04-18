#!/bin/bash

set -euo pipefail
set -x

export PYTHONUNBUFFERED=1

# Set the model path. Replace with your local path if needed.
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"

# Training and validation data
TRAIN_FILES="xytian1008/VAPO-Thinker-train36k@train"
VAL_FILES="xytian1008/VAPO-Thinker-val1k@val"

# Other config options
CONFIG_FILE="examples/config.yaml"
FORMAT_PROMPT="examples/format_prompt/vapo.jinja"
REWARD_FUNCTION="examples/reward_function/vapo.py:compute_score"

# Hyperparameters
MICRO_BATCH_SIZE_PER_DEVICE_FOR_UPDATE=4
MICRO_BATCH_SIZE_PER_DEVICE_FOR_EXPERIENCE=16
MAX_PROMPT_LENGTH=4096
ROLLOUT_BATCH_SIZE=384
GLOBAL_BATCH_SIZE=128
TORCH_DTYPE=bf16
TENSOR_PARALLEL_SIZE=1
ANCHOR_K=20
REWARD_TYPE=sequential
VAL_BEFORE_TRAIN=True
VAL_ONLY=False
NGPUS_PER_NODE=8
TOTAL_EPOCHS=5
VAL_FREQ=3
SAVE_FREQ=20
SAVE_LIMIT=-1
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
DISABLE_KL=True
ONLINE_FILTERING=True
FILTER_OVERLONG_PROMPTS=True
FILTER_OVERLONG_PROMPTS_WORKERS=128
ROLLOUT_N=5
LR=5e-6


# Set experiment name based on key variables
EXPERIMENT_NAME="exp_$(basename "${MODEL_PATH}")_$(basename "${TRAIN_FILES}")_vapo_grpo_k${ANCHOR_K}_tp${TENSOR_PARALLEL_SIZE}_b${GLOBAL_BATCH_SIZE}"

python3 -m verl.trainer.main \
    config="${CONFIG_FILE}" \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.format_prompt="${FORMAT_PROMPT}" \
    data.rollout_batch_size="${ROLLOUT_BATCH_SIZE}" \
    data.filter_overlong_prompts="${FILTER_OVERLONG_PROMPTS}" \
    data.filter_overlong_prompts_workers="${FILTER_OVERLONG_PROMPTS_WORKERS}" \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.global_batch_size="${GLOBAL_BATCH_SIZE}" \
    worker.actor.fsdp.torch_dtype="${TORCH_DTYPE}" \
    worker.actor.clip_ratio_low="${CLIP_RATIO_LOW}" \
    worker.actor.clip_ratio_high="${CLIP_RATIO_HIGH}" \
    worker.actor.micro_batch_size_per_device_for_update="${MICRO_BATCH_SIZE_PER_DEVICE_FOR_UPDATE}" \
    worker.actor.micro_batch_size_per_device_for_experience="${MICRO_BATCH_SIZE_PER_DEVICE_FOR_EXPERIENCE}" \
    worker.actor.optim.lr="${LR}" \
    worker.rollout.tensor_parallel_size="${TENSOR_PARALLEL_SIZE}" \
    worker.rollout.anchor_k="${ANCHOR_K}" \
    worker.rollout.n="${ROLLOUT_N}" \
    worker.reward.reward_type="${REWARD_TYPE}" \
    worker.reward.reward_function="${REWARD_FUNCTION}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.val_only="${VAL_ONLY}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.val_freq="${VAL_FREQ}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.save_limit="${SAVE_LIMIT}" \
    algorithm.disable_kl="${DISABLE_KL}" \
    algorithm.online_filtering="${ONLINE_FILTERING}"


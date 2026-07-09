#!/usr/bin/env bash
set -euo pipefail
set -x

# Minimal end-to-end PPO debug run on GSM8K using verl's built-in reward_score.gsm8k.
# Override the variables below from the command line, e.g.:
#   MODEL_PATH=Qwen/Qwen3-0.6B TOTAL_TRAINING_STEPS=1 bash examples/ppo_trainer/run_gsm8k_ppo_debug.sh

export HF_HOME=/root/autodl-tmp/model_weights

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k}"
DEBUG_DATA_DIR="${DEBUG_DATA_DIR:-$HOME/data/gsm8k_debug}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-0.6B}"

N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-1}"
NNODES="${NNODES:-1}"
TP_SIZE="${TP_SIZE:-1}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-256}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-5}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-64}"
VAL_SAMPLES="${VAL_SAMPLES:-16}"
RUN_VALIDATION="${RUN_VALIDATION:-0}"

if [[ ! -f "${DATA_DIR}/train.parquet" || ! -f "${DATA_DIR}/test.parquet" ]]; then
    mkdir -p "${DATA_DIR}"
    python3 "${PROJECT_DIR}/examples/data_preprocess/gsm8k.py" --local_dir "${DATA_DIR}"
fi

mkdir -p "${DEBUG_DATA_DIR}"
python3 - "${DATA_DIR}" "${DEBUG_DATA_DIR}" "${TRAIN_SAMPLES}" "${VAL_SAMPLES}" <<'PY'
import os
import sys

import datasets

src_dir, dst_dir, train_samples, val_samples = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
splits = {
    "train": ("train.parquet", train_samples),
    "test": ("test.parquet", val_samples),
}

for split, (filename, limit) in splits.items():
    src = os.path.join(src_dir, filename)
    dst = os.path.join(dst_dir, filename)
    dataset = datasets.load_dataset("parquet", data_files=src, split="train")
    dataset = dataset.select(range(min(limit, len(dataset))))
    dataset.to_parquet(dst)
    print(f"Wrote {len(dataset)} {split} examples to {dst}")
PY

if [[ "${RUN_VALIDATION}" == "1" ]]; then
    VAL_BEFORE_TRAIN=True
    TEST_FREQ=1
else
    VAL_BEFORE_TRAIN=False
    TEST_FREQ=-1
fi

cd "${PROJECT_DIR}"
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    algorithm.use_kl_in_reward=False \
    data.train_files="${DEBUG_DATA_DIR}/train.parquet" \
    data.val_files="${DEBUG_DATA_DIR}/test.parquet" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.val_batch_size="${VAL_SAMPLES}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
    critic.model.path="${MODEL_PATH}" \
    critic.model.use_remove_padding=False \
    critic.model.enable_gradient_checkpointing=True \
    critic.optim.lr=1e-5 \
    critic.ppo_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    reward_model.enable=False \
    custom_reward_function.path=null \
    trainer.critic_warmup=0 \
    "trainer.logger=['console']" \
    trainer.project_name=verl_debug_gsm8k \
    trainer.experiment_name=ppo_default_reward \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.save_freq=-1 \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.total_epochs=1 \
    trainer.resume_mode=disable \
    "$@"

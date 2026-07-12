#!/bin/bash
set -x

# DATA_HOME=${DATA_HOME:-"${HOME}/verl"}
export HF_HOME='/root/autodl-tmp/model_weights'
DATA_HOME='/root/autodl-tmp/verl'

mkdir -p $DATA_HOME/data

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DATASET_DIR="$DATA_HOME/data/retool_multi_turn_sft_preprocessed"
if [ ! -f "$SFT_DATASET_DIR/train.parquet" ] || [ ! -f "$SFT_DATASET_DIR/test.parquet" ]; then
  python "$SCRIPT_DIR/retool_multi_turn_sft_preprocess.py" --local_dir "$SFT_DATASET_DIR"
fi

TRAIN_DATA=$SFT_DATASET_DIR/train.parquet
EVAL_DATA=$SFT_DATASET_DIR/test.parquet
MODEL_PATH=Qwen/Qwen3-4B
SAVE_PATH=$DATA_HOME/checkpoints

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_DATA \
    data.val_files=$EVAL_DATA \
    data.max_length=4096 \
    data.train_batch_size=8 \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.multiturn.tools_key=tools \
    data.micro_batch_size_per_gpu=2 \
    model.partial_pretrain=$MODEL_PATH \
    model.trust_remote_code=true \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=multiturn-sft \
    trainer.experiment_name=multiturn-sft-qwen-3-8b-instruct-sp2 \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=2 \
    trainer.default_hdfs_dir=null \
    ulysses_sequence_parallel_size=4 \
    use_remove_padding=true
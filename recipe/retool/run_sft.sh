#!/bin/bash
set -x

# DATA_HOME=${DATA_HOME:-"${HOME}/verl"}
export HF_HOME='/root/autodl-tmp/model_weights'
DATA_HOME='/root/autodl-tmp/verl'

mkdir -p $DATA_HOME/data

SFT_DATASET_PATH="$DATA_HOME/data/retool_sft_dataset.parquet"
if [ ! -f "$SFT_DATASET_PATH" ]; then
  wget -O "$SFT_DATASET_PATH" \
    "https://huggingface.co/datasets/vermouth1992/ReTool-SFT/resolve/main/data/train-00000-of-00001.parquet"
fi

TRAIN_DATA=$SFT_DATASET_PATH
EVAL_DATA=$SFT_DATASET_PATH
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
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=multiturn-sft \
    trainer.experiment_name=multiturn-sft-qwen-3-8b-instruct-sp2 \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=2 \
    trainer.default_hdfs_dir=null \
    ulysses_sequence_parallel_size=4 \
    use_remove_padding=true
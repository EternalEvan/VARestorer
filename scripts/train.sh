#!/usr/bin/env bash

set -x

SINGLE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
nproc_per_node=${NPROC_PER_NODE:-8}
node_rank=${NODE_RANK:-0}
master_addr=${MASTER_ADDR:-127.0.0.1}
master_port=${MASTER_PORT:-12345}

export OMP_NUM_THREADS=1
# export NCCL_IB_DISABLE=1  
# export NCCL_SOCKET_IFNAME=eth0
# unset PYTORCH_CUDA_ALLOC_CONF 
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}

BED=checkpoints
LOCAL_OUT=local_output
mkdir -p "${BED}"
mkdir -p "${LOCAL_OUT}"
mkdir -p visualize_train

export COMPILE_GAN=0
export USE_TIMELINE_SDK=1
export CUDA_TIMER_STREAM_KAFKA_CLUSTER=bmq_data_va
export CUDA_TIMER_STREAM_KAFKA_TOPIC=megatron_cuda_timer_tracing_original_v2
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

if command -v wandb >/dev/null 2>&1; then
    wandb offline
fi
exp_name=${EXP_NAME:-varestorer}
bed_path=checkpoints/${exp_name}/
data_path=${DATA_PATH:-data/LSDIR/splits}
video_data_path=${VIDEO_DATA_PATH:-}
current_time=$(date +"%Y-%m-%d_%H-%M-%S")
local_out_path=$LOCAL_OUT/${exp_name}_${current_time}

swinir_path=${SWINIR_PATH:-weights/general_swinir_v1.ckpt}
blip_path=${BLIP_PATH:-weights/blip-image-captioning-large}
pretrained_checkpoint_path=${PRETRAINED_CHECKPOINT_PATH:-weights/infinity_2b_reg.pth}


TORCHRUN=${TORCHRUN:-torchrun}
"${TORCHRUN}" \
--nproc_per_node=${nproc_per_node} \
--nnodes=1 \
--node_rank=${node_rank} \
--master_addr=${master_addr} \
--master_port=${master_port} \
train.py \
--ep=${EP:-2} \
--opt=adamw \
--cum=${CUM:-3} \
--sche=lin0 \
--fp16=${FP16:-2} \
--ada=0.9_0.97 \
--tini=-1 \
--tclip=5 \
--flash=0 \
--alng=5e-06 \
--saln=1 \
--cos=1 \
--enable_checkpointing=full-block \
--local_out_path "${local_out_path}" \
--task_type='sr' \
--bed="${bed_path}" \
--data_path="${data_path}" \
--video_data_path="${video_data_path}" \
--exp_name="${exp_name}" \
--tblr=1e-4 \
--pn ${PN:-0.25M} \
--model=${MODEL:-2bc8} \
--lbs=${LBS:-8} \
--workers=${WORKERS:-8} \
--short_cap_prob 0.5 \
--online_t5=1 \
--use_streaming_dataset 1 \
--iterable_data_buffersize ${ITERABLE_DATA_BUFFERSIZE:-30000} \
--Ct5=2048 \
--t5_path=weights/flan-t5-xl \
--vae_type 32 \
--vae_ckpt=weights/infinity_vae_d32reg.pth \
--wp 0.00000001 \
--wpe=1 \
--dynamic_resolution_across_gpus 1 \
--enable_dynamic_length_prompt 1 \
--reweight_loss_by_scale 1 \
--add_lvl_embeding_only_first_block 1 \
--rope2d_each_sa_layer 1 \
--rope2d_normalized_by_hw 2 \
--use_fsdp_model_ema 0 \
--always_training_scales 100 \
--use_bit_label 1 \
--zero=${ZERO:-2} \
--save_model_iters_freq ${SAVE_MODEL_ITERS_FREQ:-300} \
--log_freq ${LOG_FREQ:-20} \
--checkpoint_type='torch' \
--prefetch_factor=${PREFETCH_FACTOR:-16} \
--noise_apply_strength 0.3 \
--noise_apply_layers 0 \
--apply_spatial_patchify 0 \
--use_flex_attn=False \
--pad=1 \
--debug_bsc=0 \
--rush_resume="${pretrained_checkpoint_path}" \
--fused_norm True \
--swinir_path="${swinir_path}" \
--blip_path="${blip_path}" 

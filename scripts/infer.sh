#!/bin/bash

# set arguments for inference
export CUDA_VISIBLE_DEVICES="0"
pn=0.25M
model_type=infinity_2b
use_scale_schedule_embedding=0
use_bit_label=1
checkpoint_type='torch'
infinity_model_path=weights/varestorer.pth
vae_type=32
vae_path=weights/infinity_vae_d32reg.pth
input_path=assets/inputs/dog.png
output_path=dog.png

cfg=1
tau=1
rope2d_normalized_by_hw=2
add_lvl_embeding_only_first_block=1
rope2d_each_sa_layer=1
text_encoder_ckpt=weights/flan-t5-xl
text_channels=2048
apply_spatial_patchify=0
sr_scale=4
tiled=0
tile_size=512
tile_overlap=128

# --prompt "A joyful corgi runs across a green lawn, bright daylight, soft focus, lively and playful." \
# --prompt "A serene snowy forest reflects in a calm river beneath a pastel winter sunset sky." \

python tools/run_varestorer.py \
--cfg ${cfg} \
--tau ${tau} \
--pn ${pn} \
--model_path ${infinity_model_path} \
--vae_type ${vae_type} \
--vae_path ${vae_path} \
--add_lvl_embeding_only_first_block ${add_lvl_embeding_only_first_block} \
--use_bit_label ${use_bit_label} \
--model_type ${model_type} \
--rope2d_each_sa_layer ${rope2d_each_sa_layer} \
--rope2d_normalized_by_hw ${rope2d_normalized_by_hw} \
--use_scale_schedule_embedding ${use_scale_schedule_embedding} \
--cfg ${cfg} \
--tau ${tau} \
--checkpoint_type ${checkpoint_type} \
--text_encoder_ckpt ${text_encoder_ckpt} \
--text_channels ${text_channels} \
--apply_spatial_patchify ${apply_spatial_patchify} \
--prompt "A joyful corgi runs across a green lawn, bright daylight, soft focus, lively and playful." \
--seed 1 \
--lq_img_path ${input_path} \
--save_file ${output_path} \
--sr_scale ${sr_scale} \
--tiled ${tiled} \
--tile_size ${tile_size} \
--tile_overlap ${tile_overlap}

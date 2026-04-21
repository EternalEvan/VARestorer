import os
import os.path as osp
import hashlib
import time
import argparse
import json
import shutil
import glob
import re
import sys

import cv2
from tqdm.auto import tqdm
import torch
import numpy as np
from pytorch_lightning import seed_everything

from run_varestorer import *
from conf import HF_TOKEN, HF_HOME
from transformers import BlipForConditionalGeneration,BlipProcessor

# set environment variables
os.environ['HF_TOKEN'] = HF_TOKEN
os.environ['HF_HOME'] = HF_HOME
os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = '1'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument('--out_dir', type=str, default='')
    parser.add_argument('--n_samples', type=int, default=1)
    parser.add_argument('--metadata_file', type=str, default='evaluation/image_reward/benchmark-prompts.json')
    parser.add_argument('--rewrite_prompt', type=int, default=0, choices=[0,1])
    ###
    parser.add_argument('--noise_apply_layers',type=int,default=0)
    parser.add_argument('--noise_apply_requant',type=int,default=1)
    parser.add_argument('--noise_apply_strength',type=float,default=0.3)
    parser.add_argument('--debug_bsc',type=int,default=0)
    ###
    args = parser.parse_args()

    # parse cfg
    args.cfg = list(map(float, args.cfg.split(',')))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]
    
    with open(args.metadata_file) as fp:
        metadatas = json.load(fp)

    if args.model_type == 'sdxl':
        from diffusers import DiffusionPipeline
        base = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to("cuda")
        refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to("cuda")
    elif args.model_type == 'sd3':
        from diffusers import StableDiffusion3Pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
        pipe = pipe.to("cuda")
    elif args.model_type == 'pixart_sigma':
        from diffusers import PixArtSigmaPipeline
        pipe = PixArtSigmaPipeline.from_pretrained(
            "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", torch_dtype=torch.float16
        ).to("cuda")
    elif args.model_type == 'flux_1_dev':
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")
    elif args.model_type == 'flux_1_dev_schnell':
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cuda")
    elif 'infinity' in args.model_type:
        # load text encoder
        text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
        # load vae
        vae = load_visual_tokenizer(args)
        # load infinity
        infinity = load_transformer(vae, args)
        if args.rewrite_prompt:
            from tools.prompt_rewriter import PromptRewriter
            prompt_rewriter = PromptRewriter(system='', few_shot_history=[])
    
    
    os.makedirs(args.out_dir,exist_ok=True)
    save_metadatas = []
    
    
    #####
    blip_processor = BlipProcessor.from_pretrained("weights/blip-image-captioning-large")
    blip_model = BlipForConditionalGeneration.from_pretrained("weights/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")
    #####
    
    #####
    swinir_config = {
        "target": "infinity.models.swinir.SwinIR",  
        "params": {                               
            "img_size": 64,
            "patch_size": 1,
            "in_chans": 3,
            "embed_dim": 180,
            "depths": [6, 6, 6, 6, 6, 6, 6, 6],
            "num_heads": [6, 6, 6, 6, 6, 6, 6, 6],
            "window_size": 8,
            "mlp_ratio": 2,
            "sf": 8,
            "img_range": 1.0,
            "upsampler": "nearest+conv",
            "resi_connection": "1conv",
            "unshuffle": True,
            "unshuffle_scale": 8
        }
    }    
    swinir: SwinIR = instantiate_from_config(swinir_config)
    sd = torch.load('weights/general_swinir_v1.ckpt', map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {
        (k[len("module.") :] if k.startswith("module.") else k): v
        for k, v in sd.items()
    }
    swinir.load_state_dict(sd, strict=True)
    for p in swinir.parameters():
        p.requires_grad = False
    swinir.eval().to("cuda")
    #####
    
    for index, metadata in tqdm(enumerate(metadatas)):
        seed_everything(args.seed)
        
        lq_img_path = metadata['lq_img_path']
        prompt = metadata.get('prompt', None)
        img_name = os.path.relpath(lq_img_path, start=os.path.dirname(lq_img_path))
        sample_path = os.path.join(args.out_dir, img_name)

        tau = args.tau
        cfg = args.cfg
        if args.rewrite_prompt:
            refined_prompt = prompt_rewriter.rewrite(prompt)
            input_key_val = extract_key_val(refined_prompt)
            prompt = input_key_val['prompt']
            print(f'prompt: {prompt}, refined_prompt: {refined_prompt}')
        
        images = []
        bitwise_self_correction= BitwiseSelfCorrection(vae, args)
        for _ in range(args.n_samples):   #####n_samples==1
            t1 = time.time()
            if args.model_type == 'sdxl':
                image = base(
                    prompt=prompt,
                    num_inference_steps=40,
                    denoising_end=0.8,
                    output_type="latent",
                ).images
                image = refiner(
                    prompt=prompt,
                    num_inference_steps=40,
                    denoising_start=0.8,
                    image=image,
                ).images[0]
            elif args.model_type == 'sd3':
                image = pipe(
                    prompt,
                    negative_prompt="",
                    num_inference_steps=28,
                    guidance_scale=7.0,
                    num_images_per_prompt=1,
                ).images[0]
            elif args.model_type == 'flux_1_dev':
                image = pipe(
                    prompt,
                    height=1024,
                    width=1024,
                    guidance_scale=3.5,
                    num_inference_steps=50,
                    max_sequence_length=512,
                    num_images_per_prompt=1,
                ).images[0]
            elif args.model_type == 'flux_1_dev_schnell':
                image = pipe(
                    prompt,
                    height=1024,
                    width=1024,
                    guidance_scale=0.0,
                    num_inference_steps=4,
                    max_sequence_length=256,
                    generator=torch.Generator("cpu").manual_seed(0)
                ).images[0]
            elif args.model_type == 'pixart_sigma':
                image = pipe(prompt).images[0]
            elif 'infinity' in args.model_type:
                h_div_w_template = 1.000
                scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]['scales']
                scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
                tgt_h, tgt_w = dynamic_resolution_h_w[h_div_w_template][args.pn]['pixel']
                # image,prompt = gen_one_img_eval(infinity, 
                #                     vae, 
                #                     text_tokenizer, 
                #                     text_encoder,
                #                     prompt, 
                #                     tau_list=tau, 
                #                     cfg_sc=3, 
                #                     cfg_list=cfg, 
                #                     scale_schedule=scale_schedule, 
                #                     cfg_insertion_layer=[args.cfg_insertion_layer], 
                #                     vae_type=args.vae_type, 
                #                     lq_img_path=lq_img_path,
                #                     args=args,
                #                     blip_model=blip_model,
                #                     blip_processor=blip_processor,
                #                     )
                image,prompt = gen_one_img_eval_long(infinity, 
                                    vae, 
                                    text_tokenizer, 
                                    text_encoder,
                                    prompt, 
                                    tau_list=tau, 
                                    cfg_sc=3, 
                                    cfg_list=cfg, 
                                    scale_schedule=scale_schedule, 
                                    cfg_insertion_layer=[args.cfg_insertion_layer], 
                                    vae_type=args.vae_type, 
                                    lq_img_path=lq_img_path,
                                    args=args,
                                    blip_model=blip_model,
                                    blip_processor=blip_processor,
                                    swinir=swinir,
                                    bitwise_self_correction=bitwise_self_correction
                                    )
            else:
                raise ValueError
            t2 = time.time()
            images.append(image)
        
        
        for i, image in enumerate(images):
            if 'infinity' in args.model_type:
                cv2.imwrite(sample_path, image.cpu().numpy())
            else:
                image.save(sample_path)
                
        metadata['prompt']=prompt
        save_metadatas.append(metadata)

    save_metadata_file_path = os.path.join(os.path.dirname(args.metadata_file), "metadata_w_prompt.json")
    with open(save_metadata_file_path, "w") as fp:
        json.dump(save_metadatas, fp)




import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os.path as osp
from typing import List
from contextlib import nullcontext
import math
import hashlib
import yaml
import argparse
import shutil
import re

import cv2
import numpy as np
import torch
torch._dynamo.config.cache_size_limit=64
import pandas as pd
from transformers import AutoTokenizer, T5EncoderModel, T5TokenizerFast
from PIL import Image, ImageEnhance
import torch.nn.functional as F
from torch.cuda.amp import autocast

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from infinity.models.infinity import Infinity,BInfinity
from infinity.models.basic import *
import PIL.Image as PImage
from torchvision.transforms.functional import to_tensor
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
import pdb
from torchvision import transforms
from infinity.models.bitwise_self_correction import BitwiseSelfCorrection
from infinity.models.swinir import SwinIR
import importlib
from lora_diffusion import inject_trainable_lora
from transformers import BlipForConditionalGeneration,BlipProcessor

def get_obj_from_str(string: str, reload: bool = False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def extract_key_val(text):
    pattern = r'<(.+?):(.+?)>'
    matches = re.findall(pattern, text)
    key_val = {}
    for match in matches:
        key_val[match[0]] = match[1].lstrip()
    return key_val

def encode_prompt(text_tokenizer, text_encoder, prompt, enable_positive_prompt=False):
    if enable_positive_prompt:
        print(f'before positive_prompt aug: {prompt}')
        prompt = aug_with_positive_prompt(prompt)
        print(f'after positive_prompt aug: {prompt}')
    # print(f'prompt={prompt}')
    captions = [prompt]
    tokens = text_tokenizer(text=captions, max_length=512, padding='max_length', truncation=True, return_tensors='pt')  # todo: put this into dataset
    input_ids = tokens.input_ids.cuda(non_blocking=True)
    mask = tokens.attention_mask.cuda(non_blocking=True)
    text_features = text_encoder(input_ids=input_ids, attention_mask=mask)['last_hidden_state'].float()
    lens: List[int] = mask.sum(dim=-1).tolist()
    cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
    Ltext = max(lens)    
    kv_compact = []
    for len_i, feat_i in zip(lens, text_features.unbind(0)):
        kv_compact.append(feat_i[:len_i])
    kv_compact = torch.cat(kv_compact, dim=0)
    text_cond_tuple = (kv_compact, lens, cu_seqlens_k, Ltext)
    return text_cond_tuple

def aug_with_positive_prompt(prompt):
    for key in ['man', 'woman', 'men', 'women', 'boy', 'girl', 'child', 'person', 'human', 'adult', 'teenager', 'employee', 
                'employer', 'worker', 'mother', 'father', 'sister', 'brother', 'grandmother', 'grandfather', 'son', 'daughter']:
        if key in prompt:
            prompt = prompt + '. very smooth faces, good looking faces, face to the camera, perfect facial features'
            break
    return prompt


def enhance_image(image):
    for t in range(1):
        contrast_image = image.copy()
        contrast_enhancer = ImageEnhance.Contrast(contrast_image)
        contrast_image = contrast_enhancer.enhance(1.05)
        color_image = contrast_image.copy()
        color_enhancer = ImageEnhance.Color(color_image)
        color_image = color_enhancer.enhance(1.05)
    return color_image


def load_swinir_model(device, swinir=None):
    if swinir is not None:
        return swinir

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
    swinir = instantiate_from_config(swinir_config)
    sd = torch.load('weights/general_swinir_v1.ckpt', map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in sd.items()
    }
    swinir.load_state_dict(sd, strict=True)
    for p in swinir.parameters():
        p.requires_grad = False
    swinir.eval().to(device)
    return swinir


def pil_to_lq_tensor(pil_img, device):
    transform = transforms.ToTensor()
    lq_img = transform(pil_img)
    lq_img = lq_img * 2 - 1
    return lq_img.unsqueeze(0).to(device, non_blocking=True)


def run_swinir(lq_img, swinir):
    lq_img = (lq_img + 1) / 2
    lq_img = swinir(lq_img)
    lq_img = lq_img + lq_img - 1
    return lq_img


def prepare_prompt_conditions(
    text_tokenizer,
    text_encoder,
    prompt,
    negative_prompt='',
    enable_positive_prompt=0,
):
    text_cond_tuple = encode_prompt(text_tokenizer, text_encoder, prompt, enable_positive_prompt)
    if negative_prompt:
        negative_label_B_or_BLT = encode_prompt(text_tokenizer, text_encoder, negative_prompt)
    else:
        negative_label_B_or_BLT = None
    return text_cond_tuple, negative_label_B_or_BLT


def gaussian_tile_weights(tile_height, tile_width):
    from numpy import exp, pi, sqrt

    var = 0.01
    x_mid = (tile_width - 1) / 2
    y_mid = (tile_height - 1) / 2
    x_probs = [exp(-(x - x_mid) * (x - x_mid) / (tile_width * tile_width) / (2 * var)) / sqrt(2 * pi * var) for x in range(tile_width)]
    y_probs = [exp(-(y - y_mid) * (y - y_mid) / (tile_height * tile_height) / (2 * var)) / sqrt(2 * pi * var) for y in range(tile_height)]
    weights = np.outer(y_probs, x_probs).astype(np.float32)
    weights /= weights.max()
    return weights[..., None]


def compute_tile_positions(length, tile_size, tile_overlap):
    if length <= tile_size:
        return [0]

    stride = tile_size - tile_overlap
    if stride <= 0:
        raise ValueError(f'tile_overlap must be smaller than tile_size, got {tile_overlap=} {tile_size=}')

    positions = list(range(0, length - tile_size + 1, stride))
    if positions[-1] != length - tile_size:
        positions.append(length - tile_size)
    return positions


def crop_tile_with_padding(image_np, top, left, tile_size):
    bottom = min(top + tile_size, image_np.shape[0])
    right = min(left + tile_size, image_np.shape[1])
    tile = image_np[top:bottom, left:right]

    pad_bottom = tile_size - tile.shape[0]
    pad_right = tile_size - tile.shape[1]
    if pad_bottom > 0 or pad_right > 0:
        try:
            tile = np.pad(tile, ((0, pad_bottom), (0, pad_right), (0, 0)), mode='reflect')
        except ValueError:
            tile = np.pad(tile, ((0, pad_bottom), (0, pad_right), (0, 0)), mode='edge')

    return Image.fromarray(tile), bottom - top, right - left


def pad_image_for_tiling(image_np, border_pad):
    if border_pad <= 0:
        return image_np, 0

    try:
        padded = np.pad(image_np, ((border_pad, border_pad), (border_pad, border_pad), (0, 0)), mode='reflect')
    except ValueError:
        padded = np.pad(image_np, ((border_pad, border_pad), (border_pad, border_pad), (0, 0)), mode='edge')
    return padded, border_pad


def resize_image_to_sr_scale(pil_img, sr_scale):
    if sr_scale <= 0:
        raise ValueError(f'sr_scale must be positive, got {sr_scale}')

    width, height = pil_img.size
    target_width = max(1, int(round(width * sr_scale)))
    target_height = max(1, int(round(height * sr_scale)))
    if (target_width, target_height) != (width, height):
        pil_img = pil_img.resize((target_width, target_height), resample=PImage.LANCZOS)
    return pil_img, target_width, target_height


def resize_output_tensor(image_tensor, target_width, target_height):
    image_np = image_tensor.detach().cpu().numpy().astype(np.uint8)
    resized = Image.fromarray(image_np).resize((target_width, target_height), resample=PImage.LANCZOS)
    return torch.from_numpy(np.asarray(resized))


def infer_single_sr_tile(
    infinity_test,
    vae,
    scale_schedule,
    text_cond_tuple,
    negative_label_B_or_BLT,
    lq_img,
    cfg_list=[],
    tau_list=[],
    top_k=900,
    top_p=0.97,
    cfg_sc=3,
    cfg_exp_k=0.0,
    cfg_insertion_layer=-5,
    vae_type=0,
    gumbel=0,
    softmax_merge_topk=-1,
    gt_leak=-1,
    gt_ls_Bl=None,
    g_seed=None,
    sampling_per_bits=1,
    args=None,
    bitwise_self_correction=None,
    requant_mode='flip',
):
    device = lq_img.device

    with torch.amp.autocast('cuda', enabled=False):
        if infinity_test.apply_spatial_patchify:
            vae_scale_schedule = [(pt, 2 * ph, 2 * pw) for pt, ph, pw in scale_schedule]
        else:
            vae_scale_schedule = scale_schedule
        raw_features_lq, _, _ = vae.encode_for_raw_features(lq_img, scale_schedule=vae_scale_schedule)

    if bitwise_self_correction is None:
        if args is None:
            args = globals().get('args', None)
        bitwise_self_correction = BitwiseSelfCorrection(vae, args)
    if requant_mode == 'flip':
        x_BLC_wo_prefix_lq, _ = bitwise_self_correction.flip_requant(vae_scale_schedule, lq_img, raw_features_lq, device)
    elif requant_mode == 'long':
        x_BLC_wo_prefix_lq, _ = bitwise_self_correction.long_flip_requant(vae_scale_schedule, lq_img, raw_features_lq, device)
    else:
        raise ValueError(f'Unsupported requant_mode: {requant_mode}')

    if not isinstance(cfg_list, list):
        cfg_list = [cfg_list] * len(scale_schedule)
    if not isinstance(tau_list, list):
        tau_list = [tau_list] * len(scale_schedule)

    autocast_ctx = torch.cuda.amp.autocast(enabled=device.type == 'cuda', dtype=torch.bfloat16, cache_enabled=True) if device.type == 'cuda' else nullcontext()
    with autocast_ctx:
        _, _, img_list = infinity_test.autoregressive_infer_cfg(
            vae=vae,
            scale_schedule=scale_schedule,
            label_B_or_BLT=text_cond_tuple,
            g_seed=g_seed,
            B=1,
            negative_label_B_or_BLT=negative_label_B_or_BLT,
            force_gt_Bhw=None,
            cfg_sc=cfg_sc,
            cfg_list=cfg_list,
            tau_list=tau_list,
            top_k=top_k,
            top_p=top_p,
            returns_vemb=1,
            ratio_Bl1=None,
            gumbel=gumbel,
            norm_cfg=False,
            cfg_exp_k=cfg_exp_k,
            cfg_insertion_layer=cfg_insertion_layer,
            vae_type=vae_type,
            softmax_merge_topk=softmax_merge_topk,
            ret_img=True,
            trunk_scale=1000,
            gt_leak=gt_leak,
            gt_ls_Bl=gt_ls_Bl,
            inference_mode=True,
            sampling_per_bits=sampling_per_bits,
            x_BLC_wo_prefix_lq=x_BLC_wo_prefix_lq,
        )
    return img_list[0]



def gen_one_img(
    infinity_test, 
    vae, 
    text_tokenizer,
    text_encoder,
    prompt, 
    cfg_list=[],
    tau_list=[],
    negative_prompt='',
    scale_schedule=None,
    top_k=900,
    top_p=0.97,
    cfg_sc=3,
    cfg_exp_k=0.0,
    cfg_insertion_layer=-5,
    vae_type=0,
    gumbel=0,
    softmax_merge_topk=-1,
    gt_leak=-1,
    gt_ls_Bl=None,
    g_seed=None,
    sampling_per_bits=1,
    enable_positive_prompt=0,
    lq_img_path='',
    gt_img_path='',
    args=None,
    swinir=None,
    bitwise_self_correction=None,
):
    lq_img = Image.open(lq_img_path)
    if lq_img.mode != "RGB":
        lq_img = lq_img.convert("RGB")
    lq_img = lq_img.resize((512,512))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    swinir = load_swinir_model(device, swinir=swinir)
    lq_img = pil_to_lq_tensor(lq_img, device)
    lq_img = run_swinir(lq_img, swinir)

    text_cond_tuple, negative_label_B_or_BLT = prepare_prompt_conditions(
        text_tokenizer,
        text_encoder,
        prompt,
        negative_prompt=negative_prompt,
        enable_positive_prompt=enable_positive_prompt,
    )
    print(f'cfg: {cfg_list}, tau: {tau_list}')
    img = infer_single_sr_tile(
        infinity_test,
        vae,
        scale_schedule,
        text_cond_tuple,
        negative_label_B_or_BLT,
        lq_img,
        cfg_list=cfg_list,
        tau_list=tau_list,
        top_k=top_k,
        top_p=top_p,
        cfg_sc=cfg_sc,
        cfg_exp_k=cfg_exp_k,
        cfg_insertion_layer=cfg_insertion_layer,
        vae_type=vae_type,
        gumbel=gumbel,
        softmax_merge_topk=softmax_merge_topk,
        gt_leak=gt_leak,
        gt_ls_Bl=gt_ls_Bl,
        g_seed=g_seed,
        sampling_per_bits=sampling_per_bits,
        args=args,
        bitwise_self_correction=bitwise_self_correction,
    )
    return img


@torch.no_grad()
def gen_one_img_anyres(
    infinity_test,
    vae,
    text_tokenizer,
    text_encoder,
    prompt,
    cfg_list=[],
    tau_list=[],
    negative_prompt='',
    scale_schedule=None,
    top_k=900,
    top_p=0.97,
    cfg_sc=3,
    cfg_exp_k=0.0,
    cfg_insertion_layer=-5,
    vae_type=0,
    gumbel=0,
    softmax_merge_topk=-1,
    gt_leak=-1,
    gt_ls_Bl=None,
    g_seed=None,
    sampling_per_bits=1,
    enable_positive_prompt=0,
    lq_img_path='',
    gt_img_path='',
    args=None,
    swinir=None,
    bitwise_self_correction=None,
    tile_size=512,
    tile_overlap=128,
    sr_scale=1.0,
    tiled=0,
):
    if tile_size != 512:
        raise ValueError(f'Current B inference pipeline expects tile_size=512, got {tile_size}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    swinir = load_swinir_model(device, swinir=swinir)

    if bitwise_self_correction is None:
        if args is None:
            args = globals().get('args', None)
        bitwise_self_correction = BitwiseSelfCorrection(vae, args)

    lq_img = Image.open(lq_img_path)
    if lq_img.mode != "RGB":
        lq_img = lq_img.convert("RGB")
    original_width, original_height = lq_img.size
    lq_img, target_width, target_height = resize_image_to_sr_scale(lq_img, sr_scale)
    print(f'input size: {original_width}x{original_height}, sr_scale: {sr_scale}, target size: {target_width}x{target_height}, tiled: {tiled}')

    text_cond_tuple, negative_label_B_or_BLT = prepare_prompt_conditions(
        text_tokenizer,
        text_encoder,
        prompt,
        negative_prompt=negative_prompt,
        enable_positive_prompt=enable_positive_prompt,
    )

    width, height = lq_img.size
    if not tiled:
        print(f'use resized single-pass inference for {width}x{height}')
        resized_lq = lq_img.resize((tile_size, tile_size), resample=PImage.LANCZOS)
        lq_tensor = pil_to_lq_tensor(resized_lq, device)
        lq_tensor = run_swinir(lq_tensor, swinir)

        img = infer_single_sr_tile(
            infinity_test,
            vae,
            scale_schedule,
            text_cond_tuple,
            negative_label_B_or_BLT,
            lq_tensor,
            cfg_list=cfg_list,
            tau_list=tau_list,
            top_k=top_k,
            top_p=top_p,
            cfg_sc=cfg_sc,
            cfg_exp_k=cfg_exp_k,
            cfg_insertion_layer=cfg_insertion_layer,
            vae_type=vae_type,
            gumbel=gumbel,
            softmax_merge_topk=softmax_merge_topk,
            gt_leak=gt_leak,
            gt_ls_Bl=gt_ls_Bl,
            g_seed=g_seed,
            sampling_per_bits=sampling_per_bits,
            args=args,
            bitwise_self_correction=bitwise_self_correction,
            requant_mode='long',
        )

        if width != tile_size or height != tile_size:
            img = resize_output_tensor(img, width, height)
        return img

    image_np = np.asarray(lq_img)
    image_np, border_pad = pad_image_for_tiling(image_np, tile_overlap)
    padded_height, padded_width = image_np.shape[:2]

    xs = compute_tile_positions(padded_width, tile_size, tile_overlap)
    ys = compute_tile_positions(padded_height, tile_size, tile_overlap)
    tile_weights = gaussian_tile_weights(tile_size, tile_size)

    preds = np.zeros((padded_height, padded_width, 3), dtype=np.float32)
    contributors = np.zeros((padded_height, padded_width, 1), dtype=np.float32)

    total_tiles = len(xs) * len(ys)
    tile_index = 0
    for top in ys:
        for left in xs:
            tile_index += 1
            tile_pil, valid_h, valid_w = crop_tile_with_padding(image_np, top, left, tile_size)
            tile_tensor = pil_to_lq_tensor(tile_pil, device)
            tile_tensor = run_swinir(tile_tensor, swinir)

            tile_img = infer_single_sr_tile(
                infinity_test,
                vae,
                scale_schedule,
                text_cond_tuple,
                negative_label_B_or_BLT,
                tile_tensor,
                cfg_list=cfg_list,
                tau_list=tau_list,
                top_k=top_k,
                top_p=top_p,
                cfg_sc=cfg_sc,
                cfg_exp_k=cfg_exp_k,
                cfg_insertion_layer=cfg_insertion_layer,
                vae_type=vae_type,
                gumbel=gumbel,
                softmax_merge_topk=softmax_merge_topk,
                gt_leak=gt_leak,
                gt_ls_Bl=gt_ls_Bl,
                g_seed=g_seed,
                sampling_per_bits=sampling_per_bits,
                args=args,
                bitwise_self_correction=bitwise_self_correction,
                requant_mode='long',
            )

            tile_img = tile_img.detach().cpu().numpy().astype(np.float32)
            valid_weights = tile_weights[:valid_h, :valid_w]
            preds[top:top + valid_h, left:left + valid_w] += tile_img[:valid_h, :valid_w] * valid_weights
            contributors[top:top + valid_h, left:left + valid_w] += valid_weights
            print(f'processed tile {tile_index}/{total_tiles}: top={top}, left={left}, size={valid_h}x{valid_w}')

    preds /= np.clip(contributors, 1e-8, None)
    preds = np.clip(np.rint(preds), 0, 255).astype(np.uint8)
    if border_pad > 0:
        preds = preds[border_pad:border_pad + height, border_pad:border_pad + width]
    return torch.from_numpy(preds)

@torch.no_grad()
def gen_one_img_eval(
    infinity_test, 
    vae, 
    text_tokenizer,
    text_encoder,
    prompt,
    cfg_list=[],
    tau_list=[],
    negative_prompt='',
    scale_schedule=None,
    top_k=900,
    top_p=0.97,
    cfg_sc=3,
    cfg_exp_k=0.0,
    cfg_insertion_layer=-5,
    vae_type=0,
    gumbel=0,
    softmax_merge_topk=-1,
    gt_leak=-1,
    gt_ls_Bl=None,
    g_seed=None,
    sampling_per_bits=1,
    enable_positive_prompt=0,
    lq_img_path='',
    args=None,
    blip_model=None,
    blip_processor=None,
):
    lq_img = Image.open(lq_img_path)
    if lq_img.mode != "RGB":
        lq_img = lq_img.convert("RGB")
        
    # if scale_schedule[-1][-1]==16:
    #     lq_img = lq_img.resize((256,256))
    
    lq_img = lq_img.resize((512,512))

    transform = transforms.ToTensor()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lq_img = transform(lq_img)
    lq_img = lq_img*2-1
    lq_img = lq_img.unsqueeze(0).to(device, non_blocking=True)
    
    ##### swinir    
    lq_img = (lq_img+1)/2
    lq_img = swinir(lq_img) 
    lq_img = lq_img + lq_img -1
    ##### swinir
    
    #####blip
    if not prompt:
        raw_image = Image.open(lq_img_path).convert('RGB')
        inputs = blip_processor(raw_image, return_tensors="pt").to("cuda", torch.float16)
        out = blip_model.generate(**inputs)
        prompt = blip_processor.decode(out[0], skip_special_tokens=True)
    #####
    
    with torch.amp.autocast('cuda', enabled=False):
        with torch.no_grad():
            if infinity_test.apply_spatial_patchify:
                vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
            else:
                vae_scale_schedule = scale_schedule
            # raw_features, _, _ = vae.encode_for_raw_features(gt_img, scale_schedule=vae_scale_schedule)
            raw_features_lq, _, _ = vae.encode_for_raw_features(lq_img, scale_schedule=vae_scale_schedule)
            
    #####need to change
    bitwise_self_correction= BitwiseSelfCorrection(vae, args)
    x_BLC_wo_prefix_lq,_ = bitwise_self_correction.flip_requant(vae_scale_schedule, lq_img, raw_features_lq, device)
    
    if not isinstance(cfg_list, list):
        cfg_list = [cfg_list] * len(scale_schedule)
    if not isinstance(tau_list, list):
        tau_list = [tau_list] * len(scale_schedule)
    text_cond_tuple = encode_prompt(text_tokenizer, text_encoder, prompt, enable_positive_prompt)
    if negative_prompt:
        negative_label_B_or_BLT = encode_prompt(text_tokenizer, text_encoder, negative_prompt)
    else:
        negative_label_B_or_BLT = None

    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):
        ### single step
        _, _, img_list = infinity_test.autoregressive_infer_cfg(
            vae=vae,
            scale_schedule=scale_schedule,
            label_B_or_BLT=text_cond_tuple, g_seed=g_seed,
            B=1, negative_label_B_or_BLT=negative_label_B_or_BLT, force_gt_Bhw=None,
            cfg_sc=cfg_sc, cfg_list=cfg_list, tau_list=tau_list, top_k=top_k, top_p=top_p,
            returns_vemb=1, ratio_Bl1=None, gumbel=gumbel, norm_cfg=False,
            cfg_exp_k=cfg_exp_k, cfg_insertion_layer=cfg_insertion_layer,
            vae_type=vae_type, softmax_merge_topk=softmax_merge_topk,
            ret_img=True, trunk_scale=1000,
            gt_leak=gt_leak, gt_ls_Bl=gt_ls_Bl, inference_mode=True,
            sampling_per_bits=sampling_per_bits,
            x_BLC_wo_prefix_lq=x_BLC_wo_prefix_lq,
        )
        # ###
    img = img_list[0]
    return img,prompt

@torch.no_grad()
def gen_one_img_eval_long(
    infinity_test, 
    vae, 
    text_tokenizer,
    text_encoder,
    prompt,
    cfg_list=[],
    tau_list=[],
    negative_prompt='',
    scale_schedule=None,
    top_k=900,
    top_p=0.97,
    cfg_sc=3,
    cfg_exp_k=0.0,
    cfg_insertion_layer=-5,
    vae_type=0,
    gumbel=0,
    softmax_merge_topk=-1,
    gt_leak=-1,
    gt_ls_Bl=None,
    g_seed=None,
    sampling_per_bits=1,
    enable_positive_prompt=0,
    lq_img_path='',
    args=None,
    blip_model=None,
    blip_processor=None,
    swinir=None,
    bitwise_self_correction=None
):
    lq_img = Image.open(lq_img_path)
    if lq_img.mode != "RGB":
        lq_img = lq_img.convert("RGB")
        
    # if scale_schedule[-1][-1]==16:
    #     lq_img = lq_img.resize((256,256))
    
    lq_img = lq_img.resize((512,512))

    transform = transforms.ToTensor()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lq_img = transform(lq_img)
    lq_img = lq_img*2-1
    lq_img = lq_img.unsqueeze(0).to(device, non_blocking=True)
    
    lq_img = (lq_img+1)/2
    lq_img = swinir(lq_img) 
    lq_img = lq_img + lq_img -1
    ##### swinir
    
    #####blip
    if not prompt:
        raw_image = Image.open(lq_img_path).convert('RGB')
        inputs = blip_processor(raw_image, return_tensors="pt").to("cuda", torch.float16)
        out = blip_model.generate(**inputs)
        prompt = blip_processor.decode(out[0], skip_special_tokens=True)
    #####
    
    with torch.amp.autocast('cuda', enabled=False):
        with torch.no_grad():
            if infinity_test.apply_spatial_patchify:
                vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
            else:
                vae_scale_schedule = scale_schedule
            # raw_features, _, _ = vae.encode_for_raw_features(gt_img, scale_schedule=vae_scale_schedule)
            raw_features_lq, _, _ = vae.encode_for_raw_features(lq_img, scale_schedule=vae_scale_schedule)
    #####need to change
    # x_BLC_wo_prefix_lq,_ = bitwise_self_correction.flip_requant(vae_scale_schedule, lq_img, raw_features_lq, device)
    # x_BLC_w_prefix_lq,_ = bitwise_self_correction.my_flip_requant(vae_scale_schedule, lq_img, raw_features_lq, device)
    # last_scale_length = scale_schedule[-1][0] * scale_schedule[-1][1] * scale_schedule[-1][2]
    # x_BLC_wo_prefix_lq_long = torch.cat([x_BLC_wo_prefix_lq,x_BLC_w_prefix_lq[:,-last_scale_length:,:]],dim = 1)
    
    x_BLC_wo_prefix_lq_long,_ = bitwise_self_correction.long_flip_requant(vae_scale_schedule, lq_img, raw_features_lq, device)
    #####
    
    if not isinstance(cfg_list, list):
        cfg_list = [cfg_list] * len(scale_schedule)
    if not isinstance(tau_list, list):
        tau_list = [tau_list] * len(scale_schedule)
    text_cond_tuple = encode_prompt(text_tokenizer, text_encoder, prompt, enable_positive_prompt)
    if negative_prompt:
        negative_label_B_or_BLT = encode_prompt(text_tokenizer, text_encoder, negative_prompt)
    else:
        negative_label_B_or_BLT = None

    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):
        ### single step
        _, _, img_list = infinity_test.autoregressive_infer_cfg(
            vae=vae,
            scale_schedule=scale_schedule,
            label_B_or_BLT=text_cond_tuple, g_seed=g_seed,
            B=1, negative_label_B_or_BLT=negative_label_B_or_BLT, force_gt_Bhw=None,
            cfg_sc=cfg_sc, cfg_list=cfg_list, tau_list=tau_list, top_k=top_k, top_p=top_p,
            returns_vemb=1, ratio_Bl1=None, gumbel=gumbel, norm_cfg=False,
            cfg_exp_k=cfg_exp_k, cfg_insertion_layer=cfg_insertion_layer,
            vae_type=vae_type, softmax_merge_topk=softmax_merge_topk,
            ret_img=True, trunk_scale=1000,
            gt_leak=gt_leak, gt_ls_Bl=gt_ls_Bl, inference_mode=True,
            sampling_per_bits=sampling_per_bits,
            x_BLC_wo_prefix_lq=x_BLC_wo_prefix_lq_long,
        )
        # ###
    img = img_list[0]
    return img,prompt

def get_prompt_id(prompt):
    md5 = hashlib.md5()
    md5.update(prompt.encode('utf-8'))
    prompt_id = md5.hexdigest()
    return prompt_id

def save_slim_model(infinity_model_path, save_file=None, device='cpu', key='gpt_fsdp'):
    print('[Save slim model]')
    full_ckpt = torch.load(infinity_model_path, map_location=device)
    infinity_slim = full_ckpt['trainer'][key]
    # ema_state_dict = cpu_d['trainer'].get('gpt_ema_fsdp', state_dict)
    if not save_file:
        save_file = osp.splitext(infinity_model_path)[0] + '-slim.pth'
    print(f'Save to {save_file}')
    torch.save(infinity_slim, save_file)
    print('[Save slim model] done')
    return save_file

def load_tokenizer(t5_path =''):
    print(f'[Loading tokenizer and text encoder]')
    text_tokenizer: T5TokenizerFast = AutoTokenizer.from_pretrained(t5_path, revision=None, legacy=True)
    text_tokenizer.model_max_length = 512
    text_encoder: T5EncoderModel = T5EncoderModel.from_pretrained(t5_path, torch_dtype=torch.float16)
    text_encoder.to('cuda')
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    return text_tokenizer, text_encoder

def load_infinity(
    rope2d_each_sa_layer, 
    rope2d_normalized_by_hw, 
    use_scale_schedule_embedding, 
    pn, 
    use_bit_label, 
    add_lvl_embeding_only_first_block, 
    model_path='', 
    scale_schedule=None, 
    vae=None, 
    device='cuda', 
    model_kwargs=None,
    text_channels=2048,
    apply_spatial_patchify=0,
    use_flex_attn=False,
    bf16=False,
):
    print(f'[Loading Infinity]')
    text_maxlen = 512
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
        infinity_test: Infinity = BInfinity(
            vae_local=vae, text_channels=text_channels, text_maxlen=text_maxlen,
            shared_aln=True, raw_scale_schedule=scale_schedule,
            checkpointing='full-block',
            customized_flash_attn=False,
            fused_norm=True,
            pad_to_multiplier=1,
            use_flex_attn=use_flex_attn,
            add_lvl_embeding_only_first_block=add_lvl_embeding_only_first_block,
            use_bit_label=use_bit_label,
            rope2d_each_sa_layer=rope2d_each_sa_layer,
            rope2d_normalized_by_hw=rope2d_normalized_by_hw,
            pn=pn,
            apply_spatial_patchify=apply_spatial_patchify,
            inference_mode=True,
            train_h_div_w_list=[1.0],
            **model_kwargs,
        ).to(device=device)
        print(f'[you selected Infinity with {model_kwargs=}] model size: {sum(p.numel() for p in infinity_test.parameters())/1e9:.2f}B, bf16={bf16}')

        if bf16:
            for block in infinity_test.unregistered_blocks:
                block.bfloat16()

        ### my code
        # unet_lora_params, train_names = inject_trainable_lora(infinity_test.block_chunks)
        unet_lora_params, train_names = inject_trainable_lora(infinity_test.block_chunks, target_replace_module={"CrossAttention", "SelfAttention"}, r=32)
        
        infinity_test.eval()
        infinity_test.requires_grad_(False)

        infinity_test.cuda()
        torch.cuda.empty_cache()

        print(f'[Load Infinity weights]')
        checkpoint = torch.load(model_path,map_location=device)
        infinity_test.load_state_dict(checkpoint,strict=True)
        
        # state_dict =  checkpoint['infinity']  
        # lora_params = {
        # k: v for k, v in state_dict.items() 
        # if 'lora' in k.lower()  
        # }
        # torch.save(lora_params, 'infinity_lora.pth')
        # pdb.set_trace()
        
        infinity_test.rng = torch.Generator(device=device)
        return infinity_test

def transform(pil_img, tgt_h, tgt_w):
    width, height = pil_img.size
    if width / height <= tgt_w / tgt_h:
        resized_width = tgt_w
        resized_height = int(tgt_w / (width / height))
    else:
        resized_height = tgt_h
        resized_width = int((width / height) * tgt_h)
    pil_img = pil_img.resize((resized_width, resized_height), resample=PImage.LANCZOS)
    # crop the center out
    arr = np.array(pil_img)
    crop_y = (arr.shape[0] - tgt_h) // 2
    crop_x = (arr.shape[1] - tgt_w) // 2
    im = to_tensor(arr[crop_y: crop_y + tgt_h, crop_x: crop_x + tgt_w])
    return im.add(im).add_(-1)

def joint_vi_vae_encode_decode(vae, image_path, scale_schedule, device, tgt_h, tgt_w):
    pil_image = Image.open(image_path).convert('RGB')
    inp = transform(pil_image, tgt_h, tgt_w)
    inp = inp.unsqueeze(0).to(device)
    scale_schedule = [(item[0], item[1], item[2]) for item in scale_schedule]
    h, z, _, all_bit_indices, _, infinity_input = vae.encode(inp, scale_schedule=scale_schedule)
    recons_img = vae.decode(z)[0]
    if len(recons_img.shape) == 4:
        recons_img = recons_img.squeeze(1)
    print(f'recons: z.shape: {z.shape}, recons_img shape: {recons_img.shape}')
    recons_img = (recons_img + 1) / 2
    recons_img = recons_img.permute(1, 2, 0).mul_(255).cpu().numpy().astype(np.uint8)
    gt_img = (inp[0] + 1) / 2
    gt_img = gt_img.permute(1, 2, 0).mul_(255).cpu().numpy().astype(np.uint8)
    print(recons_img.shape, gt_img.shape)
    return gt_img, recons_img, all_bit_indices

def load_visual_tokenizer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load vae
    if args.vae_type in [16,18,20,24,32,64]:
        from infinity.models.bsq_vae.vae import vae_model
        schedule_mode = "dynamic"
        codebook_dim = args.vae_type
        codebook_size = 2**codebook_dim
        if args.apply_spatial_patchify:
            patch_size = 8
            encoder_ch_mult=[1, 2, 4, 4]
            decoder_ch_mult=[1, 2, 4, 4]
        else:
            patch_size = 16
            encoder_ch_mult=[1, 2, 4, 4, 4]
            decoder_ch_mult=[1, 2, 4, 4, 4]
        vae = vae_model(args.vae_path, schedule_mode, codebook_dim, codebook_size, patch_size=patch_size, 
                        encoder_ch_mult=encoder_ch_mult, decoder_ch_mult=decoder_ch_mult, test_mode=True).to(device)
    else:
        raise ValueError(f'vae_type={args.vae_type} not supported')
    return vae

def load_visual_tokenizer_lora(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load vae
    if args.vae_type in [16,18,20,24,32,64]:
        from infinity.models.bsq_vae.vae import vae_model_lora
        schedule_mode = "dynamic"
        codebook_dim = args.vae_type
        codebook_size = 2**codebook_dim
        if args.apply_spatial_patchify:
            patch_size = 8
            encoder_ch_mult=[1, 2, 4, 4]
            decoder_ch_mult=[1, 2, 4, 4]
        else:
            patch_size = 16
            encoder_ch_mult=[1, 2, 4, 4, 4]
            decoder_ch_mult=[1, 2, 4, 4, 4]
        vae = vae_model_lora(args.vae_path, schedule_mode, codebook_dim, codebook_size, patch_size=patch_size, 
                        encoder_ch_mult=encoder_ch_mult, decoder_ch_mult=decoder_ch_mult, test_mode=True).to(device)
    else:
        raise ValueError(f'vae_type={args.vae_type} not supported')
    return vae

def load_transformer(vae, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = args.model_path

    if args.model_type == 'infinity_2b':
        kwargs_model = dict(depth=32, embed_dim=2048, num_heads=2048//128, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8) # 2b model
    elif args.model_type == 'infinity_layer12':
        kwargs_model = dict(depth=12, embed_dim=768, num_heads=8, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer16':
        kwargs_model = dict(depth=16, embed_dim=1152, num_heads=12, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer24':
        kwargs_model = dict(depth=24, embed_dim=1536, num_heads=16, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer32':
        kwargs_model = dict(depth=32, embed_dim=2080, num_heads=20, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer40':
        kwargs_model = dict(depth=40, embed_dim=2688, num_heads=24, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer48':
        kwargs_model = dict(depth=48, embed_dim=3360, num_heads=28, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    infinity = load_infinity(
        rope2d_each_sa_layer=args.rope2d_each_sa_layer, 
        rope2d_normalized_by_hw=args.rope2d_normalized_by_hw,
        use_scale_schedule_embedding=args.use_scale_schedule_embedding,
        pn=args.pn,
        use_bit_label=args.use_bit_label, 
        add_lvl_embeding_only_first_block=args.add_lvl_embeding_only_first_block, 
        model_path=model_path, 
        scale_schedule=None, 
        vae=vae, 
        device=device, 
        model_kwargs=kwargs_model,
        text_channels=args.text_channels,
        apply_spatial_patchify=args.apply_spatial_patchify,
        use_flex_attn=args.use_flex_attn,
        bf16=args.bf16,
    )
    return infinity

def add_common_arguments(parser):
    parser.add_argument('--cfg', type=str, default='3')
    parser.add_argument('--tau', type=float, default=1)
    parser.add_argument('--pn', type=str, required=True, choices=['0.06M', '0.25M', '1M'])
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--cfg_insertion_layer', type=int, default=0)
    parser.add_argument('--vae_type', type=int, default=1)
    parser.add_argument('--vae_path', type=str, default='')
    parser.add_argument('--add_lvl_embeding_only_first_block', type=int, default=0, choices=[0,1])
    parser.add_argument('--use_bit_label', type=int, default=1, choices=[0,1])
    parser.add_argument('--model_type', type=str, default='infinity_2b')
    parser.add_argument('--rope2d_each_sa_layer', type=int, default=1, choices=[0,1])
    parser.add_argument('--rope2d_normalized_by_hw', type=int, default=2, choices=[0,1,2])
    parser.add_argument('--use_scale_schedule_embedding', type=int, default=0, choices=[0,1])
    parser.add_argument('--sampling_per_bits', type=int, default=1, choices=[1,2,4,8,16])
    parser.add_argument('--text_encoder_ckpt', type=str, default='')
    parser.add_argument('--text_channels', type=int, default=2048)
    parser.add_argument('--apply_spatial_patchify', type=int, default=0, choices=[0,1])
    parser.add_argument('--h_div_w_template', type=float, default=1.000)
    parser.add_argument('--use_flex_attn', type=int, default=0, choices=[0,1])
    parser.add_argument('--enable_positive_prompt', type=int, default=0, choices=[0,1])
    parser.add_argument('--cache_dir', type=str, default='/dev/shm')
    parser.add_argument('--checkpoint_type', type=str, default='torch')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bf16', type=int, default=1, choices=[0,1])
    parser.add_argument('--tile_size', type=int, default=512)
    parser.add_argument('--tile_overlap', type=int, default=128)
    parser.add_argument('--sr_scale', type=float, default=1.0)
    parser.add_argument('--tiled', type=int, default=0, choices=[0,1])

def encode_and_decode(lq_img_path,vae,save_path):
    
    lq_img = Image.open(lq_img_path)
    if lq_img.mode != "RGB":
        lq_img = lq_img.convert("RGB")
    transform = transforms.ToTensor()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lq_img = transform(lq_img)
    lq_img = lq_img*2-1
    x = lq_img.unsqueeze(0).to(device, non_blocking=True)

    is_image = x.ndim == 4
    if not is_image:
        B, C, T, H, W = x.shape
    else:
        B, C, H, W = x.shape
        T = 1
    ptdtype = {None: torch.float32, 'fp32': torch.float32, 'bf16': torch.bfloat16}
    enc_dtype = ptdtype[vae.args.encoder_dtype]

    with torch.amp.autocast("cuda", dtype=enc_dtype):
        h, hs, hs_mid = vae.encoder(x, return_hidden=True) # B C H W or B C T H W
    hs = [_h.detach() for _h in hs]
    hs_mid = [_h.detach() for _h in hs_mid]
    h = h.to(dtype=torch.float32)
    # print(z.shape)
    # Multiscale LFQ         
    # z, all_indices, all_loss = vae.quantizer(h)
    z,_,_,_,_,_ = vae.quantizer(h)
    x_recon = vae.decoder(z)
    
    x_recon = (x_recon+1)/2
    x_recon = x_recon.squeeze(0)
    to_pil = transforms.ToPILImage()
    x_recon = to_pil(x_recon)  
    x_recon.save(save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument('--prompt', type=str, default='a dog')
    parser.add_argument('--save_file', type=str, default='./tmp.jpg')
    parser.add_argument('--lq_img_path', type=str, default='')
    parser.add_argument('--noise_apply_layers',type=int,default=-1)
    parser.add_argument('--noise_apply_requant',type=int,default=1)
    parser.add_argument('--noise_apply_strength',type=float,default=0.3)
    parser.add_argument('--debug_bsc',type=int,default=0)
    args = parser.parse_args()
    # noise_apply_layers: int = 13        # Bitwise Self-Correction: apply noise to layers, -1 means not apply noise
    # noise_apply_strength: float = 0.3    # Bitwise Self-Correction: apply noise strength, -1 means not apply noise
    # noise_apply_requant: int = 1        # Bitwise Self-Correction: requant after apply noise
    # debug_bsc: int = 0   

    # parse cfg
    args.cfg = list(map(float, args.cfg.split(',')))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]
    
    # load text encoder
    text_tokenizer, text_encoder = load_tokenizer(t5_path =args.text_encoder_ckpt)
    # load vae
    vae = load_visual_tokenizer(args)
    # load infinity
    infinity = load_transformer(vae, args)
    bitwise_self_correction = BitwiseSelfCorrection(vae, args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    swinir = load_swinir_model(device)
    
    scale_schedule = dynamic_resolution_h_w[args.h_div_w_template][args.pn]['scales']
    scale_schedule = [ (1, h, w) for (_, h, w) in scale_schedule]
    

    with autocast(dtype=torch.bfloat16):
        with torch.no_grad():
            generated_image = gen_one_img_anyres(
                infinity,
                vae,
                text_tokenizer,
                text_encoder,
                args.prompt,
                g_seed=args.seed,
                gt_leak=0,
                gt_ls_Bl=None,
                cfg_list=args.cfg,
                tau_list=args.tau,
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[args.cfg_insertion_layer],
                vae_type=args.vae_type,
                sampling_per_bits=args.sampling_per_bits,
                enable_positive_prompt=args.enable_positive_prompt,
                lq_img_path=args.lq_img_path,
                args=args,
                swinir=swinir,
                bitwise_self_correction=bitwise_self_correction,
                tile_size=args.tile_size,
                tile_overlap=args.tile_overlap,
                sr_scale=args.sr_scale,
                tiled=args.tiled,
            )
    os.makedirs(osp.dirname(osp.abspath(args.save_file)), exist_ok=True)
    cv2.imwrite(args.save_file, generated_image.cpu().numpy())
    print(f'Save to {osp.abspath(args.save_file)}')

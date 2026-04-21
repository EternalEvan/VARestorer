import gc
import os
import subprocess
import time
import re
from typing import List, Optional, Tuple

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

import glob
import shutil
from infinity.utils import arg_util
import infinity.utils.dist as dist
import pdb

from torch.distributed.fsdp.api import FullOptimStateDictConfig, FullStateDictConfig, StateDictType

def glob_with_epoch_iter(pattern, recursive=False): 
    def extract_ep_iter(filename):
        match = re.search(r'ep(\d+)-iter(\d+)', filename)
        if match:
            ep = int(match.group(1))
            iter_idx = int(match.group(2))
            return ep, iter_idx
        return 0, 0
    return sorted(glob.glob(pattern, recursive=recursive), key=lambda x: extract_ep_iter(os.path.basename(x)), reverse=True)


def glob_with_global_step(pattern, recursive=False): 
    def extract_ep_iter(filename):
        match = re.search(r'global_step_(\d+)', filename)
        if match:
            iter_idx = int(match.group(1))
            return iter_idx
        return 0
    return sorted(glob.glob(pattern, recursive=recursive), key=lambda x: extract_ep_iter(os.path.basename(x)), reverse=True)
        

class CKPTSaver(object):
    def __init__(self, is_master: bool, eval_milestone: List[Tuple[float, float]]):
        self.is_master = is_master
        self.time_stamp = torch.tensor([time.time() - 1e5, time.time()], device=dist.get_device())
        self.sp_also: subprocess.Popen = None
        self.sp_best: subprocess.Popen = None
        self.sp_backup: subprocess.Popen = None
        self.acc_str, self.eval_milestone = '[no acc str]', eval_milestone
    
    def sav(
        self, args: arg_util.Args, g_it: int, next_ep: int, next_it: int, trainer,
        acc_str: Optional[str] = None, eval_milestone: Optional[List[Tuple[float, float]]] = None,
        also_save_to: str = None, best_save_to: str = None,
    ):
        self.time_stamp[1] = time.time()
        dist.broadcast(self.time_stamp, src_rank=0)
        last_save_time, cur_time = self.time_stamp.cpu().tolist()
        
        #my code
        # auto_save = cur_time - last_save_time > 20 * 60
        auto_save = True
        need_save = also_save_to is not None or best_save_to is not None or next_ep == args.ep or auto_save
        if not need_save:
            return
        
        if acc_str is not None: self.acc_str = acc_str
        if eval_milestone is not None: self.eval_milestone = eval_milestone
        
        fname = f'ar-ckpt-giter{g_it//1000:03d}K-ep{next_ep}-iter{next_it}-last.pth' if args.gpt_training else f'ckpt-last.pth'
        local_out_ckpt = os.path.join(args.local_out_path, fname)
        
        # NOTE: all rank should call this state_dict(), not master only!
        # trainer_state = trainer.state_dict()
        # with FSDP.state_dict_type(trainer.gpt, StateDictType.FULL_STATE_DICT):
        #     car_block_chunks_state = trainer.gpt.car_block_chunks.state_dict()
        #     car_var_conv_state = trainer.gpt.car_var_conv.state_dict()
        #     car_skip_norm_state = trainer.gpt.car_skip_norm.state_dict()
        #     car_skip_linear_state = trainer.gpt.car_skip_linear.state_dict()
        fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        fulloptstate_save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(trainer.gpt, StateDictType.FULL_STATE_DICT, fullstate_save_policy, fulloptstate_save_policy):
                infinity_state = trainer.gpt.state_dict()
        # from torch.distributed.fsdp import FullyShardedDataParallel
        # print(f"{isinstance(trainer.vae_local, FullyShardedDataParallel)}")
        # print(f"i{isinstance(trainer.gpt, FullyShardedDataParallel)}")
        # with FSDP.state_dict_type(trainer.vae_local, StateDictType.FULL_STATE_DICT, fullstate_save_policy, fulloptstate_save_policy):
        # vae_state = trainer.vae_local.state_dict()
        
        if self.is_master:
            stt = time.time()
            #my code
            # torch.save({
            #     'args':         args.state_dict(),
            #     'gpt_training': args.gpt_training,
            #     'arch':         args.model if args.gpt_training else args.vv,
            #     'epoch':        next_ep,
            #     'iter':         next_it,
            #     'trainer':      trainer_state,
            #     'acc_str':      self.acc_str,
            #     'milestones':   self.eval_milestone,
            # }, local_out_ckpt)
            torch.save({
                'args':         args.state_dict(),
                'gpt_training': args.gpt_training,
                'arch':         args.model if args.gpt_training else args.vv,
                'epoch':        next_ep,  #start from 1
                'iter':         next_it,  #start from 1
                # 'trainer':      trainer_state,
                # 'car_block_chunks': car_block_chunks_state,
                # 'car_var_conv': car_var_conv_state,
                # 'car_skip_norm':car_skip_norm_state,
                # 'car_skip_linear':car_skip_linear_state,
                # 'controlnet_state_dict': {'car_block_chunks': trainer.gpt.car_block_chunks.state_dict(),
                #                           'car_var_conv': trainer.gpt.car_var_conv.state_dict(),
                #                           'car_skip_norm':trainer.gpt.car_skip_norm.state_dict(),
                #                           'car_skip_linear':trainer.gpt.car_skip_linear.state_dict()
                #                           },
                'infinity':     infinity_state,
                # 'vae':          vae_state,
                'acc_str':      self.acc_str,
                'milestones':   self.eval_milestone,
            }, local_out_ckpt)
            print(f'[CKPTSaver][rank00] start: {also_save_to=} {best_save_to=} {(next_ep == args.ep)=} {auto_save=}  |  see {local_out_ckpt}', flush=True)
            print(f'[CKPTSaver][rank00] dbg: {args.bed=}', flush=True)
            #my code                
            # if auto_save:
            #     if self.sp_backup is not None:
            #         self.sp_backup.wait(timeout=300); self.sp_backup.kill(); self.sp_backup.communicate()
            #     self.time_stamp[0] = time.time()

            #     def auto_sync(source_filename, target_filename):
            #         cmd = f'cp -r {source_filename} {target_filename}'
            #         self.sp_backup = subprocess.Popen(cmd, shell=True, bufsize=-1)
            #         print(f'[CKPTSaver] auto_save cmd: {cmd}', flush=True)

            #     local_files = glob.glob(f"{args.local_out_path}/*")
            #     for filename in local_files:
            #         basename = os.path.basename(filename)
            #         target_filename = f'{args.bed}/{basename}'
            #         if basename.endswith('.pth'):
            #             if not os.path.isfile(target_filename):
            #                 auto_sync(filename, target_filename)
            #         else:
            #             auto_sync(filename, target_filename)                    
            cost = time.time() - stt
            print(f'[CKPTSaver][rank00] cost: {cost:.2f}s', flush=True)
        
        # del trainer_state
        # del car_block_chunks_state
        # del car_var_conv_state
        # del car_skip_norm_state
        # del car_skip_linear_state
        del infinity_state
        # del vae_state
        time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
        dist.barrier()
        
    def sav_w_vae(
        self, args: arg_util.Args, g_it: int, next_ep: int, next_it: int, trainer,
        acc_str: Optional[str] = None, eval_milestone: Optional[List[Tuple[float, float]]] = None,
        also_save_to: str = None, best_save_to: str = None,
    ):
        self.time_stamp[1] = time.time()
        dist.broadcast(self.time_stamp, src_rank=0)
        last_save_time, cur_time = self.time_stamp.cpu().tolist()
        

        auto_save = True
        need_save = also_save_to is not None or best_save_to is not None or next_ep == args.ep or auto_save
        if not need_save:
            return
        
        if acc_str is not None: self.acc_str = acc_str
        if eval_milestone is not None: self.eval_milestone = eval_milestone
        
        fname = f'ar-ckpt-giter{g_it//1000:03d}K-ep{next_ep}-iter{next_it}-last.pth' if args.gpt_training else f'ckpt-last.pth'
        local_out_ckpt = os.path.join(args.local_out_path, fname)
        
        # NOTE: all rank should call this state_dict(), not master only!
        # trainer_state = trainer.state_dict()
        # with FSDP.state_dict_type(trainer.gpt, StateDictType.FULL_STATE_DICT):
        #     car_block_chunks_state = trainer.gpt.car_block_chunks.state_dict()
        #     car_var_conv_state = trainer.gpt.car_var_conv.state_dict()
        #     car_skip_norm_state = trainer.gpt.car_skip_norm.state_dict()
        #     car_skip_linear_state = trainer.gpt.car_skip_linear.state_dict()
        fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        fulloptstate_save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(trainer.gpt, StateDictType.FULL_STATE_DICT, fullstate_save_policy, fulloptstate_save_policy):
                infinity_state = trainer.gpt.state_dict()
        vae_state = trainer.vae_local.state_dict()
        
        if self.is_master:
            stt = time.time()
            torch.save({
                'args':         args.state_dict(),
                'gpt_training': args.gpt_training,
                'arch':         args.model if args.gpt_training else args.vv,
                'epoch':        next_ep,  #start from 1
                'iter':         next_it,  #start from 1
                'infinity':     infinity_state,
                'vae':          vae_state,
                'acc_str':      self.acc_str,
                'milestones':   self.eval_milestone,
            }, local_out_ckpt)
            print(f'[CKPTSaver][rank00] start: {also_save_to=} {best_save_to=} {(next_ep == args.ep)=} {auto_save=}  |  see {local_out_ckpt}', flush=True)
            print(f'[CKPTSaver][rank00] dbg: {args.bed=}', flush=True)               
            cost = time.time() - stt
            print(f'[CKPTSaver][rank00] cost: {cost:.2f}s', flush=True)
        
        del infinity_state
        del vae_state
        time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
        dist.barrier()
        

def auto_resume(args: arg_util.Args, pattern='ckpt*.pth') -> Tuple[List[str], int, int, str, List[Tuple[float, float]], dict, dict]:
    info = []
    resume = ''
    if args.auto_resume:
        for dd in (args.local_out_path, args.bed):
            all_ckpt = glob_with_epoch_iter(os.path.join(dd, pattern))
            if len(all_ckpt): break
        if len(all_ckpt) == 0:
            info.append(f'[auto_resume] no ckpt found @ {pattern}')
            info.append(f'[auto_resume quit]')
        else:
            resume = all_ckpt[0]
            info.append(f'[auto_resume] auto load from @ {resume} ...')
    else:
        info.append(f'[auto_resume] disabled')
        info.append(f'[auto_resume quit]')
    
    if len(resume) == 0:
        return info, 0, 0, '[no acc str]', [], {}, {}

    print(f'auto resume from {resume}')

    try:
        ckpt = torch.load(resume, map_location='cpu')
    except Exception as e:
        info.append(f'[auto_resume] failed, {e} @ {resume}')
        if len(all_ckpt) < 2:
            return info, 0, 0, '[no acc str]', [], {}, {}
        try: # another chance to load from bytenas
            ckpt = torch.load(all_ckpt[1], map_location='cpu')
        except Exception as e:
            info.append(f'[auto_resume] failed, {e} @ {all_ckpt[1]}')
            return info, 0, 0, '[no acc str]', [], {}, {}
    
    dist.barrier()
    ep, it = ckpt['epoch'], ckpt['iter']
    eval_milestone = ckpt.get('milestones', [])
    info.append(f'[auto_resume success] resume from ep{ep}, it{it},    eval_milestone: {eval_milestone}')
    return info, ep, it, ckpt.get('acc_str', '[no acc str]'), eval_milestone, ckpt['trainer'], ckpt['args']

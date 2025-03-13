import sys
import torch
from torch import nn
import argparse
import logging
import math
import os
import shutil
from copy import deepcopy
import types
import gc
from time import time

import einops
from rich import print
import os.path as osp
import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, DPMSolverMultistepScheduler, UNet2DConditionModel, UniPCMultistepScheduler, EulerAncestralDiscreteScheduler, DiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.attention import BasicTransformerBlock, _chunked_feed_forward
from diffusers.models.attention_processor import Attention
from PIL import Image
from mmengine.config import Config, DictAction
from diffusers import ControlNetModel
import safetensors
from src.utils.funcs import *
from src.utils.seg_palette import palette#151个
import pickle
from src.utils.funcs import *
from contextlib import contextmanager
import wandb
from peft import LoraConfig, set_peft_model_state_dict, PeftModel, get_peft_model
from time import time
from torch.utils.data.dataloader import default_collate

import os
import PIL.Image
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from src.utils.causal_loss import ForCausalLMLoss
from tokenizers import AddedToken

from abc import ABC, abstractmethod

class Base_System(nn.Module):
    def __init__(self, 
    ) -> None:
        super().__init__()

    @property
    def set_dataset(self,):
        assert False
    @abstractmethod
    def prepare_trainable(self,):
        pass
    def get_child(self, model):
        print([n for n,t in model.named_children()])
    def get_dtype(self, model):
        return next(model.parameters()).dtype
    def get_device(self, model):
        return next(model.parameters()).device
    def load_dataset_and_loader(self,):
        if self.args.test: self.args.train_data = self.args.test_data

        train_dataset, test_dataset, train_dataloader, test_dataloader = self.set_dataset(##
            self.args, 
            self.args.train_data, 
            self.args.test_data, 
            self.args.train_batch_size, 
            self.args.test_batch_size,
            tokenizer=None, 
            accelerator=None,
            collate_fn=None,
        )
        return train_dataset, test_dataset, train_dataloader, test_dataloader

    def load_dataset_and_loader_2(self,train_dataset=None, test_dataset=None):
        if self.args.test: self.args.train_data_2 = self.args.test_data_2

        train_dataset, test_dataset, train_dataloader, test_dataloader = self.set_dataset(
            self.args, 
            self.args.train_data_2, 
            self.args.test_data_2, 
            self.args.train_batch_size_2, 
            self.args.test_batch_size_2,
            tokenizer=None,
            accelerator=None,
            collate_fn=None,
            train_dataset=train_dataset, test_dataset=train_dataset
        )
        return train_dataset, test_dataset, train_dataloader, test_dataloader

    def setup_data(self, accelerator):
        args = self.args
        train_dataset, test_dataset, train_dataloader, test_dataloader = self.load_dataset_and_loader()

        print(f"len(train_dataset): {len(train_dataset)}", end='\n\n')
        print(f"len(test_dataset): {len(test_dataset)}", end='\n\n')
        
        train_dataloader, test_dataloader = accelerator.prepare(train_dataloader, test_dataloader)
        
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        return train_dataloader, train_dataset

    @abstractmethod
    def forward(self, batch):
        pass

    def resume(self, accelerator=None, stage1_path=None):
        args = self.args
        global_step = 0
        if stage1_path is None:
            if args.resume != "latest":
                if isinstance(args.resume, int):
                    path = f"checkpoint-{args.resume}"
                    path = os.path.join(args.output_dir, path)
                else:
                    path = args.resume
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(args.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None
                if path is not None:
                    path = os.path.join(args.output_dir, path)

            if path is None:
                print(f"Checkpoint '{args.resume}' does not exist. Starting a new training run.")
                args.resume = None
            else:
                print(f"Resuming from checkpoint {path}")
                global_step = int(os.path.basename(path).split("-")[1])

                state_dict = torch.load(osp.join(path, 'trainable_model_parameters.pth'), map_location=torch.device('cpu'))
                mesg = self.load_state_dict(state_dict, strict=False)
                print(mesg)

                # if accelerator is not None:
                #     accelerator.load_state(path, map_location='cpu', strict=False)#key缺少增多问题不大，但是shape要一致
        else:
            state_dict = torch.load(osp.join(stage1_path, 'trainable_model_parameters.pth'), map_location=torch.device('cpu'))
            mesg = self.load_state_dict(state_dict, strict=False)
            print(mesg)

        return global_step
    
    def save_para(self, global_step, accelerator):
        args = self.args
        if args.checkpoints_total_limit is not None:
            checkpoints = os.listdir(args.output_dir)
            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

            if len(checkpoints) >= args.checkpoints_total_limit:
                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                removing_checkpoints = checkpoints[0:num_to_remove]
                print(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                print(f"removing checkpoints: {', '.join(removing_checkpoints)}")
                for removing_checkpoint in removing_checkpoints:
                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                    shutil.rmtree(removing_checkpoint)

        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        mkdir(save_path)
        # accelerator.save_state(save_path)
        # print(f"Saved state to {save_path}")

        # state_dict = {k:v for k,v in self.state_dict().items() if v.requires_grad} #是tensor不是parameter所以requires_grad都是false
        state_dict = {k:v for k,v in self.named_parameters() if v.requires_grad}
        torch.save(state_dict, osp.join(save_path, 'trainable_model_parameters.pth'))

    def check_para(self, model):
        params = [p for p in model.parameters()]
        print('-------------------')
        print(model.__class__, get_parameter_number_params(params))
    
    def get_trainable_para(self, accelerator):
        params = [p for p in self.parameters() if p.requires_grad]
        if accelerator.is_main_process:
            paramsx = [p for p in self.parameters()]
            print(self.__class__, get_parameter_number_params(paramsx))
        return params
    
    def get_trainable_para_lr(self, accelerator):
        names = []
        params_with_lr = []
        for name, p in self.named_parameters():
            if p.requires_grad:
                lr = self.args.learning_rate
                names.append(name)
                params_with_lr.append({'params': p, 'lr':lr})

        params = [p for p in self.parameters() if p.requires_grad]

        if accelerator.is_main_process:
            path = osp.join(self.args.output_dir, 'params.jsonl')
            save_jsonl(path, names)
            for module in self.trainable:
                part_params = [p for p in module.parameters()]
                print('---trainable---')
                print(module.__class__, get_parameter_number_params(part_params))

            paramsx = [p for p in self.parameters()]
            print('++++ sum ++++')
            print(self.__class__, get_parameter_number_params(paramsx))

        return params_with_lr, params

    @property
    def device(self,):
        assert False

    @property
    def dtype(self):
        assert False
    
    @staticmethod
    def freeze_params(params):
        for param in params:
            param.requires_grad = False

    @staticmethod
    def unfreeze_params(params):
        for param in params:
            param.requires_grad = True

    @staticmethod
    def check_dropout_enable(model):
        for name, module in model.named_modules():
            if isinstance(module, nn.Dropout):
                print(f"Dropout layer '{name}' is {'enabled' if module.training else 'disabled'}")

    @abstractmethod
    def validation(self, global_step=0, accelerator=None,):
        pass

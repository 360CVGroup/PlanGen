import sys;sys.path.insert(0, './three_party/Janus')
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
from functools import partial
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
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from src.utils.funcs import *
import pickle
from contextlib import contextmanager
import wandb
from peft import LoraConfig, set_peft_model_state_dict, PeftModel, get_peft_model, TaskType
from time import time
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import ToPILImage, ToTensor
to_pil = ToPILImage()
to_ts = ToTensor()
import os
import PIL.Image
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from src.utils.causal_loss import ForCausalLMLoss
from tokenizers import AddedToken
from .dataset.set_dataset import set_dataset
import traceback

from transformers import AutoModel, AutoTokenizer
import fire

import json
import os

from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from pytorch_lightning import seed_everything
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from project.base.base_system import Base_System
from lightning.pytorch.utilities import CombinedLoader
from .dataset.set_dataset import get_dataset
from transformers.models.llama.modeling_llama import *

class System(Base_System):
    def __init__(self, 
        args=None,
        accelerator=None,
    ) -> None:
        super().__init__()
        if args.test and args.test_data.data_name=='1k':
            args.max_test_len=-1

        self.args = args
        self.accelerator = accelerator

        model_path = self.args.janus_path
        vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        tokenizer = vl_chat_processor.tokenizer

        vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )

        self.vl_chat_processor = vl_chat_processor
        self.ori_image_proc = self.vl_chat_processor.image_processor.__class__.__call__
        self.vl_chat_processor.image_processor.__class__.__call__ = self.hack_image_proc

        self.tokenizer = tokenizer
        self.vl_gpt = vl_gpt
        if self.vl_gpt.vision_model.vision_tower.ignore_head:
            vl_gpt.vision_model.vision_tower.attn_pool = None
        # ['vision_model', 'aligner', 'gen_vision_model', 'gen_aligner', 'gen_head', 'gen_embed', 'language_model']
        # ['model', 'lm_head']

        if self.args.use_special_tokens:
            res = tokenizer.add_tokens([
                AddedToken("<grounding>", special=True),
                AddedToken("</grounding>", special=True),
                AddedToken("<box>", special=True),
                AddedToken("</box>", special=True),
                AddedToken("<ref>", special=True),
                AddedToken("</ref>", special=True),
            ])
            print('\nadd special tokens', res)

        if self.args.use_numhw_tokens:
            hw_list = []
            for i in range(100):
                hw_list.append(AddedToken(f"<h{i}>", special=True))
                hw_list.append(AddedToken(f"<w{i}>", special=True))
            res = tokenizer.add_tokens(hw_list)
            print('\nadd hw_num tokens', res)

        img_size = self.args.janus_hw
        self.image_token_num_per_image = (self.args.janus_hw//16)**2
        self.vl_chat_processor.num_image_tokens = self.image_token_num_per_image
        self.vl_chat_processor.image_processor.image_size = self.args.janus_hw

        self.prepare_trainable()

    def hack_image_proc(self, image, return_tensors='pt'):
        if isinstance(image, torch.Tensor):
            class ImagesOutputs:
                def __init__(self, pixel_values):
                    self.pixel_values = pixel_values
            return ImagesOutputs(image)
        else:
            return self.ori_image_proc(
                self.vl_chat_processor.image_processor,#self
                image, #images
                return_tensors=return_tensors
            )

    def prepare_trainable(self,):
        self.trainable = []
        self.non_trainable = []

        self.freeze_params(self.parameters())

        if self.args.gradient_checkpointing_enable:
            self.vl_gpt.language_model.gradient_checkpointing_enable()

        if self.args.tuning_mode == 'all':
            self.trainable.append(self)
        elif self.args.tuning_mode == 'lm':
            self.trainable.append(self.vl_gpt.language_model)
        elif self.args.tuning_mode == 'lora':
            transformer_lora_config = LoraConfig(
                r=self.args.lora_rank,
                lora_alpha=self.args.lora_alpha,
                init_lora_weights="gaussian",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )#15MB
            self.vl_gpt.language_model.enable_input_require_grads()
            self.vl_gpt = get_peft_model(self.vl_gpt, transformer_lora_config)

            if self.args.tune_token_when_lora and (self.args.use_special_tokens or self.args.use_numhw_tokens):
                self.unfreeze_params(self.vl_gpt.language_model.model.embed_tokens.parameters())

        elif self.args.tuning_mode == 'lora_ranni':
            peft_config = LoraConfig(
                r=64,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                lora_dropout=0.05,
                bias="none",
            )#30MB
            self.vl_gpt = get_peft_model(self.vl_gpt, peft_config)
        elif self.args.tuning_mode == 'stage1':
            self.trainable.append(self.vl_gpt.aligner)
            self.trainable.append(self.vl_gpt.gen_aligner)
            self.trainable.append(self.vl_gpt.gen_head)
        elif self.args.tuning_mode == 'stage2':
            self.trainable.append(self)
            self.non_trainable.append(self.vl_gpt.vision_model)
            self.non_trainable.append(self.vl_gpt.gen_vision_model)
        elif self.args.tuning_mode == 'stage2_lora':
            self.trainable.append(self)
            self.non_trainable.append(self.vl_gpt.vision_model)
            self.non_trainable.append(self.vl_gpt.gen_vision_model)
        elif self.args.tuning_mode == 'stage3':
            self.trainable.append(self)
            self.non_trainable.append(self.vl_gpt.gen_vision_model)
        else:
            assert False

        for module in self.trainable:
            self.unfreeze_params(module.parameters())

        for module in self.non_trainable:
            self.freeze_params(module.parameters())

    def wrap_t2i_prompt(self, 
        caption="a yellow car in front of the tree"
    ):
        conversation = [
            {
                "role": "<|User|>",
                "content": caption,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt="",
        )
        prompt = sft_format + self.vl_chat_processor.image_start_tag

        inputs_ids = self.vl_chat_processor.tokenizer.encode(prompt)
        inputs_ids = torch.LongTensor(inputs_ids)
        return prompt, inputs_ids

    def wrap_uni_prompt(self, 
        caption="a yellow car in front of the tree",
        grounding=None,
        in_stage1=False,
    ):
        conversation = [
            {
                "role": "<|User|>",
                "content": caption,
            },
            {"role": "<|Assistant|>", "content": f"{grounding}"},#可能dropout
        ]

        sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt="",
        )

        if in_stage1:
            prompt = sft_format
        else:
            prompt = sft_format + self.vl_chat_processor.image_start_tag

        inputs_ids = self.vl_chat_processor.tokenizer.encode(prompt)
        inputs_ids = torch.LongTensor(inputs_ids)

        if in_stage1:
            inputs_ids = inputs_ids[...,:-1]
        return prompt, inputs_ids

    def wrap_mmu_prompt(self, 
        question="a yellow car in front of the tree",
        image=None,
        answer="",
    ):
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{question}",
                "images": [image],
            },
            {"role": "<|Assistant|>", "content": f"{answer}"},
        ]

        if isinstance(image, torch.Tensor):
            pil_images = image
        else:
            pil_images = load_pil_images(conversation)

        prepare_inputs = self.vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(self.device)

        prepare_inputs['pixel_values'] = prepare_inputs['pixel_values'].to(torch.bfloat16)

        # # run image encoder to get the image embeddings
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        return prepare_inputs, inputs_embeds

    def decode_text(self, inputs_ids):
        return self.tokenizer.decode(inputs_ids, skip_special_tokens=False)

    def decode_plan_text_batch(self, inputs_ids):
        texts = ["<grounding>"+self.decode_text(t) for t in inputs_ids]
        new_texts = []
        for text in texts:
            end_pos = text.find("</grounding>")
            if end_pos != -1:
                result = text[:end_pos + len("</grounding>")]
            else:
                result = "<grounding>"+"</grounding>"
            new_texts.append(result)
        return new_texts

    def get_pr_grounding_part(self, text):
        pos = text.find("<grounding>")
        if pos != -1:
            text = text[pos:]
        return text
    
    def decode_mmu_text_batch(self, inputs_ids):
        new_ids = []
        for ids in inputs_ids:
            try:
                pos = torch.where(ids==self.tokenizer.eos_token_id)[0][0].item()
                ids = ids[:pos]
            except:
                pass
            new_ids.append(ids)
        inputs_ids = [t for t in new_ids]
        texts = [self.decode_text(t) for t in inputs_ids]
        return texts

    @torch.inference_mode()
    def uni_generate(
        self,
        batch = None,
        gen_path = None,
        batch_idx = None,
        accelerator = None, ###
        prompt: str = None,
        temperature: float = 1,
        parallel_size: int = 4,#16
        cfg_weight: float = 5,
        patch_size: int = 16,
        pred_layout = True,
        pred_image = True,
        save_local = True,
        use_uni_prompt_in_t2i = True,
        is_mmu = False,
        **kwargs,
    ):
        parallel_size = self.args.parallel_size
        img_size = self.args.janus_hw
        image_token_num_per_image = (self.args.janus_hw//16)**2

        print('\n uni...')

        base_caption = batch['base_caption']
        gt_image = batch['image']
        gt_grounding = batch['gt_grounding']

        bs = len(base_caption)

        self.vl_gpt.eval()

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            print(base_caption)

            if pred_layout:
                if is_mmu:
                    prepare_inputs = batch['prepare_inputs_infer']
                    inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
                    attention_mask = prepare_inputs['attention_mask']
                else:
                    inputs_ids = batch['uni_stage1_inputs_ids']
                    attention_mask = batch['uni_stage1_attention_mask']
                    inputs_embeds = self.vl_gpt.language_model.get_input_embeddings()(inputs_ids.to(self.device))
                outputs = self.x2t(inputs_embeds, attention_mask)

                if is_mmu:
                    pr_grounding = self.decode_mmu_text_batch(outputs)
                else:
                    pr_grounding = self.decode_plan_text_batch(outputs)

                if pred_image:
                    bs = len(pr_grounding)
                    all_inputs_ids = []
                    for base_caption_i, grounding_prompt in zip(base_caption, pr_grounding):
                        _, inputs_ids = self.wrap_uni_prompt(base_caption_i, grounding_prompt)
                        all_inputs_ids.append(inputs_ids)
                    uni_inputs_ids, uni_attention_mask = self.pad_input_ids(all_inputs_ids)
                    uni_attention_mask = torch.cat([uni_attention_mask, torch.ones((bs, self.image_token_num_per_image))], dim=-1)
                    batch.update(dict(
                        uni_inputs_ids=uni_inputs_ids.to(self.device),
                        uni_attention_mask=uni_attention_mask.to(self.device)
                    ))
            else:
                pr_grounding = gt_grounding

            if pred_image:
                batch_new = self.t2i_infer_collate_batch(batch, use_uni=use_uni_prompt_in_t2i)
                cfg_emb=None
                cfg_inputs_ids = batch_new['cfg_inputs_ids']
                cfg_attention_mask = batch_new['cfg_attention_mask']

                func = self.t2i
                pr_image, edit_mask = func(
                    None, parallel_size, image_token_num_per_image, cfg_weight, temperature, img_size, patch_size, gt_image, batch, 
                    mask=cfg_attention_mask, 
                    tokens=cfg_inputs_ids,
                    emb=cfg_emb,
                )
                pr_image = pr_image.float()
            else:
                pr_image = gt_image
                edit_mask = None

        self.vl_gpt.train()
        self.clean(accelerator)

        if save_local:
            data = dict(
                base_caption=base_caption, gt_grounding=gt_grounding, pr_grounding=pr_grounding if pred_layout else ''
            )
            json_path = osp.join(gen_path, str(batch_idx)+'_layout.json')
            save_json(json_path, data)

            # if edit_mask is not None:
            #     gt_image[:,0][edit_mask[:,0]==1] = 150

            vis = torch.cat([gt_image, pr_image], dim=0)
            x_grounding = [t for t in gt_grounding]
            for i in range(parallel_size):
                x_grounding += pr_grounding
            assert len(vis) == len(x_grounding)
            vis = torch.cat([vis, donorm_pt(torch.ones_like(gt_image))], dim=0)
            x_grounding += gt_grounding
            if pred_layout:
                vis = torch.cat([vis, donorm_pt(torch.ones_like(pr_image))], dim=0)
                x_grounding += pr_grounding
            if edit_mask is not None:
                vis = torch.cat([vis, edit_mask], dim=0)
                x_grounding += gt_grounding
            if 'edited_image' in batch:
                vis = torch.cat([vis, batch['edited_image']], dim=0)
                x_grounding += gt_grounding
                vis = torch.cat([vis, gt_image], dim=0)
                x_grounding += gt_grounding
            vis = self.vis_image(vis, x_grounding)
            vis = denorm_pt(vis)
            img_path = osp.join(gen_path, str(batch_idx)+'.png')
            save_img(vis, img_path, bs=bs)

            img_each_path = osp.join(gen_path, str(batch_idx))
            mkdir(img_each_path)
            for i in range(len(vis)):
                col = i % bs
                row = i // bs
                to_pil(vis[i]).save(f"{img_each_path}/{row}_{col}.png")

        return dict(
            pr_grounding=pr_grounding, 
            pr_image=pr_image,
        )

    def trans_gr_to_creati(self, prompt):
        pattern = r"<ref>(.*?)</ref><box>\[(.*?)\]</box>"
        matches = re.findall(pattern, prompt)
        prompts = []
        boxes = []
        for desc, box in matches:
            ori_x1, ori_y1, ori_x2, ori_y2 = map(int, box.split(","))
            x1 = ori_x1/1000
            x2 = ori_x2/1000
            y1 = ori_y1/1000
            y2 = ori_y2/1000
            prompts.append(desc)
            boxes.append([x1,y1,x2,y2])
        return boxes, prompts

    def vis_image(self, vis, pr_grounding):
        vis = denorm_pt(vis)
        assert isinstance(pr_grounding, list)
        try:
            assert len(vis) == len(pr_grounding)
        except:
            import pdb;pdb.set_trace()

        creati_style=True
        if creati_style:
            h = 384
            out_vis = []
            for i in range(len(vis)):
                image = to_pil(vis[i])
                boxes, caps = self.trans_gr_to_creati(pr_grounding[i])
                show_input = {"boxes":scale_boxes(boxes,h,h), "labels":caps}
                bbox_visualization_img = bbox_visualization(image,show_input)
                out_vis.append(to_ts(bbox_visualization_img))
            out_vis = torch.stack(out_vis, 0)
            vis = donorm_pt(out_vis)
        else:
            for i in range(len(vis)):
                img = self.draw_boxes_on_image(
                    to_pil(vis[i]),
                    pr_grounding[i],
                )
                vis[i] = donorm_pt(to_ts(img))
        return vis

    def clean(self, accelerator):
        torch.cuda.empty_cache()
        gc.collect()
        if accelerator is not None:
            accelerator.free_memory()

    def draw_boxes_on_image(self, *args):
        return draw_boxes_on_image(*args, use_centerhw=self.args.use_centerhw)

    def x2t(self, inputs_embeds, attention_mask=None):
        return self.vl_gpt.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True,
            )

    def t2i(self, inputs_ids, parallel_size, image_token_num_per_image, cfg_weight, temperature, img_size, patch_size, gt_image=None, batch=None, mask=None, tokens=None, emb=None):
        generator = torch.Generator(device='cuda').manual_seed(self.args.seed)

        if self.args.use_teacher_forcing and gt_image is not None:
            with torch.no_grad():
                gt_images = gt_image.bfloat16()
                bs = gt_images.shape[0]
                gt_labels = self.vl_gpt.gen_vision_model.encode(gt_images)[-1][-1].reshape(bs,-1) # torch.Size([self.image_token_num_per_image])
        else:
            gt_labels = None

        if tokens is None and emb is None:
            tokens = torch.zeros((parallel_size*2, len(inputs_ids)), dtype=torch.int).cuda()
            for i in range(parallel_size*2):
                tokens[i, :] = inputs_ids
                if i % 2 != 0:
                    tokens[i, 1:-1] = self.vl_chat_processor.pad_id
            inputs_embeds = self.vl_gpt.language_model.get_input_embeddings()(tokens)
        else:
            if tokens is None:
                inputs_embeds = emb
            else:
                tokens = torch.cat([tokens]*parallel_size)
                inputs_embeds = self.vl_gpt.language_model.get_input_embeddings()(tokens)
            mask = torch.cat([mask]*parallel_size)

        num_gen = inputs_embeds.shape[0] // 2

        generated_tokens = self.sample_image(inputs_embeds, num_gen, image_token_num_per_image, mask, cfg_weight, temperature, generator, batch, gt_labels)

        dec = self.vl_gpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[num_gen, 8, img_size//patch_size, img_size//patch_size])

        if self.args.use_teacher_forcing:
            mask_image = batch['edit_region']
            bs = mask_image.shape[0]
            mask_image = resize_pt(mask_image.reshape(bs,1,24,24).repeat(1,3,1,1), self.args.janus_hw).to(dec)
        else:
            mask_image = None


        return dec, mask_image

    def sample_image(self, inputs_embeds, num_gen, image_token_num_per_image, mask, cfg_weight, temperature, generator, batch, gt_labels, ):
        generated_tokens = torch.zeros((num_gen, image_token_num_per_image), dtype=torch.int).cuda()

        for i in tqdm(range(image_token_num_per_image)):
            outputs = self.vl_gpt.language_model.model(
                inputs_embeds=inputs_embeds, 
                attention_mask=mask.to(self.device) if mask is not None else None,
                use_cache=True, 
                past_key_values=outputs.past_key_values if i != 0 else None,
            )
            hidden_states = outputs.last_hidden_state
            
            logits = self.vl_gpt.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]

            if self.args.cfg_weight is not None:
                cfg_weight = self.args.cfg_weight
                
            
            logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)
            # 8,16384

            next_token = torch.multinomial(probs, num_samples=1, generator=generator)#bs,1

            if self.args.use_teacher_forcing:
                edit_region = batch['edit_region']
                bs = len(edit_region)
                for bid in range(bs):
                    if edit_region[bid,i].item() == 0:
                        next_token[bid,0] = gt_labels[bid,i]

            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = self.vl_gpt.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)
            # inputs_embeds = torch.cat([inputs_embeds, img_embeds.unsqueeze(dim=1)], dim=1) ## todo

        return generated_tokens

    def t2i_infer_collate(self, batch):
        batch = default_collate(batch)

        ##### t2i
        bs = len(batch['prompt'])
        all_inputs_ids = []
        for prompt in batch['prompt']:
            wrapped_prompt, inputs_ids = self.wrap_t2i_prompt(prompt)
            all_inputs_ids.append(inputs_ids)

        max_length = max(map(len, all_inputs_ids))

        padded_all_inputs_ids = torch.ones((bs, max_length))*self.vl_chat_processor.pad_id
        padded_all_attention_mask = torch.zeros((bs, max_length))
        for i, inputs_ids in enumerate(all_inputs_ids):
            padded_all_inputs_ids[i, -len(inputs_ids):] = inputs_ids
            padded_all_attention_mask[i, -len(inputs_ids):] = 1
        padded_all_inputs_ids = padded_all_inputs_ids.int()
        padded_all_attention_mask = torch.cat([padded_all_attention_mask, torch.ones((bs, self.image_token_num_per_image))], dim=-1)
        padded_all_attention_mask = padded_all_attention_mask.int()
        
        batch.update(dict(
            cfg_inputs_ids=padded_all_inputs_ids,
            cfg_attention_mask=padded_all_attention_mask
        ))
        return batch

    def t2i_infer_collate_batch(self, 
        batch, 
        use_uni=False,
    ):
        bs = len(batch['prompt'])

        if use_uni:
            t2i_inputs_ids = batch['uni_inputs_ids']
            t2i_attention_mask = batch['uni_attention_mask']
        else:
            assert False
            t2i_inputs_ids = batch['t2i_inputs_ids']
            t2i_attention_mask = batch['t2i_attention_mask']

        max_length = t2i_inputs_ids.shape[-1]

        if self.args.use_neg_box:
            neg_all_inputs_ids = []
            for base_caption, grounding_prompt in zip(batch['neg_base_caption'], batch['neg_gt_grounding']):
                _, inputs_ids = self.wrap_uni_prompt(base_caption, grounding_prompt)
                neg_all_inputs_ids.append(inputs_ids)

            max_length_neg = max([len(t) for t in neg_all_inputs_ids])
            if max_length_neg > max_length:
                need_pad = max_length_neg - max_length
                
                t2i_inputs_ids = torch.cat([torch.ones((bs,need_pad)).to(t2i_inputs_ids)*self.vl_chat_processor.pad_id, t2i_inputs_ids], dim=1)

                t2i_attention_mask = torch.cat([torch.zeros((bs,need_pad)).to(t2i_attention_mask)*self.vl_chat_processor.pad_id, t2i_attention_mask], dim=1)
                max_length = max_length_neg

            uni_inputs_ids, uni_attention_mask = self.pad_input_ids(neg_all_inputs_ids, max_length)
            uni_attention_mask_image = torch.cat([uni_attention_mask, torch.ones((bs, self.image_token_num_per_image))], dim=-1)
            neg_ids = uni_inputs_ids
            neg_mask = uni_attention_mask_image
        else:
            # _, neg_inputs_ids = self.wrap_uni_prompt(self.args.neg_prompt, '<grounding></grounding>')
            _, neg_inputs_ids = self.wrap_uni_prompt(self.args.neg_prompt, '')
            # _, neg_inputs_ids = self.wrap_t2i_prompt(self.args.neg_prompt)

            max_length_neg = neg_inputs_ids.shape[-1]
            if max_length_neg > max_length:
                need_pad = max_length_neg - max_length
                
                t2i_inputs_ids = torch.cat([torch.ones((bs,need_pad)).to(t2i_inputs_ids)*self.vl_chat_processor.pad_id, t2i_inputs_ids], dim=1)

                t2i_attention_mask = torch.cat([torch.zeros((bs,need_pad)).to(t2i_attention_mask)*self.vl_chat_processor.pad_id, t2i_attention_mask], dim=1)
                max_length = max_length_neg

            neg_ids, neg_mask = self.pad_input_ids([neg_inputs_ids]*bs, max_length=max_length)
            neg_mask = torch.cat([neg_mask, torch.ones((bs, self.image_token_num_per_image))], dim=-1)

        neg_mask2 = neg_mask

        padded_all_inputs_ids = torch.stack([t2i_inputs_ids, neg_ids.to(self.device)], dim=1).view(bs*2,-1)
        padded_all_attention_mask = torch.stack([t2i_attention_mask, neg_mask2.to(self.device)], dim=1).view(bs*2,-1)
        
        batch.update(dict(
            cfg_inputs_ids=padded_all_inputs_ids.int(),
            cfg_attention_mask=padded_all_attention_mask.int()
        ))
        return batch
    
    def pad_input_ids(self, all_inputs_ids, max_length=None):
        bs = len(all_inputs_ids)

        if self.args.debug_max_seq_len is not None:
            # print('debugging...')
            max_length = self.args.debug_max_seq_len
        if max_length is None:
            max_length = max(map(len, all_inputs_ids))

        padded_all_inputs_ids = torch.ones((bs, max_length))*self.vl_chat_processor.pad_id
        padded_all_attention_mask = torch.zeros((bs, max_length))
        for i, inputs_ids in enumerate(all_inputs_ids):
            padded_all_inputs_ids[i, -len(inputs_ids):] = inputs_ids
            padded_all_attention_mask[i, -len(inputs_ids):] = 1

        if self.args.test or self.args.func is not None:
            pass
        else:
            if padded_all_inputs_ids.shape[1] > self.args.max_seq_len:
                print('pad_input_ids: extend max_seq_len!!!') ## todo
                print(padded_all_inputs_ids.shape)

                num_start = padded_all_inputs_ids.shape[1] - self.args.max_seq_len
                padded_all_inputs_ids = padded_all_inputs_ids[:, num_start:]
                padded_all_attention_mask = padded_all_attention_mask[:, num_start:]

        return padded_all_inputs_ids.int(), padded_all_attention_mask.int()


    def mmu_collatexx(self, batch, pass_default=False):
        if pass_default:
            pass
        else:
            try:
                batch = default_collate(batch) #cpu
            except Exception as e:
                print(e)
                print(batch[0].keys())
                print(batch[1].keys())
                traceback.print_exc()
                import pdb;pdb.set_trace()
        return batch
    
    def mmu_collate(self, batch, pass_default=False):
        if pass_default:
            pass
        else:
            try:
                batch = default_collate(batch) #cpu
            except Exception as e:
                print(e)
                print(batch[0].keys())
                print(batch[1].keys())
                traceback.print_exc()
                import pdb;pdb.set_trace()

        if self.args.func == 'minicpm_cap':
            return batch

        bs = len(batch['prompt'])

        ##### t2i
        all_inputs_ids = []
        for prompt in batch['prompt']:
            _, inputs_ids = self.wrap_t2i_prompt(prompt)
            all_inputs_ids.append(inputs_ids)
        t2i_inputs_ids, t2i_attention_mask = self.pad_input_ids(all_inputs_ids)
        t2i_attention_mask = torch.cat([t2i_attention_mask, torch.ones((bs, self.image_token_num_per_image))], dim=-1)
        batch.update(dict(
            t2i_inputs_ids=t2i_inputs_ids,
            t2i_attention_mask=t2i_attention_mask
        ))

        ### uni
        all_inputs_ids = []
        for base_caption, grounding_prompt in zip(batch['base_caption'], batch['gt_grounding']):
            _, inputs_ids = self.wrap_uni_prompt(base_caption, grounding_prompt)
            all_inputs_ids.append(inputs_ids)
        uni_inputs_ids, uni_attention_mask = self.pad_input_ids(all_inputs_ids)
        uni_attention_mask_image = torch.cat([uni_attention_mask, torch.ones((bs, self.image_token_num_per_image))], dim=-1)
        batch.update(dict(
            uni_inputs_ids=uni_inputs_ids,
            uni_attention_mask=uni_attention_mask_image
        ))
        # uni_attention_mask_image: bs, seq

        # uni_stage1
        all_inputs_ids = []
        for base_caption, grounding_prompt in zip(batch['base_caption'], batch['gt_grounding']):
            _, inputs_ids = self.wrap_uni_prompt(base_caption, "<grounding>", in_stage1=True)
            all_inputs_ids.append(inputs_ids)
        uni_inputs_ids, uni_attention_mask = self.pad_input_ids(all_inputs_ids)
        batch.update(dict(
            uni_stage1_inputs_ids=uni_inputs_ids,
            uni_stage1_attention_mask=uni_attention_mask
        ))

        ### mmu
        all_prepares = []
        image = batch['image']
        answer = batch['prompt']
        question = "Please describe this image and then give the description and bounding box of each object in the image."
        for i in range(len(image)):
            conversation = [
                {"role": "<|User|>",
                "content": f"<image_placeholder>\n{question}",
                "images": [image[i:i+1]],},
                {"role": "<|Assistant|>", "content": f"{answer[i]}"},
            ]
            prepare = self.vl_chat_processor.process_one(
                prompt=None, 
                conversations=conversation, 
                images=image[i:i+1]
            )
            all_prepares.append(prepare)
        prepare_inputs = self.vl_chat_processor.batchify(all_prepares)
        batch.update(dict(
            prepare_inputs=prepare_inputs,
        ))

        ### mmu_infer
        all_prepares = []
        image = batch['image']
        answer = batch['prompt']
        question = "Please describe this image and then give the description and bounding box of each object in the image."
        for i in range(len(image)):
            conversation = [
                {"role": "<|User|>",
                "content": f"<image_placeholder>\n{question}",
                "images": [image[i:i+1]],},
                {"role": "<|Assistant|>", "content": ""},
            ]
            prepare = self.vl_chat_processor.process_one(
                prompt=None, 
                conversations=conversation, 
                images=image[i:i+1]
            )
            all_prepares.append(prepare)
        prepare_inputs = self.vl_chat_processor.batchify(all_prepares)
        batch.update(dict(
            prepare_inputs_infer=prepare_inputs,
        ))
        return batch

    def forward_mmu(self, batch, is_plan=False):

        bs = len(batch['prompt'])

        if is_plan:
            padded_all_inputs_ids = batch['uni_inputs_ids']
            padded_all_attention_mask = batch['uni_attention_mask']

            padded_all_inputs_embeds = self.vl_gpt.language_model.get_input_embeddings()(padded_all_inputs_ids)

        else:
            prepare_inputs = batch['prepare_inputs']
            inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs) # dict_keys(['sft_format', 'input_ids', 'pixel_values', 'attention_mask', 'images_seq_mask', 'images_emb_mask'])

            padded_all_inputs_embeds = inputs_embeds
            padded_all_inputs_ids = prepare_inputs['input_ids']
            padded_all_inputs_ids[padded_all_inputs_ids==100581] = self.vl_chat_processor.pad_id
            padded_all_attention_mask = prepare_inputs['attention_mask']

            if self.args.test or self.args.func is not None:
                pass
            else:
                if padded_all_inputs_embeds.shape[1] > self.args.max_seq_len + 576:
                    print('mmu exceeds maximum length')
                    start = padded_all_inputs_embeds.shape[1] - (self.args.max_seq_len + 576)
                    padded_all_inputs_embeds = padded_all_inputs_embeds[:, start:]
                    padded_all_inputs_ids = padded_all_inputs_ids[:, start:]
                    padded_all_attention_mask = padded_all_attention_mask[:, start:]

        # run the model to get the response
        outputs = self.vl_gpt.language_model.model(
            inputs_embeds=padded_all_inputs_embeds,
            attention_mask=padded_all_attention_mask.to(self.device),
            past_key_values=None,
            use_cache=False,
        )
        hidden_states = outputs.last_hidden_state

        logits = self.vl_gpt.language_model.lm_head(hidden_states)# torch.Size([8, 896, 102400])
        loss = self.cal_lm_loss(
            logits=logits,
            labels=padded_all_inputs_ids.to(torch.long), 
            ignore_index = self.vl_chat_processor.pad_id,
            vocab_size=logits.shape[-1],
        )

        if is_plan:
            return dict(loss_plan_lm=loss)
        else:
            return dict(loss_mmu=loss)

    def cal_lm_loss(
        self, 
        logits,
        labels,
        ignore_index,
        vocab_size,
        is_image=False,
    ):
        return ForCausalLMLoss(
            logits=logits,
            labels=labels,
            ignore_index=ignore_index,
            vocab_size=vocab_size,
        )

    def forward_t2i(self, batch, is_uni=False):
        bs = len(batch['prompt'])
        if is_uni:
            padded_all_inputs_ids = batch['uni_inputs_ids']
            padded_all_attention_mask = batch['uni_attention_mask']
            inputs_embeds = self.vl_gpt.language_model.get_input_embeddings()(padded_all_inputs_ids)
        else:
            padded_all_inputs_ids = batch['t2i_inputs_ids']
            padded_all_attention_mask = batch['t2i_attention_mask']
            inputs_embeds = self.vl_gpt.language_model.get_input_embeddings()(padded_all_inputs_ids)

        with torch.no_grad():
            images = batch['image'].bfloat16()
            all_labels = self.vl_gpt.gen_vision_model.encode(images)[-1][-1].reshape(bs, -1)

        img_embeds = self.vl_gpt.prepare_gen_img_embeds(all_labels)

        num_image_token = img_embeds.shape[1]

        inputs_embeds = torch.cat([inputs_embeds, img_embeds], dim=1)

        outputs = self.vl_gpt.language_model.model(
            inputs_embeds=inputs_embeds, 
            attention_mask=padded_all_attention_mask,
            use_cache=False, 
            past_key_values=None,
            # image_position_ids=image_position_ids,
        )
        hidden_states = outputs.last_hidden_state

        logits = self.vl_gpt.gen_head(hidden_states[:, -(num_image_token+1):, :])


        if self.args.use_local_edit_loss:
            edit_region = batch['edit_region'].bool()
            all_labels = all_labels.clone()
            all_labels[edit_region==0] = self.vl_chat_processor.pad_id
            all_labels = all_labels.detach()

        loss_t2i = self.cal_lm_loss(
            logits=logits,
            labels=torch.cat([torch.zeros((bs,1)).to(all_labels), all_labels], dim=1), 
            ignore_index = self.vl_chat_processor.pad_id,
            vocab_size=logits.shape[-1],
            is_image=True,
        )

        if is_uni:
            logits_lm = self.vl_gpt.language_model.lm_head(hidden_states[:, :-(num_image_token), :])

            loss_lm= self.cal_lm_loss(
                logits=logits_lm,
                labels=padded_all_inputs_ids.to(torch.long), 
                ignore_index = self.vl_chat_processor.pad_id,
                vocab_size=logits_lm.shape[-1],
            )
            loss_dict = dict(
                loss_uni_t2i=loss_t2i,
                loss_uni_lm=loss_lm,
            )
        else:
            loss_dict = dict(loss_t2i=loss_t2i)

        return loss_dict

    def forward_uni(self, batch):
        return self.forward_t2i(batch, is_uni=True)

    def forward_plan(self, batch):
        return self.forward_mmu(batch, is_plan=True)
    
    def setup_data(self, accelerator):
        args = self.args

        if args.debug:
            args.max_val_len = 1

        test_dataset, test_dataloader = get_dataset(
            args,
            args.test_data.data_name, 
            args.test_data.batch_size, 
            is_test=True,
            collate_fn=self.mmu_collate,
        )

        test_dataloader = accelerator.prepare(test_dataloader)
        print(f"test_dataloader: {args.test_data.data_name}, {len(test_dataloader)}")
        self.test_dataset = test_dataset
        self.test_dataloader = test_dataloader

        if self.args.test or self.args.func is not None:
            train_dataset = test_dataset
            train_dataloader = test_dataloader
            self.train_dataset = train_dataset
            self.train_dataloader = train_dataloader
        else:
            iterables_train = {}
            flow2task = {}
            train_datasets = []
            dataset_dict = {}
            for flow_id, data_item in enumerate(args.train_data):
                if self.args.debug:
                    data_item.batch_size = 2
                if self.args.no_full or self.args.debug:
                    if data_item.data_name == 'hico_full':
                        data_item.data_name = 'hico'
                    elif isinstance(data_item.data_name, list):
                         for i in range(len(data_item.data_name)):
                            if data_item.data_name[i] == 'hico_full':
                                data_item.data_name[i] = 'hico'

                dataset_same = dataset_dict.get(str(data_item.data_name), None)
                train_dataset, train_dataloader = get_dataset(
                    args,
                    data_item.data_name, 
                    data_item.batch_size, 
                    collate_fn=self.mmu_collate,
                    dataset=dataset_same,
                )
                dataset_dict[str(data_item.data_name)] = train_dataset

                train_dataloader = accelerator.prepare(train_dataloader)

                print(f"\ntrain_dataset_{flow_id}: {data_item.data_name}, {len(train_dataset)}")
                iterables_train[flow_id] = train_dataloader
                flow2task[flow_id] = data_item.task_type

                train_datasets.append(train_dataset)

            self.flow2task = flow2task

            train_dataloader = CombinedLoader(iterables_train, mode="min_size")
            train_dataloader = iter(train_dataloader)


            print(f"\nAll len(train_dataloader): {len(train_dataloader)}")

            self.train_dataset = train_dataset
            self.train_dataloader = train_dataloader

        test_dataset[0]
        test_dataset[1]
        test_dataset[2]
        return train_dataloader, train_dataset

    def forward(self, batch):
        if self.args.scale_emb_grad is not None:
            a = self.args.scale_emb_grad
            self.vl_gpt.language_model.model.embed_tokens.data = self.vl_gpt.language_model.model.embed_tokens.weight * a + self.vl_gpt.language_model.model.embed_tokens.weight.detach() * (1 - a)

        batch, idx1, idx2 = batch

        loss_dict = {}

        forward_funcs = dict(
            t2i=self.forward_t2i,
            mmu=self.forward_mmu,
            uni=self.forward_uni,
            plan=self.forward_plan,
            edit=self.forward_edit,
        )

        for flow_id in batch:
            task_type = self.flow2task[flow_id]
            func = forward_funcs[task_type]
            loss_dict_sub = func(batch[flow_id])
            loss_dict_sub = {f"{k}_{flow_id}":v for k,v in loss_dict_sub.items()}
            loss_dict.update(loss_dict_sub)

        loss = 0
        for k in loss_dict:
            loss_i = loss_dict[k] * getattr(self.args, f'{k}_scale', 1)
            if 'lm' in k and self.args.plan_lr_scale is not None:
                loss_i = loss_i * self.args.plan_lr_scale
            loss += loss_i
            loss_dict[k] = loss_i.detach().item()

        return loss, loss_dict

    def validation(
        self, 
        global_step=0, 
        accelerator=None, 
        test_mode=False,
        val_num=None
    ):
        args = self.args
        test_mode = args.test or test_mode
        val_num = val_num or args.max_test_len

        if test_mode:
            patha = osp.join(args.output_dir, 'test', f"{args.test_data.data_name}_{args.test_data.task_type}_{val_num}")
            path = osp.join(patha, f"{global_step}")
            batch_path = osp.join(patha, f"{global_step}_batch")
            mkdir(osp.join(path, "gt_image"))
            mkdir(osp.join(path, "pr_image"))
            mkdir(osp.join(path, "image_ids"))
            mkdir(osp.join(path, "gt_image_ids"))
        else:
            path = osp.join(args.output_dir, 'val')
            batch_path = path
        mkdir(path)
        mkdir(batch_path)

        kwargs = {}
        func = self.uni_generate
        if args.test_data.task_type == 't2i':
            kwargs.update(pred_layout=False)
            kwargs.update(use_uni_prompt_in_t2i=False)
        elif args.test_data.task_type == 'uni_2stage':
            pass
        elif args.test_data.task_type == 'uni':
            kwargs.update(pred_layout=False)
        elif args.test_data.task_type == 'mmu':
            kwargs.update(pred_image=False)
            kwargs.update(is_mmu=True)
        elif args.test_data.task_type == 'plan':
            kwargs.update(pred_image=False)
        else:
            assert False

        rets = []
        for idx, batch in enumerate(tqdm(self.test_dataloader)):
            if val_num != -1 and idx >= val_num: break
            if idx >= self.args.test_start:
                pass
            else:
                continue
            
            batch_str = f'{idx}' if test_mode else f'{global_step}_{idx}'
            
            out = func(
                batch=batch, 
                batch_idx=batch_str,
                gen_path=batch_path, 
                accelerator=accelerator,
                parallel_size=1,
                **kwargs,
            )

            if not test_mode:
                break

            gt_image = batch['image']
            image_id = batch['image_id']
            edited_image = batch.get('edited_image', None)
            H = batch['H']
            W = batch['W']
            pr_image = out['pr_image']
            pr_grounding = out['pr_grounding']

            bs = gt_image.shape[0]

            print(path)
            for i in range(len(gt_image)):
                if image_id[i] != '':
                    # pil = to_pil(denorm_pt(pr_image[i]))
                    # h,w = H[i],W[i]
                    # if h * w != 0:
                    #     pil = pil.resize((w,h))
                    to_pil(denorm_pt(pr_image[i])).save(f"{path}/image_ids/{image_id[i]}.jpg")
                    to_pil(denorm_pt(gt_image[i])).save(f"{path}/gt_image_ids/{image_id[i]}.jpg")

                p = self.args.parallel_size
                if p > 1:
                    for t in range(self.args.parallel_size):
                        to_pil(denorm_pt(pr_image[i*p+t])).save(f"{path}/pr_image/{idx*bs+i}_{t}.png")
                else:
                    to_pil(denorm_pt(pr_image[i*p])).save(f"{path}/pr_image/{idx*bs+i}.png")
                to_pil(denorm_pt(gt_image[i])).save(f"{path}/gt_image/{idx*bs+i}.png")

                if edited_image is not None:
                    mkdir(f"{path}/edited_image/")
                    to_pil(denorm_pt(edited_image[i])).save(f"{path}/edited_image/{idx*bs+i}.png")

    @property
    def device(self,):
        return self.get_device(self.vl_gpt)

    @property
    def dtype(self):
        return self.get_dtype(self.vl_gpt)
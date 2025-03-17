import os
os.environ["WANDB_MODE"]="offline"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import datasets
from datasets import load_dataset, ClassLabel, concatenate_datasets
import torch
import numpy as np
import random
from PIL import Image
import json
import copy
from torchvision import transforms
import pickle 
import re
import cv2

from .code_hico.debug_grit import build_grit_dsets
from datasets import load_dataset
import json
import os
from collections import defaultdict
import random
import PIL
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from src.utils.funcs import *
from .sam.sam_traindata import BboxDataset_sam
from .layoutgpt.data_layoutgpt import Dataset_layout
from .edit.dataset_edit import Dataset_edit
from .edit.dataset_edit_coco_rm import Dataset_edit_coco_rm
from .edit.dataset_edit_coco_edit import Dataset_edit_coco_edit
from .coco.data_coco import Dataset_coco
from .oim.data_oim import Dataset_oim
from .hico7k.data_7k import Dataset_7k
from .plan.data_plan import Dataset_plan


class Hico_dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args=None,
        is_test=False,
        tiny_hico_data=True,
        hico_split='train',
        is_sam=False,
        is_mb=False,
        is_7k=False,
        is_plan=False,
        is_creati=False,
        is_layout=False,
        is_edit=False,
        is_coco=False,
        for_rm=False,
        is_oim=False,
        is_gen=False,
        use_1k=False,
        can_dropout=False,
        coco_split='train',
        edit_split='',
        mb_split='train',
        oim_split='train',
        plan_split='llama',
        **kwargs,
    ):
        self.args = args
        self.is_sam = is_sam
        self.is_7k = is_7k
        self.is_mb = is_mb
        self.is_coco = is_coco
        self.is_oim = is_oim
        self.is_creati = is_creati
        self.is_layout = is_layout
        self.is_edit = is_edit
        self.is_gen = is_gen
        self.is_test = is_test
        self.use_1k = use_1k
        self.is_plan = is_plan
        self.can_dropout = can_dropout
        
        if self.is_creati:
            self.dataset = self.load_creati()
        elif self.is_plan:
            self.dataset = Dataset_plan(args, model=plan_split)
        elif self.is_mb:
            self.dataset = Dataset_mb(args, split=mb_split)
        elif self.is_7k:
            self.dataset = Dataset_7k(args)
        elif self.is_coco:
            self.dataset = Dataset_coco(split=coco_split, for_rm=for_rm)
        elif self.is_oim:
            self.dataset = Dataset_oim(args, split=oim_split)
        elif self.is_sam:
            self.dataset = self.load_sam()
        elif self.is_layout:
            self.dataset = Dataset_layout()
        elif self.is_gen:
            self.dataset = Dataset_gen()
        elif self.is_edit:
            if edit_split == 'rm_coco':
                self.dataset = Dataset_edit_coco_rm()
            elif edit_split == 'edit_coco':
                self.dataset = Dataset_edit_coco_edit()
            else:
                assert False
        else:
            self.dataset = self.load_hico(tiny_hico_data, hico_split)

    def load_sam(self):
        dataset_path = self.args.layoutsam_path
        dataset = load_dataset(dataset_path, split='train')
        dataset = BboxDataset_sam(dataset)
        return dataset

    def load_creati(self):
        dataset_path = self.args.layoutsam_eval_path
        dataset = load_dataset(dataset_path, split='test')
        dataset = BboxDataset_sam(dataset, is_testset=True)
        return dataset

    def load_hico(
        self, 
        tiny_hico_data,
        hico_split,
    ):
        args = self.args
        if tiny_hico_data:
            #cfg_data = OmegaConf.load("/home/jovyan/boomcheng-data-shcdt/herunze/code/base/PlanGen/project/plangen/dataset/code_hico/latent_LayoutDiffusion_large.yaml")
            cfg_data = OmegaConf.load("/home/jovyan/boomcheng-data-shcdt/herunze/code/PlanGen/project/plangen/dataset/code_hico/latent_LayoutDiffusion_large.yaml")
        else:
            cfg_data = OmegaConf.load("/home/jovyan/boomcheng-data/aigc/LayoutProj/diffusers_0263/examples/controlnet/latent_LayoutDiffusion_large.yaml")
        pretrained_model = self.args.sdxl_path
        tokenizer_one = AutoTokenizer.from_pretrained(
                    pretrained_model,
                    subfolder="tokenizer",
                    revision=None,
                    use_fast=False,
                )
        tokenizer_two = AutoTokenizer.from_pretrained(
                    pretrained_model,
                    subfolder="tokenizer_2",
                    revision=None,
                    use_fast=False,
                )
        dataset = build_grit_dsets(cfg_data, [tokenizer_one, tokenizer_two], mode=hico_split, args=args)
        return dataset
        
    def get_grounding(self, base_caption, obj_bbox, obj_class, upd_is_valid_obj=None):
        try:
            if obj_bbox.sum() == 0 or upd_is_valid_obj.sum() == 0:
                return base_caption
        except:
            print('null box')
            print(obj_bbox)
            print(obj_class)
            return base_caption

        if len(base_caption) == 0:
            full_prompt = f'<grounding>'
        else:
            full_prompt = f'{base_caption} <grounding>'

        for i in range(len(obj_bbox)):
            box = obj_bbox[i]
            des = obj_class[i]
            if upd_is_valid_obj is None or upd_is_valid_obj[i]:
                if self.args.use_textual:
                    nbox = [round(1000*t) for t in box.cpu().detach().numpy().tolist()]
                    full_prompt += f'<ref>{des}</ref>'
                    full_prompt += f'<box>{nbox}</box>'
                else:
                    nbox = [round(99*t) for t in box.cpu().detach().numpy().tolist()]
                    nbox[0] = f'<h{nbox[0]}>'
                    nbox[1] = f'<w{nbox[1]}>'
                    nbox[2] = f'<h{nbox[2]}>'
                    nbox[3] = f'<w{nbox[3]}>'   
                    full_prompt += f'<ref>{des}</ref>'
                    full_prompt += f'<box>{",".join(nbox)}</box>'
        full_prompt += '</grounding>'
        return full_prompt

    def convert_creati_to_hico(self, meta_data):
        meta_data['region_bboxes_list'] = meta_data['region_bboxes_list'][:10]
        meta_data['region_caption_list'] = meta_data['region_caption_list'][:10]
        meta_data['detail_region_caption_list'] = meta_data['detail_region_caption_list'][:10]

        obj_bbox = torch.tensor(meta_data['region_bboxes_list'])
        need_pad = 10-len(obj_bbox)
        if len(obj_bbox) < 10:
            pad_obj_bbox = torch.cat([obj_bbox, torch.zeros((need_pad, 4))], dim=0)
        else:
            pad_obj_bbox = obj_bbox
            
        obj_class = meta_data['detail_region_caption_list'] + ['']*need_pad
        obj_class_simple = meta_data['region_caption_list'] + ['']*need_pad
        upd_is_valid_obj = torch.tensor([1]*len(obj_bbox) + [0]*need_pad) #list不会collate进行concat
        pixel_values = meta_data['image']
        base_caption = meta_data['global_caption']
        image_path = meta_data['file_name']

        meta_data.update(dict(
            obj_bbox=pad_obj_bbox,
            obj_class=obj_class,
            upd_is_valid_obj=upd_is_valid_obj,
            pixel_values=pixel_values,
            image_path=image_path,
            base_caption=base_caption,
            # cond_image=cond_image,
        ))

        if self.args.use_creati_detail:
            meta_data.update(obj_class_simple=obj_class_simple)

    def convert_layout_to_hico(self, meta_data):

        # 现在都是x1,y1,x2,y2格式
        obj_bbox = torch.tensor(meta_data['obj_bbox']).clamp(0,1)
        need_pad = 10-len(obj_bbox)
        if len(obj_bbox) < 10:
            pad_obj_bbox = torch.cat([obj_bbox, torch.zeros((need_pad, 4))], dim=0)
        else:
            pad_obj_bbox = obj_bbox
            
        obj_class = meta_data['obj_class'] + ['']*need_pad
        upd_is_valid_obj = torch.tensor([1]*len(obj_bbox) + [0]*need_pad) #list不会collate进行concat

        pixel_values = meta_data.get('image', torch.zeros((3,self.args.janus_hw,self.args.janus_hw)))
        base_caption = meta_data['base_caption']
        image_path = ''

        meta_data.update(dict(
            obj_bbox=pad_obj_bbox,
            obj_class=obj_class,
            upd_is_valid_obj=upd_is_valid_obj,
            pixel_values=pixel_values,
            image_path=image_path,
            base_caption=base_caption,
            # cond_image=cond_image,
        ))

    def convert_edit_to_hico(self, meta_data):

        obj_bbox = meta_data['obj_bbox']
        need_pad = 10-len(obj_bbox)
        if len(obj_bbox) < 10:
            pad_obj_bbox = torch.cat([obj_bbox, torch.zeros((need_pad, 4))], dim=0)
        else:
            pad_obj_bbox = obj_bbox
            
        obj_class = meta_data['obj_class'] + ['']*need_pad
        upd_is_valid_obj = torch.tensor([1]*len(obj_bbox) + [0]*need_pad) #list不会collate进行concat
        pixel_values = meta_data['image']
        base_caption = meta_data['base_caption']
        image_path = meta_data['image_path']

        meta_data.update(dict(
            obj_bbox=pad_obj_bbox,
            obj_class=obj_class,
            upd_is_valid_obj=upd_is_valid_obj,
            pixel_values=pixel_values,
            image_path=image_path,
            base_caption=base_caption,
            # cond_image=cond_image,
        ))

    def preprocess_hico(self, meta_data):
        obj_bbox = torch.cat([meta_data['obj_bbox'][1:], torch.zeros((1,4))])
        
        obj_class = meta_data['obj_class'][1:] + ['']
        
        upd_is_valid_obj = torch.cat([meta_data['upd_is_valid_obj'][1:], torch.tensor([0])])
        # for i in range(10):
        #     if obj_bbox[i].sum() == 0:
        #         upd_is_valid_obj[i] = 0
        #     if obj_class[i] == '':
        #         upd_is_valid_obj[i] = 0
        
        # cond_image = meta_data['cond_image']
        # cond_image = torch.cat([cond_image[1:], torch.zeros((1,*cond_image.shape[1:]))])
        
        meta_data.update(dict(
            obj_bbox=obj_bbox,
            obj_class=obj_class,
            upd_is_valid_obj=upd_is_valid_obj,
            # cond_image=cond_image,
        ))

    def __getitem__(self, index):
        meta_data = self.dataset[index]
        old_meta_data = deepcopy(meta_data)

        if self.is_creati or self.is_sam:
            self.convert_creati_to_hico(meta_data)
        elif self.is_layout or self.is_gen or self.is_coco or self.is_oim or self.is_7k or self.is_plan:
            self.convert_layout_to_hico(meta_data)
        elif self.is_edit or self.is_mb:
            self.convert_edit_to_hico(meta_data)
        else:
            self.preprocess_hico(meta_data)
        # dict_keys(['obj_bbox', 'obj_class', 'is_valid_obj', 'upd_is_valid_obj', 'obj_class_text_ids', 'width', 'height', 'original_sizes_hw', 'num_obj_ori', 'new_height', 'new_width', 'crop_top_lefts', 'cond_image', 'base_caption', 'base_class_text_ids', 'num_selected', 'url', 'pixel_values'])

        base_caption = meta_data['base_caption']
        pixel_values = meta_data['pixel_values']
        image_path = meta_data['image_path']

        obj_bbox = meta_data['obj_bbox']
        obj_class = meta_data['obj_class']
        upd_is_valid_obj = meta_data['upd_is_valid_obj']

        assert len(obj_bbox) == len(obj_class)
        for i in range(len(obj_bbox)):
            if obj_bbox[i].sum() == 0:
                upd_is_valid_obj[i] = 0
            if obj_class[i] == '':
                upd_is_valid_obj[i] = 0

        pixel_values = resize_pt(pixel_values, self.args.janus_hw)

        prompt, gt_grounding = self.get_g_prompt(base_caption, obj_bbox, obj_class, upd_is_valid_obj)

        neg_base_caption = self.args.neg_prompt
        neg_gt_grounding = ''
        if self.is_edit:
            if 'edit_region' in meta_data:
                edit_region = meta_data['edit_region'].reshape(-1)
            else:
                h = 384//16
                edit_region = torch.zeros((h, h))
                edit_boxes = meta_data['obj_bbox_edit']
                if self.args.pad_edit_box != 0:
                    dx = edit_boxes[:,2] - edit_boxes[:,0]
                    dy = edit_boxes[:,3] - edit_boxes[:,1]
                    edit_boxes[:,0] -= dx * self.args.pad_edit_box
                    edit_boxes[:,1] -= dy * self.args.pad_edit_box
                    edit_boxes[:,2] += dx * self.args.pad_edit_box
                    edit_boxes[:,3] += dy * self.args.pad_edit_box
                    edit_boxes = edit_boxes.clamp(0,1)
                for box in edit_boxes:
                    x1, y1, x2, y2 = map(lambda x: int(h * x), box)
                    edit_region[y1:y2, x1:x2] = 1
                edit_region = edit_region.reshape(-1)

            if self.args.use_neg_box:
                neg_prompt, neg_gt_grounding = self.get_g_prompt(
                    neg_base_caption, 
                    meta_data['obj_bbox_neg'], 
                    meta_data['obj_class_neg'], 
                    upd_is_valid_obj=torch.ones((len(meta_data['obj_bbox_neg']))))##
            else:
                neg_prompt = neg_base_caption
        else:
            edit_region = torch.zeros((576))
            neg_prompt = neg_base_caption

        ret = dict(
            image=pixel_values.to(torch.float),
            base_caption=base_caption,
            prompt=prompt,
            neg_base_caption=neg_base_caption,
            neg_prompt=neg_prompt,
            gt_grounding=gt_grounding,
            neg_gt_grounding=neg_gt_grounding,
            image_path=image_path,
            edit_region=edit_region.to(torch.long),
            image_id=meta_data.get('image_id', ''),
            H=meta_data.get('H', 0),
            W=meta_data.get('W', 0),
        )

        if self.args.use_creati_detail:
            ret.update(obj_class_simple=meta_data['obj_class_simple'])

        return ret

    def get_g_prompt(self, base_caption, obj_bbox, obj_class, upd_is_valid_obj):
        if self.args.use_textual or self.args.use_numhw_tokens:
            prompt = self.get_grounding(base_caption, obj_bbox, obj_class, upd_is_valid_obj)
            gt_grounding = self.get_grounding("", obj_bbox, obj_class, upd_is_valid_obj)
        else:
            prompt = base_caption
            gt_grounding = ""
        return prompt, gt_grounding

    def __len__(self):
        if self.use_1k:
            return 1000
        return len(self.dataset)

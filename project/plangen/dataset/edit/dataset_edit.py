from torch.utils.data import Dataset
import random
from copy import deepcopy
from torchvision.transforms import Resize
from glob import glob
import torch
from PIL import Image
from torch.utils.data import Dataset
import random
from copy import deepcopy
from torchvision.transforms import Resize
from datasets import load_dataset
from src.utils.funcs import convert_to_np, load_jsonl
import numpy as np
import os
from src.utils.funcs import *

class Dataset_edit(Dataset):
    def __init__(
        self,
        args=None,
        is_test=False,
    ):
        self.args = args
        self.is_test = is_test
        self.datas = load_json("project/janus/dataset/edit/edit.json")

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, i):
        data = self.datas[i]
        base_caption = data.get('base_caption', '')
        new_grounding_prompt = data.get('new_grounding_prompt', '')
        edited_grounding_prompt = data.get('edited_grounding_prompt', '')
        neg_grounding_prompt = data.get('neg_grounding_prompt', '')
        image_path = data.get('image_path', '')

        def get_obj_from_grounding(new_grounding_prompt):
            pattern = r"<ref>(.*?)</ref><box>(.*?)</box>"
            matches = re.findall(pattern, new_grounding_prompt)
            matches = [(x,convert_coordinates(y)) for x,y in matches]

            obj_class = []
            obj_bbox = []
            for desc, box in matches:
                ori_x1, ori_y1, ori_x2, ori_y2 = map(int, box.split(","))
                cx, cy, _h, _w = ori_x1, ori_y1, ori_x2, ori_y2
                x1 = cx - _w / 2
                y1 = cy - _h / 2
                x2 = cx + _w / 2
                y2 = cy + _h / 2
                ori_x1, ori_y1, ori_x2, ori_y2 = x1, y1, x2, y2
                obj_class.append(desc)
                obj_bbox.append([x1, y1, x2, y2])
            return obj_class, obj_bbox

        obj_class, obj_bbox = get_obj_from_grounding(new_grounding_prompt)
        obj_class_edit, obj_bbox_edit = get_obj_from_grounding(edited_grounding_prompt)
        obj_class_neg, obj_bbox_neg = get_obj_from_grounding(neg_grounding_prompt)

        obj_bbox=torch.tensor(obj_bbox).clamp(0,1000)/1000
        obj_bbox_edit=torch.tensor(obj_bbox_edit).clamp(0,1000)/1000
        obj_bbox_neg=torch.tensor(obj_bbox_neg).clamp(0,1000)/1000

        image = load2ts(image_path)

        ret = dict(
            base_caption=base_caption,
            image=image,
            image_path=image_path,
            obj_class=obj_class,
            obj_bbox=obj_bbox,
            obj_bbox_edit=obj_bbox_edit,
            obj_class_neg=obj_class_neg,
            obj_bbox_neg=obj_bbox_neg,
        )
        return ret
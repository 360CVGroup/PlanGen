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

class Dataset_plan(Dataset):
    def __init__(
        self,
        args=None,
        is_test=False,
        model='llama',
    ):
        self.args = args
        self.is_test = is_test

        datas = load_json(f'gen_data/plan1k_{model}_out.json')
        self.datas = datas

        self.caps = load_jsonl('gen_data/1k_cap.jsonl')


    def __len__(self):
        return len(self.datas)

    def __getitem__(self, i):
        data = self.datas[i]
        cap = self.caps[i]

        prompt = cap

        obj_class = data['obj_class']
        obj_bbox = data['obj_bbox']

        obj_bbox = torch.tensor(obj_bbox).to(torch.float)
        obj_bbox = obj_bbox.reshape(-1,4)
        obj_bbox[:,0::2] /= 512
        obj_bbox[:,1::2] /= 512
        obj_bbox[:,2:] += obj_bbox[:,:2]

        if len(obj_bbox) > 10:
            obj_bbox = obj_bbox[:10]
            obj_class = obj_class[:10]

        ret = dict(
            base_caption=prompt,
            obj_bbox=obj_bbox,
            obj_class=obj_class,
        )
        return ret
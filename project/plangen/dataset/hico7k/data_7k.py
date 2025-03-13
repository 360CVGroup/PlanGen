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

class Dataset_7k(Dataset):
    def __init__(
        self,
        args=None,
        is_test=False,
    ):
        self.args = args
        self.is_test = is_test

        datas = load_json('/home/jovyan/boomcheng-data-shcdt/chengbo/grit-20m-512-obj3-val-7k-512s-clean-v3.json')
        self.datas = datas


    def __len__(self):
        return len(self.datas)

    def __getitem__(self, i):
        data = self.datas[i]

        obj_bbox = []
        obj_class = []

        prompt = data[1]

        h = data[3]['H']
        w = data[3]['W']

        for t in data[5]:
            obj_class.append(t[0])
            obj_bbox.append(t[1])

        obj_bbox = torch.tensor(obj_bbox)
        obj_bbox[:,0::2] /= h
        obj_bbox[:,1::2] /= w


        ret = dict(
            base_caption=prompt,
            obj_bbox=obj_bbox,
            obj_class=obj_class,
        )
        return ret
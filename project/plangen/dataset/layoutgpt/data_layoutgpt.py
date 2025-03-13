import sys;sys.path.insert(0, '/home/jovyan/boomcheng-data-shcdt/herunze/code/base/')
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

# three_party/LayoutGPT/dataset/NSR-1K/counting/counting.train.json
# three_party/LayoutGPT/dataset/NSR-1K/spatial/spatial.train.json

class Dataset_layout(Dataset):
    def __init__(
        self,
        args=None,
        is_test=False,
    ):
        self.args = args
        self.is_test = is_test

        datas_counting = load_json('three_party/LayoutGPT/dataset/NSR-1K/counting/counting.train.json') + load_json('three_party/LayoutGPT/dataset/NSR-1K/counting/counting.val.json')#38k
        datas_spatial = load_json('three_party/LayoutGPT/dataset/NSR-1K/spatial/spatial.train.json') + load_json('three_party/LayoutGPT/dataset/NSR-1K/spatial/spatial.val.json')#0.7k

        items = []
        for data in datas_counting:
            object_list = data['object_list']
            clas = [t[0] for t in object_list]
            bboxes = torch.tensor([t[1] for t in object_list])
            new_bboxes = self.convert_box(bboxes)
            items.append(dict(
                base_caption=data['prompt'],
                obj_bbox=new_bboxes,
                obj_class=clas,
            ))
        for data in datas_spatial:
            object_list = [data['obj1'], data['obj2']]
            clas = [t[0] for t in object_list]
            bboxes = torch.tensor([t[1] for t in object_list])
            new_bboxes = self.convert_box(bboxes)
            # for i in range(1):
            for i in range(10):
                items.append(dict(
                    base_caption=data['prompt'],
                    obj_bbox=new_bboxes,
                    obj_class=clas,
                ))

        self.items = items

    def convert_box(self, bboxes):
        # 提取 cx, cy, _w, _h
        cx = bboxes[:, 0]
        cy = bboxes[:, 1]
        _w = bboxes[:, 2]
        _h = bboxes[:, 3]
        # x1,y1,h,w
        # x1,y1,x2,y2

        # 计算 x1, y1, x2, y2
        x1 = cx
        y1 = cy
        x2 = cx + _w
        y2 = cy + _h

        # 将结果堆叠成一个新的 Tensor，形状为 (n, 4)
        new_bboxes = torch.stack([x1, y1, x2, y2], dim=1)
        return new_bboxes

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

if __name__ == '__main__':
    data = Dataset_layout(args=None)
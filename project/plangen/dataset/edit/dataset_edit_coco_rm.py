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

class Dataset_edit_coco_rm(Dataset):
    def __init__(
        self,
        args=None,
        is_test=False,
    ):
        self.args = args
        self.is_test = is_test
        # self.datas = load_json("project/janus/dataset/edit/edit.json")

    def __len__(self):
        return 200

    def __getitem__(self, i):
        base_caption = ''

        image_path = f'/home/jovyan/boomcheng-data-shcdt/herunze/code/base/gen_data/coco_rm_200/image/{i}.png'
        mask_path = f'/home/jovyan/boomcheng-data-shcdt/herunze/code/base/gen_data/coco_rm_200/mask/{i}.png'
        box_path = f'/home/jovyan/boomcheng-data-shcdt/herunze/code/base/gen_data/coco_rm_200/box/{i}.json'
        box_new_path = f'/home/jovyan/boomcheng-data-shcdt/herunze/code/base/gen_data/coco_rm_200/box_new/{i}.json'
        image = load2ts(image_path)
        data1 = load_json(box_path)
        obj_bbox_1, obj_class_1 = data1['obj_bbox'], data1['obj_class']
        obj_bbox_1 = torch.tensor(obj_bbox_1).reshape(1,4)

        obj_bbox_edit = obj_bbox_1

        edit_region = np.array(Image.open(mask_path).resize((24,24)).convert('RGB'))[...,0]/255

        obj_bbox = torch.zeros_like(obj_bbox_1)
        obj_class = ['']*len(obj_bbox_1)

        obj_class_neg = ['an object, a person, artifacts, disharmonious objects, abrupt objects, messy background, noisy environment']
        obj_bbox_neg = obj_bbox_1

        ret = dict(
            base_caption=base_caption,
            image=image,
            image_path=image_path,
            obj_class=obj_class,
            obj_bbox=obj_bbox,
            obj_bbox_edit=obj_bbox_edit,
            obj_class_neg=obj_class_neg,
            obj_bbox_neg=obj_bbox_neg,
            edit_region=torch.tensor(edit_region), #mask or box
        )
        return ret
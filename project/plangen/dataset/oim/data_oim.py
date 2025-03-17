from packaging import version
from PIL import Image
from torchvision import transforms
import os
import PIL
from torch.utils.data import Dataset
import torchvision
import numpy as np
import torch
import random
import albumentations as A
import copy
import cv2
import pandas as pd
import glob
from torchvision.transforms import Resize
from tqdm import tqdm

from src.utils.funcs import *
from ..coco.data_coco import filter_box, resize_and_crop

def split_list(lst, n):
    """
    将列表 lst 划分为 n 等份。

    Args:
        lst (list): 需要划分的列表。
        n (int): 划分的份数。

    Returns:
        list: 包含 n 个子列表的列表。
    """
    # 计算每份的长度
    k, m = divmod(len(lst), n)
    # 使用列表生成器划分列表
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

class Dataset_oim(Dataset):
    def __init__(
        self,
        args=None,
        data_root='/home/jovyan/multi-modal-datasets/public/OID',
        split="train",
        easy_mode=False,
    ):
        self.args = args
        self.data_root = data_root
        self.split = split
        self.easy_mode = easy_mode

        self.bbox_path_list = []
        if self.split == "train":
            bboxs_path = os.path.join(data_root, 'anno', f'oidv6-train-annotations-bbox.csv')
        elif self.split == "validation":
            bboxs_path = os.path.join(data_root, 'anno', f'validation-annotations-bbox.csv')
        else:
            bboxs_path = os.path.join(data_root, 'anno', f'test-annotations-bbox.csv') # 93w

        df_bbox = pd.read_csv(bboxs_path)
        bbox_groups = df_bbox.groupby(df_bbox.LabelName)

        #anno_files = pd.read_csv('project/janus/dataset/oim/class-descriptions-boxable.csv')
        anno_files = pd.read_csv('project/plangen/dataset/oim/class-descriptions-boxable.csv')
        self.anno_dict = anno_files.set_index(anno_files.columns[0])[anno_files.columns[1]].to_dict()

        self.image_ids = df_bbox['ImageID'].unique()
        self.df_bbox = df_bbox

        if self.args.oim_split is not None:
            self.image_ids = split_list(self.image_ids, 16)[self.args.oim_split]
            self.image_ids = self.image_ids[self.args.oim_start*8:]

        print(f"openimage, len(image): {len(self.image_ids)}")


    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, i):

        image_id = self.image_ids[i]
        bboxes = self.df_bbox[self.df_bbox['ImageID']==image_id]

        img_path = os.path.join(self.data_root, self.split, f'{image_id}.jpg')

        boxes = torch.stack([
            torch.tensor(bboxes['XMin'].tolist()), 
            torch.tensor(bboxes['YMin'].tolist()), 
            torch.tensor(bboxes['XMax'].tolist()), 
            torch.tensor(bboxes['YMax'].tolist())], 
        dim=-1)

        try:
            classes = [self.anno_dict[t].lower() for t in bboxes['LabelName']]
        except:
            return self.__getitem__((i+1)%len(self))

        # print("len(boxes)")
        # print(len(boxes))
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        sorted_indices = torch.argsort(-areas)
        # import pdb;pdb.set_trace()
        boxes = boxes[sorted_indices][:10]
        classes = np.array(classes)[sorted_indices.tolist()].tolist()[:10]

        data = [img_path, boxes, classes]
        image_path, obj_bbox, obj_class = data
        obj_bbox = obj_bbox.reshape(-1,4)

        image_pil = Image.open(image_path).convert('RGB')
        w,h = image_pil.size
        obj_bbox[:,0::2] *= w
        obj_bbox[:,1::2] *= h
        obj_bbox[:,2] = obj_bbox[:,2]-obj_bbox[:,0]
        obj_bbox[:,3] = obj_bbox[:,3]-obj_bbox[:,1]

        image_pil, obj_bbox = resize_and_crop(image_pil, obj_bbox)
        image =  transforms.ToTensor()(image_pil)
        image = image*2-1
        obj_bbox, obj_class = filter_box(obj_bbox, obj_class)
        obj_bbox = obj_bbox/384
        obj_bbox = obj_bbox.reshape(-1,4)
        obj_bbox[:,2] = obj_bbox[:,0] + obj_bbox[:,2]
        obj_bbox[:,3] = obj_bbox[:,1] + obj_bbox[:,3]

        # import pdb;pdb.set_trace()
        try:
            cap = load_jsonl(f"gen_data/oim_cap2/{image_id}.jsonl")[0]
        except:
            if self.split != 'test':
                print('no caps!!!')
                save_jsonl(f"gen_data/oim_cap_nocap2/{image_id}.jsonl", [])
            cap = ''

        # if len(cap) > 400:
        #     print('len(cap)>400')
        #     # print(cap)
        #     cap = ''

        if self.easy_mode:
            return dict(
                base_caption=cap,
                image=image,
                image_path=image_path,
                # image_pil=image_pil,
                image_id=image_id,
            )

        return dict(
            base_caption=cap,
            obj_bbox=obj_bbox,
            obj_class=obj_class,
            image=image,
            image_path=image_path,
            image_pil=image_pil,
            image_id=image_id,
        )


if __name__ == '__main__':
    dataset = OpenImagesDataset()
    for data in dataset:
        import pdb;pdb.set_trace()
        pass

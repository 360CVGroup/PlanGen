import torchvision.datasets as dset
from torchvision import transforms
from pycocotools.coco import COCO
from torch.utils.data import DataLoader

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
from torch.utils.data import ConcatDataset

def resize_and_crop(image, bboxes, target_size=384):
    """
    Resize the image with the short side to target_size, then center crop to target_size x target_size.
    Adjust the bounding boxes accordingly.

    :param image: PIL Image
    :param bboxes: numpy array of shape (n, 4) where each row is [x1, y1, w, h]
    :param target_size: int, the target size for the short side and crop
    :return: resized and cropped image, adjusted bboxes
    """
    # Get original image size
    original_width, original_height = image.size
    
    # Determine the scaling factor
    if original_width < original_height:
        scale = target_size / original_width
        new_width = target_size
        new_height = int(original_height * scale)
    else:
        scale = target_size / original_height
        new_height = target_size
        new_width = int(original_width * scale)
    
    # Resize the image
    cropped_image = image.resize((new_width, new_height), Image.BILINEAR)
    
    # Calculate the coordinates for center cropping
    left = (new_width - target_size) // 2
    top = (new_height - target_size) // 2
    right = left + target_size
    bottom = top + target_size
    
    cropped_image = cropped_image.crop((left, top, right, bottom))
    
    adjusted_bboxes = []
    for bbox in bboxes:
        x1, y1, w, h = bbox
        x1_scaled = x1 * scale
        y1_scaled = y1 * scale
        w_scaled = w * scale
        h_scaled = h * scale
        
        x1_cropped = x1_scaled - left
        y1_cropped = y1_scaled - top
        
        adjusted_bboxes.append([x1_cropped, y1_cropped, w_scaled, h_scaled])
    
    return cropped_image, np.array(adjusted_bboxes)


def filter_box(all_bbox, all_class):
    image_width = image_height = 384
    filtered_bbox = []
    filtered_class = []

    for i, (x, y, w, h) in enumerate(all_bbox):
        
        # 调整框的坐标和宽高，确保它们在图像范围内
        x2 = x + w
        y2 = y + h
        x = max(0, x)
        y = max(0, y)

        if x > 380 or y > 380:
            pass
        else:
            x2 = min(384, x2)
            y2 = min(384, y2)
            # w = min(384-x2, w)
            # h = min(384-y2, h)
            w = x2 - x
            h = y2 - y

            if w*h < 200:
                pass
            else:
                filtered_bbox.append([x, y, w, h])
                filtered_class.append(all_class[i])

    # 将列表转换为 numpy 数组
    filtered_bbox = np.array(filtered_bbox)
    filtered_class = filtered_class
    return filtered_bbox, filtered_class


class Dataset_coco(Dataset):
    def __init__(
        self,
        args=None,
        is_test=False,
        split='train',
    ):
        self.args = args
        self.is_test = is_test
        self.split = split

        transform = transforms.Compose([
            # transforms.Resize(384),
            # transforms.CenterCrop(384),
            transforms.ToTensor(),
        ])

        if self.split == 'train':
            coco_data = dset.CocoDetection(
                root='/home/jovyan/multi-modal-datasets/public/coco/train2017',
                annFile='/home/jovyan/multi-modal-datasets/public/coco/annotations/instances_train2017.json',
                transform=transform
            )
        elif self.split == 'val14':
            coco_data = dset.CocoDetection(
                root='/home/jovyan/multi-modal-datasets/public/coco/val2014',
                annFile='/home/jovyan/multi-modal-datasets/public/coco/annotations/instances_val2014.json',
                transform=transform
            )
        elif self.split == 'val17':
            coco_data = dset.CocoDetection(
                root='/home/jovyan/multi-modal-datasets/public/coco/val2017',
                annFile='/home/jovyan/multi-modal-datasets/public/coco/annotations/instances_val2017.json',
                transform=transform
            )
        else:
            assert False

        self.dataset = self.coco_train = self.coco_val = coco_data

        self.coco_caps = COCO('/home/jovyan/multi-modal-datasets/public/coco/annotations/captions_val2017.json')

        # self.dataset = ConcatDataset([coco_train, coco_val])
        # self.coco_train = coco_train
        # self.coco_val = coco_val

        print(f"coco数据集大小: {len(self)}")
        # print(f"验证集大小: {len(coco_val)}")


    # 获取类别名称的函数
    def get_name(self, category_id):
        category_info = self.coco_train.coco.loadCats(category_id)[0]
        category_name = category_info['name']
        return category_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        data = self.dataset[i]
        image, annotations = data

        if len(annotations) == 0:
            print('len(anno)==0!!!')
            return self.__getitem__((i+1)%len(self)) #要不然找不到image_id
        
        obj_bbox = []
        obj_class = []
        for ano in annotations:
            bbox = ano['bbox']
            image_id = ano['image_id']
            category_id = ano['category_id']
            clas = self.get_name(category_id)
            obj_bbox.append(bbox)
            obj_class.append(clas)

        image_pil = transforms.ToPILImage()(image)
        image_pil, obj_bbox = resize_and_crop(image_pil, obj_bbox)
        image =  transforms.ToTensor()(image_pil)
        image = image*2-1
        obj_bbox, obj_class = filter_box(obj_bbox, obj_class)
        obj_bbox = obj_bbox/384
        obj_bbox = obj_bbox.reshape(-1,4)
        obj_bbox[:,2] = obj_bbox[:,0] + obj_bbox[:,2]
        obj_bbox[:,3] = obj_bbox[:,1] + obj_bbox[:,3]

        caps = self.coco_caps.imgToAnns[image_id]

        if len(caps) == 0:
            print('no caps!!!')
            cap = ''
            # cap = " and ".join([f"a {t}" for t in obj_class])
        else:
            cap = random.choice(caps)['caption']
        
        return dict(
            base_caption=cap,
            obj_bbox=obj_bbox,
            obj_class=obj_class,
            image=image,
            image_id=f"{image_id:012d}",
        )
        
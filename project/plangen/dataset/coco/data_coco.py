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
        for_rm=False,
    ):
        self.args = args
        self.is_test = is_test
        self.split = split
        self.for_rm = for_rm

        # read MSCOCO
        ann_file = '/home/jovyan/multi-modal-datasets/public/coco/annotations/instances_val2017.json'
        ann_file_captions = '/home/jovyan/multi-modal-datasets/public/coco/annotations/captions_val2017.json'
        coco_caption = COCO(ann_file_captions)
        coco = COCO(ann_file)

        # sort indices for reproducible results
        image_ids = coco.getImgIds()
        image_ids.sort()

        self.coco = coco
        self.coco_caption = coco_caption
        self.image_ids = image_ids

    # 获取类别名称的函数
    def get_name(self, category_id):
        category_info = self.coco_train.coco.loadCats(category_id)[0]
        category_name = category_info['name']
        return category_name

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, i):
        img_id = self.image_ids[i]
        # Pick one image.
        img_info = self.coco.loadImgs([img_id])[0]
        file_name = img_info['file_name']
        image_id = img_info['id']
        height = img_info['height']
        width = img_info['width']

        # Get all the annotations for the specified image.
        ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        annotations = self.coco.loadAnns(ann_ids)

        ann_ids_captions = self.coco_caption.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns_caption = self.coco_caption.loadAnns(ann_ids_captions)[0]['caption']
        
        obj_bbox = [t['bbox'] for t in annotations]

        mask = [self.coco.annToMask(t) for t in annotations]
        mask = np.stack(mask, axis=0)
        kernel = np.ones((8, 8), np.uint8)
        dilated_array = np.zeros_like(mask)
        for i in range(mask.shape[0]):
            dilated_array[i] = cv2.dilate(mask[i], kernel, iterations=5)
        mask = dilated_array
        mask = resize_pt(torch.tensor(mask), 24)

        obj_bbox = torch.tensor(obj_bbox)
        obj_bbox = obj_bbox.reshape(-1,4)
        obj_class_id = [t['category_id'] for t in annotations]
        obj_class = [cat["name"] for cat in self.coco.loadCats(obj_class_id)]
        assert len(obj_class) == len(obj_bbox)

        image_path = osp.join('/home/jovyan/multi-modal-datasets/public/coco/val2017', f"{image_id:012d}.jpg")
        image = to_ts(Image.open(image_path).convert('RGB').resize((384,384)))
        image = image*2-1
        obj_bbox[:,0::2] /= width
        obj_bbox[:,1::2] /= height
        obj_bbox[:,2] += obj_bbox[:,0]
        obj_bbox[:,3] += obj_bbox[:,1]

        # if self.for_rm:
        #     i = random.randint(0, len(obj_bbox))
        #     obj_bbox = obj_bbox[i:i+1]
        #     obj_class = obj_class[i:i+1]
        #     mask = mask[i:i+1]

        return dict(
            base_caption=anns_caption,
            obj_bbox=obj_bbox,
            obj_class=obj_class,
            image=image,
            image_id=f"{image_id:012d}",
            H=height,
            W=width,
            mask=mask,
        )
        
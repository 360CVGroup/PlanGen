import sys;sys.path.insert(0, './')
import torchvision.datasets as dset
from torchvision import transforms
from pycocotools.coco import COCO
from torch.utils.data import DataLoader


import torchvision.datasets as dset
from torchvision import transforms
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from src.utils.funcs import *


# 设置图像转换，通常是将图像调整为固定的大小并转换为张量
transform = transforms.Compose([
    # transforms.Resize(384),  # 示例尺寸
    # transforms.CenterCrop(384),
    transforms.ToTensor(),
])

# 加载训练集
# coco_train = dset.CocoDetection(
#     root='/home/jovyan/multi-modal-datasets/public/coco/train2017',
#     annFile='/home/jovyan/multi-modal-datasets/public/coco/annotations/instances_train2017.json',
#     transform=transform
# )

# 加载验证集
coco_val = dset.CocoDetection(
    root='/home/jovyan/multi-modal-datasets/public/coco/val2017',
    annFile='/home/jovyan/multi-modal-datasets/public/coco/annotations/instances_val2017.json',
    transform=transform
)

# 初始化COCO API用于读取caption标注
coco_caps = COCO('/home/jovyan/multi-modal-datasets/public/coco/annotations/captions_val2017.json')

# 获取类别名称的函数
def get_name(category_id):
    category_info = coco_val.coco.loadCats(category_id)[0]
    category_name = category_info['name']
    return category_name

# 获取caption的函数
def get_cap(image_id):
    annIds = coco_caps.getAnnIds(imgIds=image_id)
    anns = coco_caps.loadAnns(annIds)
    return [ann['caption'] for ann in anns]

# 打印数据集大小
# print(f"训练集大小: {len(coco_train)}")
print(f"验证集大小: {len(coco_val)}")

from PIL import Image
import torchvision.transforms.functional as F
import torch

from PIL import Image
import numpy as np
from PIL import Image
import numpy as np

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
    filtered_class = np.array(filtered_class)
    return filtered_bbox, filtered_class


# 示例用法
# image_path = 'path_to_your_image.jpg'
# image = Image.open(image_path).convert("RGB")
# bboxes = torch.tensor([[40, 50, 100, 120], [60, 70, 80, 90]])  # 示例bbox [(x_min, y_min, width, height), ...]

# cropped_image, adjusted_bboxes = resize_short_side_and_center_crop(image, bboxes)

# print(f"Adjusted bboxes:\n{adjusted_bboxes}")
# 遍历训练集
for data in coco_val:
    image, annotations = data
    if len(annotations) == 0:
        continue  # 跳过没有标注的图像

    # 获取第一个标注的信息
    # 将图像从张量转换为PIL图像


    all_bbox = []
    all_class = []
    for ano in annotations:
        category_id = ano['category_id']
        bbox = ano['bbox']
        image_id = ano['image_id']

        # 打印caption和类别名称
        print(f"Image ID: {image_id}")
        # print(f"Captions: {get_cap(image_id)}")
        captions = coco_caps.imgToAnns[image_id]
        print(f"Category Name: {get_name(category_id)}")

        all_class.append(get_name(category_id))
        # all_bbox.append(bbox)
    
    all_bbox = [t['bbox'] for t in annotations]

    all_bbox = torch.tensor(all_bbox).reshape(-1,4)
    image_pil = transforms.ToPILImage()(image)
    # import pdb;pdb.set_trace()
    image_pil, all_bbox = resize_and_crop(image_pil, all_bbox)

    all_bbox,all_class = filter_box(all_bbox,all_class)
    # all_bbox, all_class
    # image_pil = transforms.ToPILImage()(image)

    # 创建画布
    fig, ax = plt.subplots(1)
    ax.imshow(image_pil)

    for (bbox, name) in zip(all_bbox, all_class):
        x, y, width, height = bbox
        rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # 添加类别名称
        plt.text(x, y, name, color='red', fontsize=12, backgroundcolor='white')

        # 显示图像
    plt.axis('off')
    plt.savefig('a.png')
    import pdb;pdb.set_trace()

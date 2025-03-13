from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import ast
import numpy as np
from src.utils.funcs import *
from ..coco.data_coco import filter_box, resize_and_crop

def adjust_and_normalize_bboxes(bboxes, orig_width, orig_height):
    normalized_bboxes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        x1_norm = round(x1 / orig_width,3)  
        y1_norm = round(y1 / orig_height,3)
        x2_norm = round(x2 / orig_width,3)
        y2_norm = round(y2 / orig_height,3)
        normalized_bboxes.append([x1_norm, y1_norm, x2_norm, y2_norm])
    
    return normalized_bboxes
    
class BboxDataset_sam(Dataset):
    def __init__(self, dataset, resolution=1024, is_testset=False,):
        self.is_testset = is_testset
        self.dataset = dataset
        self.resolution = resolution
        if self.is_testset:
            self.transform = transforms.Compose([
                transforms.Resize(
                    (resolution,resolution), interpolation=transforms.InterpolationMode.BILINEAR 
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.dataset)

    def update_item(self, item):
        #dict_keys(['bbox_info', 'global_caption', 'image_info'])
        # item.update(image=x)
        dirname, filename = item['image_path'][3:].split('/')
        image_path = osp.join('/home/jovyan/myh-data-ceph-shcdt-1/data/SAM/', str(int(dirname)), filename)
        image = Image.open(image_path).convert('RGB')
        bbox_info = item['metadata']['bbox_info']
        global_caption = item['metadata']['global_caption']
        image_info = item['metadata']['image_info']

        height = image_info['height']
        width = image_info['width']
        file_name = image_info['file_name']

        bbox_list = []
        region_captions = []
        detail_region_captions = []
        for box in bbox_info:
            bbox_list.append(box['bbox'])
            region_captions.append(box['description'])
            detail_region_captions.append(box['detail_description'])

        item.update(dict(
            global_caption=global_caption,
            image=image,
            height=height,
            width=width,
            file_name=file_name,
            bbox_list=str(bbox_list),
            region_captions=str(region_captions),
            detail_region_captions=str(detail_region_captions),
        ))

    def __getitem__(self, idx):
        item = self.dataset[idx]

        if self.is_testset:
            pass
        else:
            self.update_item(item)

        image = item['image']
        image = self.transform(image)
        # import pdb;pdb.set_trace()

        height = int(item['height'])
        width = int(item['width'])
        global_caption = item['global_caption']
        region_bboxes_list = item['bbox_list']
        detail_region_caption_list = item['detail_region_captions']
        region_caption_list = item['region_captions']
        file_name = item['file_name']

        region_bboxes_list = ast.literal_eval(region_bboxes_list)
        region_bboxes_list = adjust_and_normalize_bboxes(region_bboxes_list,width,height)
        region_bboxes_list = np.array(region_bboxes_list, dtype=np.float32)

        region_caption_list = ast.literal_eval(region_caption_list)
        detail_region_caption_list = ast.literal_eval(detail_region_caption_list)

        if self.is_testset:
            pass
        else:
            image_pil = to_pil(image)
            obj_bbox = region_bboxes_list * [width, height, width, height]
            obj_class = detail_region_caption_list

            obj_bbox[:,2] = obj_bbox[:,2] - obj_bbox[:,0]
            obj_bbox[:,3] = obj_bbox[:,3] - obj_bbox[:,1]
            image_pil, obj_bbox = resize_and_crop(image_pil, obj_bbox)
            image =  to_ts(image_pil)
            image = image*2-1
            obj_bbox, obj_class = filter_box(obj_bbox, obj_class)
            obj_bbox = obj_bbox/384
            obj_bbox = obj_bbox.reshape(-1,4)
            obj_bbox[:,2] = obj_bbox[:,0] + obj_bbox[:,2]
            obj_bbox[:,3] = obj_bbox[:,1] + obj_bbox[:,3]
            
            region_bboxes_list = obj_bbox
            detail_region_caption_list = obj_class
        
        if None in detail_region_caption_list:
            detail_region_caption_list = region_caption_list
        if None in region_caption_list:
            return self.__getitem__(self, idx+1)

        return {
            'image': image,
            'global_caption': global_caption,
            'detail_region_caption_list': detail_region_caption_list,
            'region_bboxes_list': region_bboxes_list,
            'region_caption_list': region_caption_list,
            'file_name': file_name,
            'height': height,
            'width': width
        }


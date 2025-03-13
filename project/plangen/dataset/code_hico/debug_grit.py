#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from collections import defaultdict
import random
import PIL
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

# 原文件：
#from layout_diffusion.dataset.util import image_normalize
#from layout_diffusion.dataset.augmentations import RandomSampleCrop, RandomMirror
from .dataset.util import image_normalize
from .dataset.augmentations import RandomMirror, RandomSampleCrop, CenterSampleCrop

# 修改的
#from LayoutDiffusion.layout_diffusion.dataset.util import image_normalize
#from LayoutDiffusion.layout_diffusion.dataset.augmentations import RandomMirror, RandomSampleCrop
#CenterSampleCrop=RandomSampleCrop## 没有CenterSampleCrop，先用RandomSampleCrop代替




import pdb


Image.MAX_IMAGE_PIXELS = None

class GritSceneGraphDataset(Dataset):
    def __init__(self, list_tokenizers, grit_json, 
                 image_dir, instances_json, stuff_json=None,
                 stuff_only=True, image_size=(64, 64), mask_size=16,
                 max_num_samples=None,proportion_empty_prompts=0.05,
                 include_relationships=True, min_object_size=0.02,
                 min_objects_per_image=3, max_objects_per_image=8, left_right_flip=False,
                 include_other=False, instance_whitelist=None, stuff_whitelist=None, mode='train',
                 use_deprecated_stuff2017=False, deprecated_coco_stuff_ids_txt='', filter_mode='LostGAN',
                 use_MinIoURandomCrop=False,
                 return_origin_image=False, specific_image_ids=None,
                 args=None,
                 ):
        """
        A PyTorch Dataset for loading Coco and Coco-Stuff annotations and converting
        them to scene graphs on the fly.

        Inputs:
        - image_dir: Path to a directory where images are held
        - instances_json: Path to a JSON file giving COCO annotations
        - stuff_json: (optional) Path to a JSON file giving COCO-Stuff annotations
        - stuff_only: (optional, default True) If True then only iterate over
          images which appear in stuff_json; if False then iterate over all images
          in instances_json.
        - image_size: Size (H, W) at which to load images. Default (64, 64).
        - mask_size: Size M for object segmentation masks; default 16.
        - max_num_samples: If None use all images. Other wise only use images in the
          range [0, max_num_samples). Default None.
        - include_relationships: If True then include spatial relationships; if
          False then only include the trivial __in_image__ relationship.
        - min_object_size: Ignore objects whose bounding box takes up less than
          this fraction of the image.
        - min_objects_per_image: Ignore images which have fewer than this many
          object annotations.
        - max_objects_per_image: Ignore images which have more than this many
          object annotations.
        - include_other: If True, include COCO-Stuff annotations which have category
          "other". Default is False, because I found that these were really noisy
          and pretty much impossible for the system to model.
        - instance_whitelist: None means use all instance categories. Otherwise a
          list giving a whitelist of instance category names to use.
        - stuff_whitelist: None means use all stuff categories. Otherwise a list
          giving a whitelist of stuff category names to use.
        """
        super(Dataset, self).__init__()

        self.args = args

        self.return_origin_image = return_origin_image
        if self.return_origin_image:
            self.origin_transform = T.Compose([
                T.ToTensor(),
                image_normalize()
            ])

        if stuff_only and stuff_json is None:
            print('WARNING: Got stuff_only=True but stuff_json=None.')
            print('Falling back to stuff_only=False.')

        self.proportion_empty_prompts = proportion_empty_prompts
        self.use_deprecated_stuff2017 = use_deprecated_stuff2017
        self.deprecated_coco_stuff_ids_txt = deprecated_coco_stuff_ids_txt
        self.mode = mode
        self.max_objects_per_image = max_objects_per_image
        self.image_dir = image_dir
        self.mask_size = mask_size
        self.max_num_samples = max_num_samples
        self.include_relationships = include_relationships
        self.filter_mode = filter_mode
        self.image_size = image_size
        self.min_image_size = min(self.image_size)
        self.min_object_size = min_object_size
        self.left_right_flip = left_right_flip
        if left_right_flip:
            self.random_flip = RandomMirror()

        self.layout_length = self.max_objects_per_image + 2

        self.use_MinIoURandomCrop = use_MinIoURandomCrop
        #self.use_MinIoURandomCrop = False
        if use_MinIoURandomCrop:
            self.MinIoURandomCrop = RandomSampleCrop()
            self.MinIoUCenterCrop = CenterSampleCrop()

        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize(size=image_size, antialias=True),
            image_normalize()
        ])

        self.transform_cond = T.Compose([
            T.ToTensor(),
            #T.Resize(size=image_size, antialias=True),
            #image_normalize()
        ])

        self.total_num_bbox = 0
        self.total_num_invalid_bbox = 0

        #self.tokenizers = tokenizers
        self.tokenizers_one, self.tokenizers_two = list_tokenizers

        # read grit-20m data
        with open(grit_json, 'r') as f:
            grit_data = json.load(f)

        self.image_ids = []
        self.image_id_to_objects = {}
        for idx, obj_data in grit_data.items():
            f_img_path = obj_data["f_path"]
            #list_chunks = obj_data["noun_chunks"]
            list_exps = obj_data["ref_exps"]
            image_w = obj_data["width"]
            image_h = obj_data["height"]
            caption = obj_data["caption"]
            url = obj_data["url"]

            obj_nums = len(list_exps)
            # get sub-caption
            list_bbox_info = []
            for box_info in list_exps:
                phrase_s, phrase_e, x1_norm, y1_norm, x2_norm, y2_norm, score = box_info
                phrase_s = int(phrase_s)
                phrase_e = int(phrase_e)
                phrase = caption[phrase_s:phrase_e]
                x1, y1, x2, y2 = int(x1_norm * image_w), int(y1_norm * image_h), int(x2_norm * image_w), int(y2_norm * image_h)

                x1, y1 = min(x1, image_w), min(y1, image_h)
                x2, y2 = min(x2, image_w), min(y2, image_h)
                if int(x2 - x1) < 0.05 * image_w or int(y2 - y1) < 0.05 * image_h:
                    continue
                
                #list_bbox_info.append([phrase, [x1, y1, x2, y2]])
                list_bbox_info.append([phrase, [x1, y1, int(x2 - x1), int(y2 - y1)]])
                if len(list_bbox_info) >= self.max_objects_per_image:
                    break
            if len(list_bbox_info) == 0:
                continue

            self.image_ids.append([idx, f_img_path, obj_nums])
            self.image_id_to_objects.setdefault(idx, [caption, image_w, image_h, list_bbox_info, url])

        print ("data nums : %s." % len(self.image_id_to_objects))

    def filter_invalid_bbox(self, H, W, bbox, is_valid_bbox, verbose=False):
        #pdb.set_trace()
        for idx, obj_bbox in enumerate(bbox):
            if not is_valid_bbox[idx]:
                continue
            self.total_num_bbox += 1

            x, y, w, h = obj_bbox

            if (x >= W) or (y >= H):
                is_valid_bbox[idx] = False
                self.total_num_invalid_bbox += 1
                if verbose:
                    print(
                        'total_num = {}, invalid_num = {}, x = {}, y={}, w={}, h={}, W={}, H={}'.format(
                            self.total_num_bbox, self.total_num_invalid_bbox, x, y, w, h, W, H,
                        )
                    )
                continue

            x0, y0, x1, y1 = x, y, x + w, y + h
            x1 = np.clip(x + w, 1, W)
            y1 = np.clip(y + h, 1, H)

            if (y1 - y0 < self.min_object_size * H) or (x1 - x0 < self.min_object_size * W):
                is_valid_bbox[idx] = False
                self.total_num_invalid_bbox += 1
                if verbose:
                    print(
                        'total_num = {}, invalid_num = {}, x = {}, y={}, w={}, h={}, W={}, H={}'.format(
                            self.total_num_bbox, self.total_num_invalid_bbox, x, y, w, h, W, H,
                        )
                    )
                continue
            bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3] = x0, y0, x1, y1

        return bbox, is_valid_bbox

    def total_objects(self):
        total_objs = 0
        for i, image_info in enumerate(self.image_ids):
            total_objs += image_info[2]
        return total_objs

    def get_init_meta_data(self, image_id, caption):
        #self.layout_length = self.max_objects_per_image + 2
        #pdb.set_trace()
        layout_length = self.layout_length
        #clip_text_ids = self.tokenize_caption("")
        list_clip_text_ids = self.tokenize_caption("")
        meta_data = {
            'obj_bbox': torch.zeros([layout_length, 4]),
            'obj_class': [""] * layout_length,
            'is_valid_obj': torch.zeros([layout_length]),
            'upd_is_valid_obj': torch.zeros([layout_length]),
            #'obj_class_text_ids': clip_text_ids.repeat(layout_length, 1),
            'obj_class_text_ids': [list_clip_text_ids] * layout_length,
            #'obj_class': torch.LongTensor(layout_length).fill_(self.vocab['object_name_to_idx']['__null__']),
            #'filename': self.image_id_to_filename[image_id].replace('/', '_').split('.')[0]
        }

        # The first object will be the special __image__ object
        meta_data['obj_bbox'][0] = torch.FloatTensor([0, 0, 1, 1])
        #meta_data['obj_class'][0] = self.vocab['object_name_to_idx']['__image__']
        meta_data['obj_class'][0] = caption
        meta_data['is_valid_obj'][0] = 1.0
        meta_data['upd_is_valid_obj'][0] = 1.0

        #clip_text_ids = self.tokenize_caption(caption)
        #meta_data['obj_class_text_ids'][0] = clip_text_ids
        list_clip_text_ids = self.tokenize_caption(caption)
        meta_data['obj_class_text_ids'][0] = list_clip_text_ids

        return meta_data

    def load_image(self, image_path):
        with open(image_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                image = image.convert('RGB')
        return image

    def __len__(self):
        return len(self.image_ids)

    def tokenize_caption(self, caption):
        captions = []
        #if random.random() < 0.05:
        if random.random() < self.proportion_empty_prompts:
            captions.append("")
        else:
            captions.append(caption)
        clip_inputs_one = self.tokenizers_one(
            captions, max_length = self.tokenizers_one.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        clip_inputs_two = self.tokenizers_two(
            captions, max_length = self.tokenizers_two.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return [clip_inputs_one.input_ids, clip_inputs_two.input_ids]

    def resize_img(self, image, obj_bbox, obj_class):
        # resize to self.resolution
        #pdb.set_trace()
        ori_width, ori_height = image.size
        #target_size = self.image_size
        #res_height, res_width = self.image_size
        res_min_size = self.min_image_size
        if ori_height < ori_width:
            resize_height = res_min_size
            aspect_r = ori_width / ori_height
            resize_width = int(resize_height * aspect_r)
            im_resized = image.resize((resize_width, resize_height))

            rescale = resize_height / ori_height
            re_obj_bbox = obj_bbox * rescale
        else:
            resize_width = res_min_size
            aspect_r = ori_height / ori_width
            resize_height = int(resize_width * aspect_r)
            im_resized = image.resize((resize_width, resize_height))

            rescale = resize_height / ori_height
            re_obj_bbox = obj_bbox * rescale

        return im_resized, re_obj_bbox, obj_class

    def draw_image(self, image, obj_bbox, obj_class, img_save):
        dw_img = PIL.Image.fromarray(np.uint8(image * 255))
        draw = PIL.ImageDraw.Draw(dw_img)
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        #draw.rectangle([100, 100, 300, 300], outline = (0, 255, 255), fill = (255, 0, 0), width = 10)
        for iix in range(len(obj_bbox)):
            rec = obj_bbox[iix]
            d_rec = [int(xx) for xx in rec]
            draw.rectangle(d_rec, outline = color, width = 3)

            text = obj_class[iix]
            font = ImageFont.truetype("/home/jovyan/boomcheng-data/tools/font/msyh.ttf", size=10)
            draw.text((d_rec[0], d_rec[1]), text, font = font, fill="red", align="left")
        dw_img.save(img_save)

    def draw_image_xywh(self, image, obj_bbox, obj_class, img_save):
        dw_img = PIL.Image.fromarray(np.uint8(image * 255))
        draw = PIL.ImageDraw.Draw(dw_img)
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        #draw.rectangle([100, 100, 300, 300], outline = (0, 255, 255), fill = (255, 0, 0), width = 10)
        for iix in range(len(obj_bbox)):
            rec = obj_bbox[iix]
            d_rec = [int(xx) for xx in rec]
            d_rec[2] += d_rec[0]
            d_rec[3] += d_rec[1]
            draw.rectangle(d_rec, outline = color, width = 3)

            text = obj_class[iix]
            font = ImageFont.truetype("/home/jovyan/boomcheng-data/tools/font/msyh.ttf", size=10)
            draw.text((d_rec[0], d_rec[1]), text, font = font, fill="red", align="left")
        dw_img.save(img_save)


    def __getitem__(self, index):
        """
        Get the pixels of an image, and a random synthetic scene graph for that
        image constructed on-the-fly from its COCO object annotations. We assume
        that the image will have height H, width W, C channels; there will be O
        object annotations, each of which will have both a bounding box and a
        segmentation mask of shape (M, M). There will be T triples in the scene
        graph.

        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - objs: LongTensor of shape (O,)
        - boxes: FloatTensor of shape (O, 4) giving boxes for objects in
          (x0, y0, x1, y1) format, in a [0, 1] coordinate system
        - masks: LongTensor of shape (O, M, M) giving segmentation masks for
          objects, where 0 is background and 0 is object.

        """
        f_idx, f_image_path, f_obj_nums = self.image_ids[index]

        image = self.load_image(f_image_path)
        W, H = image.size
        caption, image_w, image_h, list_bbox_info, url = self.image_id_to_objects[f_idx]

        if W != image_w or H != image_h:
            index = 0
            f_idx, f_image_path, f_obj_nums = self.image_ids[index]
            image = self.load_image(f_image_path)
            caption, image_w, image_h, list_bbox_info, url = self.image_id_to_objects[f_idx]
            

        f_img_nm = f_image_path.split("/")[-1]
        #image.save("./image_demo/%s-s0-base.jpg" % f_img_nm)
        #print (f_img_nm, image_w, image_h)

        num_obj = len(list_bbox_info)
        obj_bbox = [obj[1] for obj in list_bbox_info]   # [x, y, w, h]
        obj_bbox = np.array(obj_bbox)
        obj_class = [obj[0] for obj in list_bbox_info]
        is_valid_obj = [True for _ in range(num_obj)]

        #pdb.set_trace()
        #self.draw_image_xywh(np.array(image, dtype=np.float32) / 255.0, obj_bbox, obj_class, "./image_demo/%s-s0-base-bbox.jpg" % f_img_nm)

        #pdb.set_trace()
        # filter invalid bbox
        # bbox : [x, y, w, h] -> [x1, y1, x2, y2]
        if True:
            W, H = image.size
            obj_bbox, is_valid_obj = self.filter_invalid_bbox(H=H, W=W, bbox=obj_bbox, is_valid_bbox=is_valid_obj)

        if True:
            image, obj_bbox, obj_class = self.resize_img(image, obj_bbox, obj_class)

        # import pdb;pdb.set_trace()
        # "a.png"
        # if self.args.debug_data:
        #     # image.save("./image_demo/%s-s1-resize-out.jpg" % f_img_nm)
        #     self.draw_image(np.array(image, dtype=np.float32) / 255.0, obj_bbox, obj_class, "./image_demo/%s-s1-resize-bbox.jpg" % f_img_nm)

        if self.return_origin_image:
            origin_image = np.array(image, dtype=np.float32) / 255.0
        image = np.array(image, dtype=np.float32) / 255.0

        # self.draw_image(image, obj_bbox, obj_class, f"./a_{f_img_nm}.jpg")
        # a_000000945.jpg.jpg

        H, W, _ = image.shape
        # get meta data
        #pdb.set_trace()
        meta_data = self.get_init_meta_data(f_idx, caption)
        #meta_data['width'], meta_data['height'] = W, H
        meta_data['width'], meta_data['height'] = image_w, image_h
        meta_data['original_sizes_hw'] = (image_h, image_w)
        meta_data['num_obj_ori'] = num_obj

        #pdb.set_trace()
        for iid in range(len(is_valid_obj)):
            meta_data['is_valid_obj'][1+iid] = is_valid_obj[iid]

        # flip
        #if False:
        if self.left_right_flip and random.random() < 0.5:
            image, obj_bbox, obj_class = self.random_flip(image, obj_bbox, obj_class)
        
        base_class = obj_class
        base_bbox = obj_bbox
        base_image = PIL.Image.fromarray(np.uint8(image * 255))

        #pdb.set_trace()
        #self.draw_image(image, obj_bbox, obj_class, "./image_demo/%s-s2-flip-bbox.jpg" % f_img_nm)

        # random crop image and its bbox
        #if False:
        #pdb.set_trace()
        crop_top_left = (0,0)
        if self.use_MinIoURandomCrop:#true
            r_obj_bbox = obj_bbox[is_valid_obj]
            r_obj_class = [obj_class[ii] for ii in range(len(is_valid_obj)) if is_valid_obj[ii]]

            #try:
            if True:
                crop_top_left, image, upd_obj_bbox, upd_obj_class, upd_is_valid_obj = self.MinIoUCenterCrop(image, r_obj_bbox, r_obj_class)
            #except:
            #    print (f"=======================, index:{index}, f_idx:{f_idx}")
            #    return self.__getitem__(0)
                

            meta_data['new_height'] = image.shape[0]
            meta_data['new_width'] = image.shape[1]
            H, W, _ = image.shape
        else:
            #### add
            upd_is_valid_obj = is_valid_obj
            upd_obj_bbox = obj_bbox
            upd_obj_class = obj_class
            # upd_is_valid_obj = [1]*len(r_obj_bbox)

        meta_data["crop_top_lefts"] = crop_top_left     # (x, y)
        for iid in range(len(upd_is_valid_obj)):
            meta_data['upd_is_valid_obj'][1+iid] = int(upd_is_valid_obj[iid])

        obj_bbox, obj_class = upd_obj_bbox, upd_obj_class
        #self.draw_image(image, obj_bbox, obj_class, "./image_demo/%s-s3-crop-bbox.jpg" % f_img_nm)

        #pdb.set_trace()
        H, W, C = image.shape
        ############### condition_image #############
        list_cond_image = []
        cond_image = np.zeros_like(image, dtype=np.uint8)
        list_cond_image.append(cond_image)
        for iit in range(len(obj_bbox)):
            dot_bbox = obj_bbox[iit]
            dx1, dy1, dx2, dy2 = [int(xx) for xx in dot_bbox]
            cond_image = np.zeros_like(image, dtype=np.uint8)
            #cond_image[dy1:dy2, dx1:dx2] = 255
            cond_image[dy1:dy2, dx1:dx2] = 1
            list_cond_image.append(cond_image)

            ##print (dot_bbox, image.shape, obj_class[iit])
            #im = PIL.Image.fromarray(cond_image*255)
            #im.save("./image_demo/%s-cond-%s.jpg" % (f_img_nm, iit))

        # PIL.Image.fromarray(np.uint8(meta_data['cond_image'][0])).save('a0.jpg')

        #obj_bbox = torch.FloatTensor(obj_bbox[is_valid_obj])
        #obj_class = [obj_class[iv] for iv in range(len(is_valid_obj)) if is_valid_obj[iv]]
        obj_bbox = torch.FloatTensor(obj_bbox)

        obj_bbox[:, 0::2] = obj_bbox[:, 0::2] / W
        obj_bbox[:, 1::2] = obj_bbox[:, 1::2] / H

        num_selected = min(obj_bbox.shape[0], self.max_objects_per_image)
        selected_obj_idxs = random.sample(range(obj_bbox.shape[0]), num_selected)#[2, 0, 1]

        meta_data['obj_bbox'][1:1 + num_selected] = obj_bbox[selected_obj_idxs]
        list_text_select = [obj_class[iv] for iv in selected_obj_idxs]
        meta_data['obj_class'][1:1 + num_selected] = list_text_select #['Pink Vans, with pink roses on the outer side', 'Pink Vans, with pink roses on the outer side', 'pink roses on the outer side', 'pink roses on the outer side', '', '', '', '', '', '']

        obj_cond_image = np.stack(list_cond_image, axis=0)
        meta_data['cond_image'] = np.zeros([self.layout_length, H, W, C])
        meta_data['cond_image'][0:len(list_cond_image)] = obj_cond_image
        # torch_resize = Resize([256,256])
        # torch_resize(torch.from_numpy(meta_data['cond_image'].transpose(0,3,1,2))).permute(0,2,3,1)
        ##meta_data['cond_image'] = self.transform_cond(meta_data['cond_image'])

        #meta_data['cond_image'] = torch.from_numpy(meta_data['cond_image'].transpose(0,3,1,2)).permute(0,2,3,1)
        #meta_data['cond_image'][1:1 + num_selected] = torch.from_numpy(obj_cond_image[1:][selected_obj_idxs])
        meta_data['cond_image'][1:1 + num_selected] = obj_cond_image[1:][selected_obj_idxs]
        meta_data['cond_image'] = torch.from_numpy(meta_data['cond_image'].transpose(0,3,1,2))

        #pdb.set_trace()
        # if self.args is not None and self.args.debug_data:
        #     self.draw_image(image,  np.uint8(obj_bbox.numpy() * (self.min_image_size-1)), obj_class, "./image_demo/%s-base.jpg" % f_img_nm)
        #for iii in range(num_selected):
        #    PIL.Image.fromarray(np.uint8(meta_data['cond_image'][1+iii] * 255)).save("./image_demo/%s-cond-%s.jpg" % (f_img_nm, iii))
            
        list_clip_text_ids = self.tokenize_caption(caption)
        meta_data['base_caption'] = caption
        meta_data['base_class_text_ids'] = list_clip_text_ids

        #meta_data['is_valid_obj'][1:1 + num_selected] = 1.0
        meta_data['num_selected'] =  1 + num_selected
        meta_data['url'] = url
        #meta_data['num_obj_select'] = num_selected

        # tokenizer
        #pdb.set_trace()
        for iit in range(len(list_text_select)):
            text = list_text_select[iit]
            list_clip_text_ids = self.tokenize_caption(text)
            meta_data['obj_class_text_ids'][1+iit] = list_clip_text_ids

        if self.return_origin_image:
            meta_data['origin_image'] = self.origin_transform(origin_image)

        #trans_image = base_image
        #tmp_save_name = "./image_demo/%s-s4-ori-draw-crop.jpg" % f_img_nm
        #crop_bbox = [crop_top_left[0], crop_top_left[1], crop_top_left[0] + 512,crop_top_left[1]+512]
        #self.draw_image(np.array(trans_image, dtype=np.float32) / 255.0, [crop_bbox], [caption], tmp_save_name)

        ## upd_is_valid_obj
        #tmp_save_name = "./image_demo/%s-s5-ori-draw-crop-bbox.jpg" % f_img_nm
        ##pdb.set_trace()
        #trans_image = self.load_image(tmp_save_name)
        #upd_is_valid_obj = [int(x) for x in upd_is_valid_obj]
        #upd_base_bbox = []
        #upd_base_class = []
        ##upd_base_class.append(caption)
        ##crop_bbox = [crop_top_left[0], crop_top_left[1], crop_top_left[0] + 512,crop_top_left[1]+512]
        ##upd_base_bbox.append(crop_bbox)
        #for iid in range(len(upd_is_valid_obj)):
        #    if upd_is_valid_obj[iid]:
        #        upd_base_class.append(base_class[iid])
        #        upd_base_bbox.append(base_bbox[iid])
        #self.draw_image(np.array(trans_image, dtype=np.float32) / 255.0, upd_base_bbox, upd_base_class, tmp_save_name)

        #return self.transform(image), meta_data
        meta_data["pixel_values"] = self.transform(image)

        meta_data["image_path"] = f_image_path

        # meta_data["image_path"] = 
        return meta_data
        #return self.transform(image), meta_data

def grit_collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()   # [bs, 3, 512, 512]

    layo_cond_image = [example["cond_image"] for example in examples]   # bs, [10, 3, 512, 512]
    layo_cond_image = torch.stack(layo_cond_image)  # [bs, 10, 3, 512, 512]
    layo_cond_image = layo_cond_image.to(memory_format=torch.contiguous_format)

    original_sizes = [example["original_sizes_hw"] for example in examples] # bs
    crop_top_lefts = [example["crop_top_lefts"] for example in examples]    # bs
    base_caption = [example["base_caption"] for example in examples]        # bs
    num_selected = [example["num_selected"] for example in examples]

    base_input_ids_one = torch.concat([example["base_class_text_ids"][0] for example in examples])  # bs, 77
    base_input_ids_two = torch.concat([example["base_class_text_ids"][1] for example in examples])  # bs, 77

    list_input_ids_one = []
    list_input_ids_two = []
    for example in examples:
        list_input_text_ids = example['obj_class_text_ids']
        clip_input_ids_one = torch.concat([x[0] for x in list_input_text_ids])  # [10, 77]
        clip_input_ids_two = torch.concat([x[1] for x in list_input_text_ids])  # [10, 77]
        list_input_ids_one.append(clip_input_ids_one)
        list_input_ids_two.append(clip_input_ids_two)

    layo_input_ids_one = torch.stack(list_input_ids_one)    # bs, 10, 77
    layo_input_ids_two = torch.stack(list_input_ids_two)    # bs, 10, 77

    out_data = {
        "pixel_values": pixel_values,
        "cond_image": layo_cond_image,
        "original_sizes_hw": original_sizes,
        "crop_top_lefts": crop_top_lefts,
        "num_selected": num_selected,
        #"base_caption": base_caption,
        "base_input_ids_one": base_input_ids_one,
        "base_input_ids_two": base_input_ids_two,
        "layo_input_ids_one": layo_input_ids_one,
        "layo_input_ids_two": layo_input_ids_two,
    }
    return out_data

def grit_collate_fn_for_layout(batch):
    all_meta_data = defaultdict(list)
    all_imgs = []

    #pdb.set_trace()
    for i, (img, meta_data) in enumerate(batch):
        all_imgs.append(img[None])
        for key, value in meta_data.items():
            all_meta_data[key].append(value)

    all_imgs = torch.cat(all_imgs)
    for key, value in all_meta_data.items():
        #if key in ['obj_bbox', 'obj_class', 'is_valid_obj'] or key.startswith('labels_from_layout_to_image_at_resolution'):
        if key in ['obj_bbox'] or key.startswith('labels_from_layout_to_image_at_resolution'):
            all_meta_data[key] = torch.stack(value)

    return all_imgs, all_meta_data


def build_grit_dsets(cfg, list_tokenizer, mode='train', args=None):
    assert mode in ['train', 'val', 'test']
    params = cfg.data.parameters
    dataset = GritSceneGraphDataset(
        list_tokenizers=list_tokenizer,
        grit_json=params.grit_json,
        mode=mode,
        filter_mode=params.filter_mode,
        stuff_only=params.stuff_only,
        proportion_empty_prompts=params.proportion_empty_prompts,
        image_size=(params.image_size, params.image_size),
        mask_size=params.mask_size_for_layout_object,
        min_object_size=params.min_object_size,
        min_objects_per_image=params.min_objects_per_image,
        max_objects_per_image=params.max_objects_per_image,
        instance_whitelist=params.instance_whitelist,
        stuff_whitelist=params.stuff_whitelist,
        include_other=params.include_other,
        include_relationships=params.include_relationships,
        use_deprecated_stuff2017=params.use_deprecated_stuff2017,
        deprecated_coco_stuff_ids_txt=os.path.join(params.root_dir, params[mode].deprecated_stuff_ids_txt),
        image_dir=os.path.join(params.root_dir, params[mode].image_dir),
        instances_json=os.path.join(params.root_dir, params[mode].instances_json),
        stuff_json=os.path.join(params.root_dir, params[mode].stuff_json),
        max_num_samples=params[mode].max_num_samples,
        left_right_flip=params[mode].left_right_flip,
        use_MinIoURandomCrop=params[mode].use_MinIoURandomCrop,
        return_origin_image=params.return_origin_image,
        specific_image_ids=params[mode].specific_image_ids,
        args=args,
    )

    num_objs = dataset.total_objects()
    num_imgs = len(dataset)
    print('%s dataset has %d images and %d objects' % (mode, num_imgs, num_objs))
    print('(%.2f objects per image)' % (float(num_objs) / num_imgs))

    return dataset

if __name__ == '__main__':

    from omegaconf import OmegaConf
    cfg_data = OmegaConf.load("/home/jovyan/boomcheng-data/aigc/LayoutProj/diffusers_0263/examples/controlnet/latent_LayoutDiffusion_large.yaml")
    # cfg_data = OmegaConf.load("./code_chengbo/latent_LayoutDiffusion_large.yaml")
    from transformers import AutoTokenizer
    #pretrained_model = "/home/jovyan/fast-data/stable-diffusion-xl-base-1.0"
    pretrained_model = "/home/jovyan/boomcheng-data-shcdt/herunze/models/stable-diffusion-xl-base-1.0"
    tokenizer_one = AutoTokenizer.from_pretrained(
                pretrained_model,
                subfolder="tokenizer",
                revision=None,
                use_fast=False,
            )
    tokenizer_two = AutoTokenizer.from_pretrained(
                pretrained_model,
                subfolder="tokenizer_2",
                revision=None,
                use_fast=False,
            )

    dataset = build_grit_dsets(cfg_data, [tokenizer_one, tokenizer_two], mode='train')

    if True:
        #for ii in range(len(dataset)):
        for ii in range(852, 860):
            meta_data = dataset[ii]#dict_keys(['obj_bbox', 'obj_class', 'is_valid_obj', 'upd_is_valid_obj', 'obj_class_text_ids', 'width', 'height', 'original_sizes_hw', 'num_obj_ori', 'new_height', 'new_width', 'crop_top_lefts', 'cond_image', 'base_caption', 'base_class_text_ids', 'num_selected', 'url', 'pixel_values'])
            # torch.Size([3, 512, 512]) -1~1

            #image, meta_data = dataset[ii]
            #pdb.set_trace()
            #print (meta_data)
            print (ii,meta_data["pixel_values"].shape)
            pdb.set_trace()
            pass

    """
    from prefetch_generator import BackgroundGenerator
    class DataLoaderUpd(torch.utils.data.DataLoader):                                                                        

        def __iter__(self):
            return BackgroundGenerator(super().__iter__())

    train_dataloader = DataLoaderUpd(
            dataset,
            #collate_fn=grit_collate_fn_for_layout,
            collate_fn=grit_collate_fn,
            batch_size=4,
            num_workers=0,
            pin_memory=True
        )

    for step, batch in enumerate(train_dataloader):
        #batch_cond = batch
        pdb.set_trace()
        #print (step)
        batch_images = batch["pixel_values"]
        bsize = 4
        for cid in range(bsize):
            bsid = min(batch["num_selected"][cid], 4)
            d_cond_image = batch["cond_image"][cid][:bsid]     # [10, 3, 512, 512]
            d_layo_input_ids_one = batch["layo_input_ids_one"][cid][:bsid]      # 10, 77
            d_layo_input_ids_two = batch["layo_input_ids_two"][cid][:bsid]      # 10, 77

            d_base_input_ids_one = batch["base_input_ids_one"][cid]      # 77
            d_base_input_ids_two = batch["base_input_ids_two"][cid]      # 77 

            d_base_input_ids_one = torch.repeat_interleave(torch.unsqueeze(d_base_input_ids_one, dim=0), repeats=bsid, dim=0)   # bsid, 77
            d_base_input_ids_two = torch.repeat_interleave(torch.unsqueeze(d_base_input_ids_two, dim=0), repeats=bsid, dim=0)   # bsid, 77

            d_original_sizes_hw = batch["original_sizes_hw"][cid]   # [512, 768]
            d_crop_top_lefts = batch["crop_top_lefts"][cid]         # [88, 0]

    """




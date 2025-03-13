import os
import PIL.Image
import torch
from torch.cuda.amp import custom_bwd, custom_fwd
import csv
import numpy as np
import fire
from torchvision.transforms import Resize, CenterCrop, Normalize
import PIL
import requests
import json
import cv2
import os.path as osp
from tqdm import tqdm
from PIL import Image, ImageDraw
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from omegaconf import OmegaConf
import random
from copy import deepcopy
from torchvision import transforms
import types
from diffusers.utils.torch_utils import randn_tensor
import torch.nn.functional as F
import shutil
from diffusers.optimization import get_scheduler
from rich import print
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
import kornia
from glob import glob
from PIL import Image, ImageDraw, ImageFont
import re
from torchvision.transforms import ToPILImage
console = Console()
from torchvision.transforms import ToPILImage, ToTensor
to_pil = ToPILImage()
to_ts = ToTensor()

import pickle
def save_pickle(data, name):
    with open(name, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data

import matplotlib.pyplot as plt

def vis_mask(x, name='a.png'):
    plt.imshow(x.cpu(), cmap='Greys', interpolation='nearest')
    plt.savefig(name)
    plt.close()

def pt2clipnp(refer_image):
    refer_image = resize_pt(denorm_pt(refer_image), 224)
    refer_image = norm_pt(refer_image, [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
    return refer_image

def insert_path(path, new):
    pre,post = path.rsplit('/',1)
    new_path = osp.join(pre, new, post)
    mkdir(osp.dirname(new_path))
    return new_path

##############################

# 转换函数
def convert_coordinates(coord_str):
    # 提取所有数字部分
    parts = coord_str.split(',')
    # 处理每个部分
    result = []
    for part in parts:
        # 去掉 < 和 >，提取数字
        num = int(part.strip('<>')[1:])  # 去掉 < 和 >，并去掉首字母（h 或 w）
        # 乘以 10
        num *= 10
        result.append(str(num))
    # 拼接成目标格式
    return ','.join(result)


def draw_boxes_on_image(image, prompt, use_centerhw=False):
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("/home/jovyan/boomcheng-data/tools/font/msyh.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    pattern = r"<ref>(.*?)</ref><box>\[(.*?)\]</box>"
    matches = re.findall(pattern, prompt)

    if len(matches) == 0:
        pattern = r"<ref>(.*?)</ref><box>(.*?)</box>"
        matches = re.findall(pattern, prompt)
        try:
            matches = [(x,convert_coordinates(y)) for x,y in matches]
        except:
            matches = []

    prompts = []
    boxes = []
    h,w = image.size
    for desc, box in matches:
        try:
            ori_x1, ori_y1, ori_x2, ori_y2 = map(int, box.split(","))
        except:
            continue

        if use_centerhw:
            cx, cy, _h, _w = ori_x1, ori_y1, ori_x2, ori_y2
            x1 = cx - _w / 2
            y1 = cy - _h / 2
            x2 = cx + _w / 2
            y2 = cy + _h / 2
            ori_x1, ori_y1, ori_x2, ori_y2 = x1, y1, x2, y2

        x1 = int(ori_x1/1000*h)
        x2 = int(ori_x2/1000*h)
        y1 = int(ori_y1/1000*h)
        y2 = int(ori_y2/1000*h)

        try:
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1 - 25), desc, fill="red", font=font)
            draw.text((x1, y1 - 50), str([int(ori_x1/100), int(ori_y1/100), int(ori_x2/100), int(ori_y2/100)]), fill="red", font=font)
        except:
            pass

    return np.array(image)

def mask_image(ref_image, refer_mask, bg_mode_for_refer='white'):
    refer_mask = refer_mask.float()
    if bg_mode_for_refer == 'random':
        random_color = torch.randn(3)*2-1
    elif bg_mode_for_refer == 'white':
        random_color = torch.ones(3)*2-1
    elif bg_mode_for_refer == 'black':
        random_color = torch.zeros(3)*2-1
    else:
        assert False
    ref_image = ref_image * refer_mask + (1-refer_mask) * random_color[...,None,None]
    return ref_image

import torchvision.utils as tvu
def save_img(img, path='a.png', bs=8, pad=0):
    if isinstance(img, torch.Tensor):
        tvu.save_image(tvu.make_grid(img, nrow=bs, padding=pad), path)
    elif isinstance(img, List):
        if isinstance(img[0], torch.Tensor):
            save_img(torch.stack(img), path, bs, pad)
        elif isinstance(img[0], PIL.Image.Image):
            img = torch.cat([pil2pt(t,norm=False) for t in img])
            save_img(img, path, bs, pad)
        else:
            assert False, img[0].__class__

def get_params_opt_sch(trainable, args, accelerator):
    params = []
    for model in trainable:
        params.extend([x for x in model.parameters()])

    if accelerator.is_main_process:
        print('trainable model:')
        print([t.__class__ for t in trainable], end='\n\n')
        print([get_parameter_number(t) for t in trainable], end='\n\n')
        print("all in optimizer:", get_parameter_number_params(params), end='\n\n')

    params_with_lr = []
    for model in trainable:
        params_with_lr.append({'params': model.parameters(), 'lr':args.learning_rate})
        
    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        params_with_lr,
        # params,
        # lr=args.learning_rate, d
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    return params, optimizer, lr_scheduler

def resume_model(args, accelerator):
    global_step = 0
    if args.resume_from_checkpoint != "latest":
        if os.path.exists(args.resume_from_checkpoint):
            path = args.resume_from_checkpoint
        elif isinstance(args.resume_from_checkpoint, int):
            path = f"checkpoint-{args.resume_from_checkpoint}"
        else:
            path = os.path.basename(args.resume_from_checkpoint)
    else:
        # Get the most recent checkpoint
        dirs = os.listdir(args.output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None

    if path is None:
        accelerator.print(
            f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
        )
        args.resume_from_checkpoint = None
    else:
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(args.output_dir, path), map_location='cpu')
        global_step = int(os.path.basename(path).split("-")[1])

        # resume_global_step = global_step * args.gradient_accumulation_steps
        # first_epoch = global_step // num_update_steps_per_epoch
        # resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    return global_step

def get_latent_and_target(images, vae, scheduler, accelerator):
    latents = vae.encode(images).latent_dist.sample()
    latents = latents * vae.config.scaling_factor

    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=accelerator.device).long()
    latents = scheduler.add_noise(latents, noise, timesteps)

    if scheduler.config.prediction_type == "epsilon":
        target = noise
    elif scheduler.config.prediction_type == "v_prediction":
        target = scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {scheduler.config.prediction_type}")
    return latents, timesteps, target

def backward_and_step(args, params, loss, accelerator, optimizer, lr_scheduler):
    # avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
    # train_loss += avg_loss.item() / args.gradient_accumulation_steps

    # Backpropagate
    accelerator.backward(loss)
    if accelerator.sync_gradients:
        accelerator.clip_grad_norm_(params, args.max_grad_norm)

    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

def save_model(args, logger, accelerator, global_step):
    if args.checkpoints_total_limit is not None:
        checkpoints = os.listdir(args.output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        if len(checkpoints) >= args.checkpoints_total_limit:
            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")
            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    accelerator.save_state(save_path)
    logger.info(f"Saved state to {save_path}")

def init_latents(vae,scheduler,device,batch_size=1,height=512,width=512,generator=None):
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    shape = (batch_size, 4, height // vae_scale_factor, width // vae_scale_factor)
    latents = randn_tensor(shape, generator=generator, device=device) * scheduler.init_noise_sigma
    return latents

def vae_decode(vae, latents, generator=None):
    images = []
    for i in range(0,100,8):
        if len(latents[i:i+8]) == 0:
            break
        image = vae.decode(latents[i:i+8] / vae.config.scaling_factor, return_dict=False, generator=generator)[0]
        images.append(image)
    return images

def lines(name):
    with open(name, 'r') as f:
        contents = f.readlines()
        lines_without_newline = [line.rstrip('\n') for line in contents]
        return lines_without_newline
    
def dilate(path, size=None, iters=1):
    # gray_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    gray_image = Image.open(path).convert('L')
    if size is not None:
        gray_image = gray_image.resize((size,size))
    gray_image = np.array(gray_image)
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    dilated_image = cv2.dilate(binary_image, kernel=np.ones((3,3),np.uint8), iterations=iters)
    # cv2.imwrite('a.png', dilated_image)
    return dilated_image

def dilate_ts_test(ts, size=None, iters=1):
    # gray_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # gray_image = Image.open(path).convert('L')

    # if size is not None:
    #     gray_image = gray_image.resize((size,size))
    gray_image = (ts[0].cpu().detach().numpy()*255).astype(np.uint8)
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    dilated_image = cv2.dilate(binary_image, kernel=np.ones((7,7),np.uint8), iterations=iters)

    # import pdb;pdb.set_trace()

    dilated_image = (torch.tensor(np.array(dilated_image))/255).float()[None]

    return dilated_image

def norm1(t):
    t = t - t.min()
    return t/(t.max()+1e-6)

def seq2mat(t):
    bs, nseq = t.shape
    h = int(np.sqrt(nseq))
    t = t.reshape(bs,1,h,h)
    return Resize((32,32))(t)

def seq2mat_ca(t):
    t = t.permute(0,2,1)
    bs, _77, nseq = t.shape
    h = int(np.sqrt(nseq))
    try:
        t = t.reshape(bs,77,h,h)
    except:
        import pdb;pdb.set_trace()
        pass
    return Resize((32,32))(t)

def mkdir(path):
    os.makedirs(path, exist_ok=True)

def save_np_img(img, filename):
    Image.fromarray(img.astype(np.uint8)).save(filename)

def random_range(pad_range):
    return random.choice(range(*pad_range))

from PIL import Image, ImageDraw

def draw_rectangle_on_image(img, top_left, bottom_right):
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    
    # 使用draw.rectangle()函数绘制矩形
    # 参数是矩形两个对角点的坐标，fill是填充色，outline是边框色
    draw.rectangle([top_left, bottom_right], outline="red")
    
    # 保存修改后的图像
    # img.save(output_path)
    # return img

def get_mask_bounds(mask, pad_range=None):
    # 获取mask最左上和最右下的坐标。

    y_indices, x_indices = np.where(mask != 0)
    h,w = mask.shape

    if not len(x_indices) or not len(y_indices):
        return None

    x1,y1 = np.min(x_indices), np.min(y_indices)
    x2,y2 = np.max(x_indices), np.max(y_indices)

    if pad_range is not None:
        left_top     = (max(0, x1-random_range(pad_range)),max(0, y1-random_range(pad_range)))
        right_bottom = (min(h, x2+random_range(pad_range)),min(w, y2+random_range(pad_range)))
    else:
        left_top     = (x1, y1)
        right_bottom = (x2, y2)
    
    return left_top, right_bottom


def smooth_transition_tensor(image_a1, image_a2, mask, transition_width=5):
    """
    使用Tensor实现平滑过渡
    :param image_a1: 图像a1，形状为(bs, 3, h, w)，值范围为[0, 1]
    :param image_a2: 图像a2，形状为(bs, 3, h, w)，值范围为[0, 1]
    :param mask: 掩码图像，形状为(bs, 1, h, w)，值范围为[0, 1]
    :param transition_width: 过渡区域的宽度
    :return: 混合后的图像，形状为(bs, 3, h, w)，值范围为[0, 1]
    """
    # 使用高斯模糊对mask进行平滑处理，创建平滑过渡效果
    # 注意：这里的高斯模糊需要自定义或使用现有库，如kornia
    # 以下代码假设已有一个名为gaussian_blur的函数
    blurred_mask = gaussian_blur(mask, kernel_size=transition_width)

    blurred_mask = torch.clamp(blurred_mask, 0, 1)

    # blurred_mask = (blurred_mask > 0.).float()
    
    # 计算反向掩码
    reverse_blurred_mask = 1 - blurred_mask
    
    # 使用mask进行区域选择和混合
    blended_image = image_a1 * blurred_mask + image_a2 * reverse_blurred_mask
    
    return blended_image

def gaussian_blur(mask, kernel_size):
    """
    对mask应用高斯模糊
    :param mask: 待模糊的mask，形状为(bs, 1, h, w)
    :param kernel_size: 高斯核的大小
    :return: 模糊后的mask
    """
    # 计算padding，确保输出大小与输入相同
    padding = kernel_size // 2
    # 创建高斯核
    gaussian_kernel = kornia.filters.GaussianBlur2d((kernel_size, kernel_size), (padding, padding))
    # 应用高斯模糊
    blurred_mask = gaussian_kernel(mask)
    return blurred_mask

def crop_image_by_bounds(image, left_top, right_bottom):
    # cropped_image = image.crop((left_top[0], left_top[1], right_bottom[0], right_bottom[1]))
    cropped_image = image[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]]
    return cropped_image

def crop_image_by_bounds_ts(image, left_top, right_bottom):
    # cropped_image = image.crop((left_top[0], left_top[1], right_bottom[0], right_bottom[1]))
    if len(image.shape) == 4:
        cropped_image = image[:,:,left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]]
    else:
        cropped_image = image[:,left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]]
    return cropped_image

def get_pg():
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
    progress_columns = [
        BarColumn(bar_width=40),
        TextColumn("{task.completed}/{task.total}, {task.percentage:.1f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]
    return Progress(*progress_columns)

    # with pg as progress:
    # task = progress.add_task("[cyan]doing...", total=100)
    
    # for i in range(100):
    #     time.sleep(0.1)
    #     progress.update(task, advance=1)

def split_list(lst, n_parts=8):
    # 计算每份的大致长度，向上取整以确保所有元素都能被分配
    n = len(lst)
    part_size = -(-n // n_parts)  # 使用负数向上取整的技巧
    # 使用列表推导式和range步长来拆分列表
    return [lst[i:i + part_size] for i in range(0, n, part_size)]

def pil2pt(t, norm=True):
    if norm:
        return torch.tensor(np.array(t.convert('RGB'))).permute(2,0,1)[None]/127.5-1
    else:
        return torch.tensor(np.array(t.convert('RGB'))).permute(2,0,1)[None]/255

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def dummy_args():
    a={}
    return OmegaConf.create(a)

def post_processing(t, resize=None):
    t = t.clamp(-1,1)
    t = denorm_pt(t)
    if resize:
        t = resize_pt(t, 200)
    # bs,3,h,w
    t = t.permute(0,2,3,1).flatten(0,1)
    # bs*h,3,w
    t = pt2np(t)
    return t
    
def pt2np(x, to255=True):
    if to255:
        return (x.cpu().detach().numpy() * 255).astype(np.uint8)
    else:
        return x.cpu().detach().numpy()

def pt2pil(x, denorm=False, resize=None):
    if denorm:
        x = (x+1)/2
    if resize is not None:
        x = Resize((resize,resize))(x)
    assert len(x.shape) == 4
    imgs = (x.permute(0,2,3,1)*255).cpu().detach().numpy().astype(np.uint8)#bs,h,w,3
    return [Image.fromarray(t) for t in imgs]

def denorm_pt(pt):
    return (pt.clamp(-1,1)+1)/2

def donorm_pt(pt):
    return pt*2-1

def norm_pt(pt, mean=[0.48145466, 0.4578275, 0.40821073], var=[0.26862954, 0.26130258, 0.27577711]):
    device = pt.device
    # ret = ((pt-torch.tensor(mean).to(device))/torch.tensor(var).to(device))
    ret = ((pt.permute(0,2,3,1)-torch.tensor(mean).to(device))/torch.tensor(var).to(device)).permute(0,3,1,2)
    return ret

def resize_pt(pt,s):
    if isinstance(s, int):
        return Resize((s,s))(pt)
    else:
        assert len(s) == 2
        return Resize(s)(pt)


def pdb():
    import pdb;pdb.set_trace()
    pass

def loader(dataset, shuffle=True, batch_size=1, dataloader_num_workers=8, pin_memory=False):
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=dataloader_num_workers,
        pin_memory=pin_memory,
    )



def pillist2ts(imgs):
    imgs = [pil2pt(t, norm=False) for t in imgs]
    img = torch.cat(imgs)
    return img

def ts2np(t):
    return t.cpu().detach().numpy()

def ts2array(t):
    # 3,h,w
    # import pdb;pdb.set_trace()
    t = t.clamp(0,1)
    return (t.permute(1,2,0)*255).cpu().detach().numpy().astype(np.uint8)

def load_to_np(path, mode='RGB', size=None,):
    if size is None:
        img = np.array(Image.open(path).convert(mode))
    else:
        img = np.array(Image.open(path).resize((size,size)).convert(mode))
    return img

def load2pil(path, mode='RGB', size=None,):
    if size is None:
        img = Image.open(path).convert(mode)
    else:
        img = Image.open(path).resize((size,size)).convert(mode)
    return img

def comp_ts(imgs, texts=None, r=200):
    # bs, dim, h, w
    bs = len(imgs[0])

    new_imgs = []
    for t in imgs:
        if t.ndim == 3:
            t = t[:,None]
        if t.shape[1] == 1:
            t = t.repeat(1,3,1,1)
        if t.min() < -0.8:
            t = denorm_pt(t)
        # 0~1
        t = Resize((r,r))(t)
        t = t.cpu()
        new_imgs.append(t)

    imgs_cat = torch.cat(new_imgs, dim=0)
    img_grid = tvu.make_grid(imgs_cat, nrow=bs, padding=0)
    return img_grid

def dilate_ts(tensor, kernel_size=3, iterations=1, padding=1):
    """
    对输入的tensor执行dilate操作。
    
    参数:
    - tensor: 输入的tensor，形状为[1, 1, H, W]
    - kernel_size: 卷积核的大小
    - iterations: 膨胀操作的迭代次数
    - padding: 卷积操作的填充大小
    """
    # 创建一个卷积核
    kernel = torch.ones((1, 1, kernel_size, kernel_size)).to(tensor)
    
    # 执行指定次数的dilate操作
    for _ in range(iterations):
        tensor = F.conv2d(tensor, kernel, padding=padding)
        tensor = (tensor > 0).to(kernel) # float的话是32
    
    return tensor

# def load2ts(path, mode='RGB', resize=(256,256), norm=True):
#     if resize is not None:
#         if isinstance(resize, tuple):
#             img = np.array(Image.open(path).resize(resize).convert(mode))
#         else:
#             img = np.array(Image.open(path).resize((resize,resize)).convert(mode))
#     else:
#         img = np.array(Image.open(path).convert(mode))
#     if norm:
#         img = torch.tensor(img).permute(2,0,1)/127.5-1
#     else:
#         img = torch.tensor(img).permute(2,0,1)/255
#     return img

def load2np(path, mode='RGB', in_01=False):
    if isinstance(path, str):
        img = Image.open(path).convert(mode)
    else:
        img = path.convert(mode)
    img = np.array(img)
    if in_01:
        img = img / 255
    return img

from typing import Dict
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
def scale_boxes(boxes, width, height):
    scaled_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        scaled_box = [x_min * width, y_min * height, x_max * width, y_max * height]
        scaled_boxes.append(scaled_box)
    return scaled_boxes

def draw_mask(mask, draw, random_color=True):
    if random_color:
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            153,
        )
    else:
        color = (30, 144, 255, 153)

    nonzero_coords = np.transpose(np.nonzero(mask))
    
    for coord in nonzero_coords:
        draw.point(coord[::-1], fill=color)
        
def bbox_visualization(image_pil: Image,
              result: Dict,
              draw_width: float = 3.0,
              return_mask=True,
              font_size=20,
) -> Image:
    """Plot bounding boxes and labels on an image.

    Args:
        image_pil (PIL.Image): The input image as a PIL Image object.
        result (Dict[str, Union[torch.Tensor, List[torch.Tensor]]]): The target dictionary containing
            the bounding boxes and labels. The keys are:
                - boxes (List[int]): A list of bounding boxes in shape (N, 4), [x1, y1, x2, y2] format.
                - scores (List[float]): A list of scores for each bounding box. shape (N)
                - labels (List[str]): A list of labels for each object
                - masks (List[PIL.Image]): A list of masks in the format of PIL.Image
        draw_score (bool): Draw score on the image. Defaults to False.

    Returns:
        PIL.Image: The input image with plotted bounding boxes, labels, and masks.
    """
    # Get the bounding boxes and labels from the target dictionary
    boxes = result["boxes"]
    categorys = result["labels"]
    masks = result.get("masks", [])

    
    color_list= [(177, 214, 144),(255, 162, 76),
                (13, 146, 244),(249, 84, 84),(54, 186, 152),
                (74, 36, 157),(0, 159, 189),
                (80, 118, 135),(188, 90, 148),(119, 205, 255)]


    np.random.seed(42)

    # Find all unique categories and build a cate2color dictionary
    cate2color = {}
    unique_categorys = sorted(set(categorys))
    for idx,cate in enumerate(unique_categorys):
        cate2color[cate] = color_list[idx%len(color_list)]
    
    # Load a font with the specified size
    # font = ImageFont.truetype("utils/arial.ttf", font_size)
    font = ImageFont.truetype("/home/jovyan/boomcheng-data/tools/font/msyh.ttf", font_size)
    
    # Create a PIL ImageDraw object to draw on the input image
    if isinstance(image_pil, np.ndarray):
        image_pil = Image.fromarray(image_pil)
    draw = ImageDraw.Draw(image_pil)
    
    # Create a new binary mask image with the same size as the input image
    mask = Image.new("L", image_pil.size, 0)
    # Create a PIL ImageDraw object to draw on the mask image
    mask_draw = ImageDraw.Draw(mask)

    # Draw boxes, labels, and masks for each box and label in the target dictionary
    for box, category in zip(boxes, categorys):
        try:
            # Extract the box coordinates
            x0, y0, x1, y1 = box

            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            color = cate2color[category]

            # Draw the box outline on the input image
            draw.rectangle([x0, y0, x1, y1], outline=color, width=int(draw_width))

            # Draw the label and score on the input image
            text = f"{category}"
        
            if hasattr(font, "getbbox"):
                bbox = draw.textbbox((x0, y0), text, font)
            else:
                w, h = draw.textsize(text, font)
                bbox = (x0, y0, w + x0, y0 + h)
            draw.rectangle(bbox, fill=color)
            draw.text((x0, y0), text, fill="white",font=font)
        except:
            print('bug in draw')
    # Draw the mask on the input image if masks are provided
    if len(masks) > 0 and return_mask:
        size = image_pil.size
        mask_image = Image.new("RGBA", size, color=(0, 0, 0, 0))
        mask_draw = ImageDraw.Draw(mask_image)
        for mask in masks:
            mask = np.array(mask)[:, :, -1]
            draw_mask(mask, mask_draw)

        image_pil = Image.alpha_composite(image_pil.convert("RGBA"), mask_image).convert("RGB")
    return image_pil

def load2ts(path, mode='RGB',resize=None, norm=True, aug_trans=None, mask=None):
    if isinstance(path, str):
        img = Image.open(path).convert(mode)
    else:
        img = path.convert(mode)
    if resize:
        if not isinstance(resize, tuple):
            resize = (resize, resize)
        img = img.resize(resize)
    img = np.array(img)

    if mask is not None:
        mask = Image.open(mask).convert('RGB')
        if resize:
            if not isinstance(resize, tuple):
                resize = (resize, resize)
            mask = mask.resize(resize)
        mask = np.array(mask)

    if aug_trans is not None:
        if mask is not None:
            out = aug_trans(image=img,mask=mask)
            img = out['image']
            mask = out['mask']
        else:
            img = aug_trans(image=img)['image']
        
    img = torch.tensor(img).permute(2,0,1)
    if norm:
        img = img/127.5-1
    else:
        img = img/225

    if mask is not None:
        mask = torch.tensor(mask).permute(2,0,1)[:1]
        mask = mask/225
        return img,mask
    else:
        return img

def load2ts_ori(path, mode='RGB'):
    img = np.array(Image.open(path).convert(mode))
    img = torch.tensor(img).permute(2,0,1)/255
    return img

def tensor2np_final(images, args=None):
    if args is not None and 'tryon' in args.test_data:
        images = (Resize((256,192))(images.clamp(-1,1)) + 1)/2
    else:
        images = (Resize((256,256))(images.clamp(-1,1)) + 1)/2
    images = (images.permute(0,2,3,1).cpu().detach().numpy() * 255).astype(np.uint8) # bs,h,w,3
    # images = (images.permute(0,2,3,1).flatten(0,1).cpu().detach().numpy() * 255).astype(np.uint8)
    return images

def tensor2np_final_01(images, args=None):
    if args is not None and 'tryon' in args.test_data:
        images = Resize((256,192))(images)
    else:
        images = Resize((256,256))(images)
    images = (images.permute(0,2,3,1).cpu().detach().numpy() * 255).astype(np.uint8) # bs,h,w,3
    # images = (images.permute(0,2,3,1).flatten(0,1).cpu().detach().numpy() * 255).astype(np.uint8)
    return images

def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
    img[:h] = image
    size = 0.75
    thick = 1 # int
    textsize = cv2.getTextSize(text, font, size, thick)[0]#int
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, size, text_color, thick)#0.5是size
    return img

def convert_to_np(image, resolution, mode='RGB', p=1):
    image = image.convert(mode).resize((resolution*p, resolution))
    return np.array(image).transpose(2, 0, 1)

def convert_to_np_ct(image, resolution, mode='RGB', p=1, size=0):
    image = image.convert(mode)
    if size != 0:
        image = CenterCrop(512-size*2)(image.resize((512,512)))
    image = image.resize((resolution*p, resolution))
    image = np.array(image)
    return image.transpose(2, 0, 1)

def convert_to_np_mask(image, resolution, mode='L'):
    image = image.convert('L').resize((resolution, resolution))
    return np.array(image)[None]

def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

def tokenize_captions(captions, tokenizer):
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids

# def get_parameter_number(model):
#     total_num = sum(p.numel() for p in model.parameters())
#     trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     return {'Total': f"{total_num/1024/1024} MB", 'Trainable': f"{trainable_num/1024/1024} MB"}

def get_parameter_number_params(params):
    total_num = sum(p.numel() for p in params)/1024/1024
    trainable_num = sum(p.numel() for p in params if p.requires_grad)/1024/1024
    return f"Parameter: {round(total_num,2)} MB, Trainable: {round(trainable_num,2)} MB"

def tensor2np(images, norm=True, flatten=True, resize=True):
    if norm:
        images = (images.clamp(-1,1)+1)/2
    if resize:
        images = Resize((200,200))(images)
    if flatten:
        images = (images.permute(0,2,3,1).flatten(0,1).cpu().detach().numpy() * 255).astype(np.uint8)
    else:
        images = (images.permute(0,2,3,1).cpu().detach().numpy() * 255).astype(np.uint8)
    return images

def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result

class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)

        # dummy loss value
        return torch.zeros([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        gt_grad, = ctx.saved_tensors
        batch_size = len(gt_grad)
        return gt_grad / batch_size, None
    

def load_jsonl(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_json(filename):
    with open(filename, "r") as file:
        data = json.load(file)
    return data

def default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError

def save_json(filename, data):
    with open(filename, "w") as file:
        json.dump(data, file, default=default)

def load_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file]
    return lines

def save_jsonl(filename, data_list):
    with open(filename, 'w') as f:
        for item in data_list:
            f.write(json.dumps(item) + '\n')

def load_csv(name):
    with open(name, mode='r', encoding='utf-8') as file:
        # 创建 CSV 读取器
        reader = csv.reader(file)
        
        # 将每一行转换为列表，并存储到行的列表中
        rows_list = list(reader)

    # 打印结果
    # print(rows_list)
    return rows_list
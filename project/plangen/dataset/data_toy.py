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

class Dataset_toy(Dataset):
    def __init__(
        self,
        args,
        is_test=False,
    ):
        self.args = args
        self.is_test = is_test

    def __len__(self):
        return 100

    def __getitem__(self, i):
        prompt = 'a sks meme'
        image_path = '/home/jovyan/boomcheng-data-shcdt/herunze/code/base/Janus/images/doge.png'
        image = (torch.tensor(np.array(Image.open(image_path).resize((self.args.janus_hw,self.args.janus_hw))))/255-0.5)*2
        image = image.permute(2,0,1)

        ret = dict(
            prompt=prompt,
            image=image,
            image_path=image_path,
        )
        return ret
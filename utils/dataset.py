from torch.utils.data import Dataset
from einops import rearrange
from pathlib import Path
import torch
import torchvision
import json
from PIL import Image
import numpy as np
import math

class InstructFashionDataset(Dataset):
    def __init__(
        self,
        path: str,
        task: str,
        split: str = "train",
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        flip_prob: float = 0.0,
    ):
        assert split in ("train", "val", "test")
        assert task in ("add", "rem", "alt", "rep")
        self.path = path
        self.task = task
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.flip_prob = flip_prob

        with open(Path(self.path, f"{self.task}.json")) as f:
            self.pairs = json.load(f)[split]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        item = self.pairs[i]
        src_img = item['src_image']
        tar_img = item['tar_image']
        inst = item['instruction']

        image_0 = Image.open(Path(self.path, src_img)).convert("RGB")
        image_1 = Image.open(Path(self.path, tar_img)).convert("RGB")

        reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
        image_0 = image_0.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        image_1 = image_1.resize((reize_res, reize_res), Image.Resampling.LANCZOS)

        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")

        flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        image_0, image_1 = flip(torch.cat((image_0, image_1))).chunk(2)
        
        if self.task in ['add', 'rep']:
            ref_img = item['ref_image']
            image_2 = Image.open(Path(self.path, ref_img)).convert("RGB")
            image_2 = image_2.resize((224, 224), Image.Resampling.LANCZOS)
            image_2 = rearrange(2 * torch.tensor(np.array(image_2)).float() / 255 - 1, "h w c -> c h w")
            
            sample = dict(
                edited_pixel_values = image_1, 
                original_pixel_values = image_0, 
                reference_pixel_values = image_2,
                input_ids = inst,
                type = self.task
            )
        else:
            sample = dict(
                edited_pixel_values = image_1, 
                original_pixel_values = image_0, 
                reference_pixel_values = image_2,
                input_ids = inst,
                type = self.task
            )

        return sample

class ControlDataset(Dataset):
    def __init__(
        self,
        path: str,
        split: str = "train",
        splits: tuple = (0.9, 0.05, 0.05),
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        flip_prob: float = 0.0,
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.flip_prob = flip_prob
        self.parsing_num_class = 18

        with open(Path(self.path, "img_par.json")) as f:
            self.pairs = json.load(f)

        split_0, split_1 = {
            "train": (0.0, splits[0]),
            "val": (splits[0], splits[0] + splits[1]),
            "test": (splits[0] + splits[1], 1.0),
        }[split]

        idx_0 = math.floor(split_0 * len(self.pairs))
        idx_1 = math.floor(split_1 * len(self.pairs))
        self.pairs = self.pairs[idx_0:idx_1]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        item = self.pairs[i]
        src_img = item['src_image']
        tar_img = item['tar_image']
        tar_hp = item['tar_parsing']
        inst = item['instruction']

        image_0 = Image.open(Path(self.path, src_img)).convert("RGB")
        image_1 = Image.open(Path(self.path, tar_img)).convert("RGB")
        parsing = Image.open(Path(self.path, tar_hp)).convert("L")

        reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
        image_0 = image_0.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        image_1 = image_1.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        parsing = parsing.resize((reize_res, reize_res), Image.Resampling.NEAREST)

        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")
        parsing = rearrange(F.one_hot(torch.tensor(np.array(parsing)).long(), num_classes=self.parsing_num_class), "h w c -> c h w")
        
        flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        fliped_tensor = flip(torch.cat((image_0, image_1, parsing)))
        image_0, image_1, parsing = fliped_tensor[:3], fliped_tensor[3:6], fliped_tensor[6:]
        
        return dict(
                edited_pixel_values = image_1, 
                original_pixel_values = image_0, 
                target_parsing_values = parsing, 
                input_ids = inst
            )
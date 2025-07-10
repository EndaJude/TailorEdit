from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms
from types import MethodType
import torch
from einops import rearrange
import numpy as np
from torch.utils.data import DataLoader, Sampler, Dataset

class CombinedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.dataset_lengths = [len(ds) for ds in datasets]
        self.num_datasets = len(datasets)

    def __len__(self):
        return sum(self.dataset_lengths)

    def __getitem__(self, idx):
        # 确定当前索引来自哪个数据集
        dataset_idx = 0
        sample_idx = idx

        while sample_idx >= self.dataset_lengths[dataset_idx]:
            sample_idx -= self.dataset_lengths[dataset_idx]
            dataset_idx += 1

        return self.datasets[dataset_idx][sample_idx]

class SequentialDatasetSampler(Sampler):
    def __init__(self, dataset_lengths, batch_size):
        self.dataset_lengths = dataset_lengths
        self.batch_size = batch_size
        self.num_datasets = len(dataset_lengths)
        self.indices = [
            list(range(length)) for length in dataset_lengths
        ]
        self.current_batch_indices = [0] * self.num_datasets  # 记录每个数据集的当前批次索引
        self.last_dataset_index = -1

    def __iter__(self):
        # 循环重置
        if all(
            self.current_batch_indices[i] >= self.dataset_lengths[i]
            for i in range(self.num_datasets)
        ):
            self.current_batch_indices = [0] * self.num_datasets

        while any(
            self.current_batch_indices[i] < self.dataset_lengths[i]
            for i in range(self.num_datasets)
        ):
            dataset_idx = self.last_dataset_index
            # 找到下一个尚未完全遍历的dataset_idx
            while True:
                dataset_idx = (dataset_idx + 1) % self.num_datasets
                if self.current_batch_indices[dataset_idx] < self.dataset_lengths[dataset_idx]:
                    break

            start_idx = self.current_batch_indices[dataset_idx]
            end_idx = min(start_idx + self.batch_size, self.dataset_lengths[dataset_idx])
            base_idx = sum(self.dataset_lengths[:dataset_idx])
            self.current_batch_indices[dataset_idx] += self.batch_size
            self.last_dataset_index = dataset_idx
            yield range(base_idx + start_idx, base_idx + end_idx)

    def __len__(self):
        return sum(len(indices) for indices in self.indices) // self.batch_size

class CustomDataLoader(DataLoader):
    def __init__(self, dataloaders):
        """
        初始化自定义 DataLoader。循环依次遍历列表中Dataloader。

        Args:
            dataloaders (list): 由 DataLoader 对象组成的列表。
        """
        self.dataloaders = dataloaders
        self.iterators = [iter(dl) for dl in dataloaders]  # 创建每个 DataLoader 的迭代器
        self.active_iterators = list(range(len(dataloaders)))  # 当前活跃的迭代器索引

    def __iter__(self):
        """
        返回自定义迭代器。
        """
        self.iterators = [iter(dl) for dl in self.dataloaders]  # 重新初始化迭代器
        self.active_iterators = list(range(len(self.dataloaders)))  # 重置活跃迭代器
        return self

    def __next__(self):
        """
        实现迭代器的 next 方法。
        """
        if not self.active_iterators:
            # 如果所有迭代器都已经消耗完，停止迭代
            raise StopIteration

        for idx in list(self.active_iterators):  # 遍历活跃的迭代器
            try:
                # 尝试从当前迭代器中取出数据
                data = next(self.iterators[idx])
                self.active_iterators.remove(idx)
                self.active_iterators.append(idx)
                return data
            except StopIteration:
                # 如果当前迭代器耗尽，从活跃列表中移除
                self.active_iterators.remove(idx)

        # 如果循环结束后所有迭代器都耗尽，抛出 StopIteration
        raise StopIteration

    def __len__(self):
        """
        返回总批次数量（所有子 DataLoader 的批次数之和）。
        """
        return sum(len(dl) for dl in self.dataloaders)

def replace_call_method(module, target_class, new_method):
    """
    遍历模块，替换目标类的 __call__ 方法为新的方法。

    Args:
        module (nn.Module): 待遍历的模型。
        target_class (type): 目标类。
        new_method (callable): 新的 __call__ 方法。
    """
    # 遍历当前模块的子模块
    for name, sub_module in module.named_children():
        # 如果子模块是目标类
        if isinstance(sub_module, target_class):
            # 替换 __call__ 方法
            sub_module.__class__.__call__ = new_method
        else:
            # 递归遍历子模块
            replace_call_method(sub_module, target_class, new_method)

def update_fid(fid, gt_images, gen_images):
    if isinstance(gt_images, Image.Image) and isinstance(gen_images, Image.Image):
        transform = transforms.ToTensor()
        gt_images, gen_images = transform(gt_images), transform(gen_images)
    elif not isinstance(gt_images, torch.Tensor) or not isinstance(gt_images, torch.Tensor):
        raise ValueError(f"Type of image input for FID is valid.")
    
    fid.update(gt_images.to(fid.device, fid.dtype), real=True)
    fid.update(gen_images.to(fid.device, fid.dtype), real=False)
    
    return fid

def show_fid(fid, fid_list, output_dir):
    fid_score = fid.compute().cpu()
    print(f"FID={fid_score}")
    fid_list.append(fid_score)
    
    plt.figure()
    plt.plot(fid_list, label='FID')
    plt.xlabel('epochs')
    plt.ylabel('fid_score')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(output_dir)
    plt.close()


def draw_loss(loss_list:list, output_dir):
    plt.figure()
    plt.plot(loss_list, label='Loss')
    plt.xlabel('steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(output_dir)
    plt.close()

import torch.nn.functional as F
def prepare_data(image_path, parsing_path, instruction, reference_path, reize_res=512):
    assert image_path is not None and instruction is not None

    src_img = Image.open(image_path).convert("RGB")
    src_img = src_img.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
    src_img = rearrange(2 * torch.tensor(np.array(src_img)).float() / 255 - 1, "h w c -> c h w").unsqueeze(0)

    try:
        ref_img = Image.open(reference_path).convert("RGB")
        ref_img = ref_img.resize((224, 224), Image.Resampling.LANCZOS) if ref_img is not None else None
        ref_img = rearrange(2 * torch.tensor(np.array(ref_img)).float() / 255 - 1, "h w c -> c h w").unsqueeze(0)
    except:
        ref_img = None

    try:
        parsing = Image.open(parsing_path).convert("L")
        parsing = parsing.resize((reize_res, reize_res), Image.Resampling.NEAREST)
        # when source parsing
        parsing = torch.tensor(np.array(parsing)).unsqueeze(0)
        # when target parsing
        # parsing = rearrange(F.one_hot(torch.tensor(np.array(parsing)).long(), num_classes=18), "h w c -> c h w").unsqueeze(0)      
    except:
        # Get parsing from the source image by using an onnx parse generator.
        pass

    return dict(
        image = src_img,
        instruction = [instruction],
        parsing = parsing,
        reference = ref_img,
    )
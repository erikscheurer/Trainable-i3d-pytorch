import numpy as np
import torch
from pathlib2 import Path
from torch.utils.data import Dataset
from utils.temporal_transforms import *
from PIL import Image

import torchvision.transforms.functional as TF
import torchvision.transforms as transforms


class SpacialTransform(Dataset):
    def __init__(self, output_size=(224, 224)):
        self.output_size = output_size

    def refresh_random(self, image_np):
        # Random crop
        self.crop_dim = transforms.RandomCrop.get_params(
            Image.fromarray(image_np.astype('uint8')), output_size=self.output_size)

        # Random horizontal flipping
        self.horizontal_flip = False
        if random.random() > 0.5:
            self.horizontal_flip = True

        # Random vertical flipping
        self.vertical_flip = False
        # if random.random() > 0.5:
        #     self.vertical_flip = True

    def transform(self, imgs_np, flow = False):
        # images_np: (T, H, W, C) # C = 3 for rgb, C = 2 for flow
        # now do cropping and flipping if needed
        if not flow:
            imgs_np = imgs_np/255
        imgs = torch.from_numpy(imgs_np).permute(3, 0, 1, 2) # (C, T, H, W) for torchvison
        if not flow:
            # BGR to RGB
            imgs = imgs[[2, 1, 0], :, :]
        imgs = TF.crop(imgs, *self.crop_dim)
        if self.horizontal_flip:
            imgs = TF.hflip(imgs)
        if self.vertical_flip:
            imgs = TF.vflip(imgs)
        return imgs # (C, T, H, W)


class RGBFlowDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, split_dir, split, class_dict, sample_rate=1, sample_type="num", fps=5, out_frame_num=32, augment=False, flow=True, rgb=True):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        self.flow = flow
        self.rgb = rgb

        if split == "train":
            files =      [i.split(' ')[0] for i in (Path(split_dir)/"trainlist01.txt").open('r').read().split('\n') if i]
            files.extend([i.split(' ')[0] for i in (Path(split_dir)/"trainlist02.txt").open('r').read().split('\n') if i])
            files.extend([i.split(' ')[0] for i in (Path(split_dir)/"trainlist03.txt").open('r').read().split('\n') if i])
        elif split == "val":
            files =      [i.split(' ')[0] for i in (Path(split_dir)/"testlist01.txt").open('r').read().split('\n') if i]
        elif split == "test":
            files =      [i.split(' ')[0] for i in (Path(split_dir)/"testlist01.txt").open('r').read().split('\n') if i]
            files.extend([i.split(' ')[0] for i in (Path(split_dir)/"testlist02.txt").open('r').read().split('\n') if i])
            files.extend([i.split(' ')[0] for i in (Path(split_dir)/"testlist03.txt").open('r').read().split('\n') if i])

        self.root_dir = Path(root_dir)
        self.sub_dirs = [i for i in (self.root_dir/"flow").iterdir() if i.is_dir() and not i.stem.startswith('.')]
        self.class_names = [i.stem for i in self.sub_dirs]
        self.data_pairs = []
        self.spacial_transform = SpacialTransform()
        self.temporal_transform = TemporalRandomCrop(out_frame_num)
        self.augment = augment
        # self.temporal_transform =
        for sub_dir in self.sub_dirs: # 
            contents = [i for i in sub_dir.iterdir() if i.is_file() and not i.stem.startswith('.')]
            if contents:
                temp_flow = contents
                temp_rgb = [Path(i.as_posix().replace('/flow/', '/rgb/')) for i in contents]
                for f,r in zip(temp_flow, temp_rgb):
                    if f.parent.stem+'/'+f.stem+'.avi' in files:
                        self.data_pairs.append((r, f, class_dict[sub_dir.stem]))

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        if self.rgb:
            rgb_data = np.float32(np.load(self.data_pairs[idx][0]))
            self.spacial_transform.refresh_random(rgb_data[0])
            rgb_data = self.temporal_transform(rgb_data) # (T, H, W, C)
            rgb_data = self.spacial_transform.transform(rgb_data) # (C, T, H, W)
        else:
            rgb_data = np.zeros(1)
        if self.flow:
            flow_data = np.float32(np.load(self.data_pairs[idx][1]))
            if not self.rgb:
                self.spacial_transform.refresh_random(flow_data[0])
            flow_data = self.temporal_transform(flow_data)
            flow_data = self.spacial_transform.transform(flow_data, flow=True)
        else:
            flow_data = np.zeros(1)
        return rgb_data, flow_data, self.data_pairs[idx][2]

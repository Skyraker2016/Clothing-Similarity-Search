import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class DeepFashionDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None):
        self.data = []
        with open(root_dir + "Anno/" + txt_file) as file:
            num = int(file.readline())
            file.readline()
            for _ in range(num):
                tmp = file.readline().split()
                self.data.append({'path': root_dir+tmp[0], 'bbox': np.array([int(tmp[3]), int(tmp[4]), int(tmp[5]), int(tmp[6])]), 'type': int(tmp[1])-1})
        
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data[idx]['path']
        image = io.imread(img_name)
        bbox = self.data[idx]['bbox']
        cloth_type = self.data[idx]['type']
        sample = {'image': image, 'bbox': bbox, 'type': cloth_type}

        if self.transform:
            sample = self.transform(sample)

        return sample


type_name = ["upper body", "lower body", "full body"]
def show_bbox(image, bbox, type):
    plt.imshow(image)
    plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0],bbox[3] - bbox[1],  fill=False, edgecolor='r', linewidth=1))
    plt.text(bbox[0], bbox[1], type_name[type], fontsize=10, style='oblique', ha='center',va='top',wrap=True)

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, bbox, cloth_type = sample['image'], sample['bbox'], sample['type']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        bbox = bbox * [new_w / w, new_h / h, new_w / w, new_h / h]
        bbox = bbox.astype('int32')

        return {'image': img, 'bbox': bbox, "type": cloth_type}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, bbox, cloth_type = sample['image'], sample['bbox'], sample['type']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        bbox = bbox - [left, top, left, top]
        bbox[np.where(bbox<0)] = 0
        if (bbox[0]>=self.output_size[0]):
            bbox[0] = self.output_size[0]-1
        if (bbox[1]>=self.output_size[1]):
            bbox[1] = self.output_size[1]-1
        if (bbox[2]>=self.output_size[0]):
            bbox[2] = self.output_size[0]-1
        if (bbox[3]>=self.output_size[1]):
            bbox[3] = self.output_size[1]-1
        

        return {'image': image, 'bbox': bbox, "type": cloth_type}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, bbox, cloth_type = sample['image'], sample['bbox'], sample['type']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'bbox': torch.from_numpy(bbox),
                'type': cloth_type}




import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import os
import itertools
from torch.utils.data.sampler import Sampler
from torchvision import transforms
import cv2
import numpy as np
import torch
from .randaugment import RandAugmentMC
N_CLASSES = 9
CLASS_NAMES = [ 'ADI', 'BACK', 'LYM', 'STR', 'DEB', 'MUC', 'TUM','MUS','NORM']


def cutout(img, num_holes=8, length=28):
    """
    Args:
    img (Tensor): Tensor image of size (H, W, C). input is an image
    Returns:
    Tensor: Image with n_holes of dimension length x length cut out of it.
    """
    h = img.shape[1]
    w = img.shape[2]
    c = img.shape[0]
    mask = np.ones([h, w], np.float32)
    for _ in range(num_holes):
      y = np.random.randint(h)
      x = np.random.randint(w)
      y1 = np.clip(max(0, y - length // 2), 0, h)
      y2 = np.clip(max(0, y + length // 2), 0, h)
      x1 = np.clip(max(0, x - length // 2), 0, w)
      x2 = np.clip(max(0, x + length // 2), 0, w)
      mask[y1: y2, x1: x2] = 0
    mask = np.expand_dims(mask, 0)
    mask = torch.from_numpy(mask)
    
    mask = torch.cat((mask,mask,mask), dim=0)
    img = img * mask

    return img

class GetDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        """
        Args:
            data_dir: path to image directory.
            csv_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        super(GetDataset, self).__init__()
        file = pd.read_csv(csv_file)

        self.root_dir = root_dir
        self.images = file['image'].values
        self.labels = file.iloc[:, 1:].values.astype(int)
        self.transform = transform

    def __getitem__(self, index):
        items = self.images[index]
        image_name = os.path.join(self.root_dir, self.images[index]+'.tif')
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return items, index, image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.images)

class TwoStreamBatchSampler(Sampler):
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size



class TransformTwice:
    def __init__(self, transform):
        self.weak_transform = transform
        self.strong_transform = transforms.Compose([
                                                transforms.Resize((224, 224)),
                                                transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomVerticalFlip(),
                                                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])
                                            ])
        self.base_transform=transforms.Compose([
                                              transforms.Resize((224, 224)),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])
                                          ])

    def __call__(self, inp):
        out1 = self.weak_transform(inp)
        out2 = self.strong_transform(inp)
        out2 = cutout(out2)
        return out1,out2



def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

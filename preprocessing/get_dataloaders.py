
import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')

import torch
from torch.utils.data import DataLoader, sampler, SubsetRandomSampler
from torchvision import transforms, datasets, models
import torch.nn as nn

# Data science tools
import numpy as np

# Image manipulations
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from plotting import show_transforms
from plotting import plot_images

# TODO update get dataloaders from mixmatch or others. Try more simple function
# TODO change ImageFolder to custom dataset

def get_dataloaders(data_dir, valid_part, batch_size, image_transforms, show_transform=False, show_sample=False):
    '''
        Divide ImageFolder with train set into train and validation parts using random shuffling.
    '''
    np.random.seed(12)
    torch.manual_seed(12)
    data = {
        'train':
        datasets.ImageFolder(data_dir, image_transforms['train']),
        'val':
        datasets.ImageFolder(data_dir, image_transforms['val']),
    }
    train_idx, valid_idx = [], []
    counts = (data['train'].targets.count(i) for i in data['train'].class_to_idx.values())
    acc = 0
    for numb in counts:
        valid_split = int(np.floor(valid_part * numb))
        indices = list(range(acc, acc+numb))
        acc += numb
        np.random.shuffle(indices)
        train_idx.extend(indices[:numb-valid_split])
        valid_idx.extend(indices[numb-valid_split:])

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    if show_transform:
        show_transforms(data_dir, 0, 0, image_transforms['train'])
        # exit()

    # visualize some images
    if show_sample:
        sample_loader = DataLoader(data['train'],  batch_size=9, sampler=train_sampler,)
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        plot_images(images, labels, data['train'].classes)
        # exit()

    # Dataloader iterators
    dataloaders = {
        'train': DataLoader(data['train'],  batch_size=batch_size, sampler=train_sampler, drop_last=True),
        'val': DataLoader(data['val'],  batch_size=batch_size, sampler=valid_sampler, drop_last=True),
    }
    return dataloaders
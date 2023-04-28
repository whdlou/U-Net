from torch.utils.data import Dataset
import os
from os.path import join
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision.datapoints import Mask

import cfg
import tools

def generate_data_file(root: str = 'dataset',
                       rate: float = 0.3):
    train_dir = join(root, 'train')
    test_dir = join(root, 'test')
    train_data = [img[:-4] for img in os.listdir(join(train_dir, 'images'))]
    test_data = [img[:-4] for img in os.listdir(join(test_dir, 'images'))]
    train_data, val_data = train_test_split(train_data, test_size=rate)
    tools.write_data(join(root, 'train.txt'), train_data)
    tools.write_data(join(root, 'val.txt'), val_data)
    tools.write_data(join(root, 'test.txt'), test_data)


def mask_to_onehot(mask):
    _mask_shape = mask.shape[:-1]
    _mask = np.zeros(shape=_mask_shape)
    for i, color in enumerate(cfg.palette):
        equality = (mask == color)
        class_map = np.all(equality, axis=-1)
        _mask[class_map] = i
    return _mask


class CustomDataset(Dataset):
    def __init__(self,
                 root: str = 'dataset',
                 mode: str = 'train',
                 transforms=None,
                 test_gt=False,
                 one_hot_mask=True):
        self.mode = mode
        self.transforms = transforms
        self.test_gt = test_gt
        self.one_hot_mask = one_hot_mask
        if self.mode not in ('train', 'val', 'test'):
            raise ValueError("The value of mode must be 'train', 'val' or 'test', but not be {}".format(self.mode))
        with open(join(root, self.mode + '.txt')) as f:
            self.data_names = [data.rstrip() for data in f.readlines()]
        if self.mode == 'train' or self.mode == 'val':
            sub_dir = 'train'
        else:
            sub_dir = 'test'
        self.img_list = [join(root, sub_dir, 'images', data + cfg.img_type) for data in self.data_names]
        if self.mode == 'test' and test_gt == False:
            self.mask_list = None
        else:
            self.mask_list = [join(root, sub_dir, 'masks', data + cfg.mask_type) for data in self.data_names]

    def __getitem__(self, item):
        # TODO: input gray mask
        img = self.img_list[item]
        img = Image.open(img).convert('RGB')
        img = np.array(img).astype(np.float32)
        data_name = self.data_names[item]
        if self.mask_list is not None:
            mask = self.mask_list[item]
            mask = Image.open(mask)
            if self.one_hot_mask is False:
                mask = mask.convert('RGB')
            mask = np.array(mask)
            if self.one_hot_mask is False:
                mask = mask_to_onehot(mask)
            mask = mask.astype(np.int64)
            mask = Mask(mask)
            if self.transforms is not None:
                img, mask = self.transforms(img, mask)
            data = {
                'image': img,
                'mask': mask,
                'name': data_name
            }
        else:
            if self.transforms is not None:
                img = self.transforms(img)
            data = {
                'image': img,
                'name': data_name
            }

        return data

    def __len__(self):

        return len(self.data_names)


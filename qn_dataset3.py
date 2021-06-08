# -*- coding: utf-8 -*-

import torch.utils.data as data
import torch
from torchvision import transforms
from PIL import Image
import glob
import cv2
import hutils
import random
import numpy as np
import h5py
import h_psnr


class qnDataset(data.Dataset):
    def __init__(self, h5file, interval=0):
        super(qnDataset, self).__init__()
        self.interval = interval
        self.patch_size = 96
        self._config = {
            'scale' : 2,
            'blur' : True,
            'noise' : True,
            'jpeg' : True,
            'camera' : True,
            'rotate' : True,
            'hflip' : True
        }
        self.h5_file = h5file

    def __getitem__(self, idx):
        # HWC
        with h5py.File(self.h5_file, 'r') as f:
            randint = np.random.randint(0, self.interval + 1)
            img_GT = f['hr'][idx * (self.interval + 1) + randint]
        H, W, C = img_GT.shape

        return img_GT

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['hr']) // (self.interval + 1)

    def config(self, **kwargs):
        for k, v in kwargs.items():
            self._config[k] = v
        print(self._config)


def test():
    ds = qnDataset("../qn_dataset/vsr_train_hwcbgr.h5")
    ds.config(scale=1)

if __name__=="__main__":
    test()

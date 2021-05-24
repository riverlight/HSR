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


class noiseDataset(data.Dataset):
    def __init__(self, dataset=None, size=96):
        super(noiseDataset, self).__init__()

        base = dataset
        import os
        assert os.path.exists(base)

        # self.mat_files = sorted(glob.glob(base + '*.mat'))
        self.noise_imgs = sorted(glob.glob(base + '*.png'))
        self.pre_process = transforms.Compose([transforms.RandomCrop(size),
                                               transforms.ToTensor()])

    def __getitem__(self, index):
        # mat = loadmat(self.mat_files[index])
        # x = np.array([mat['kernel']])
        # x = np.swapaxes(x, 2, 0)
        # print(np.shape(x))
        print(self.noise_imgs[index])
        noise = self.pre_process(Image.open(self.noise_imgs[index]))
        norm_noise = (noise - torch.mean(noise, dim=[1, 2], keepdim=True))
        return norm_noise, noise

    def __len__(self):
        return len(self.noise_imgs)


# 得到 SR 数据集
class qnSRDataset(data.Dataset):
    def __init__(self, cfgDict):
        super(qnSRDataset, self).__init__()
        assert self.check_cfg(cfgDict)
        self.cfgDict = cfgDict
        self.paths_LQ, self.paths_GT = None, None
        self.sizes_LQ, self.sizes_GT = None, None

        self.paths_GT, self.sizes_GT = hutils.get_image_paths('img', cfgDict['dataroot_GT'])
        self.paths_LQ, self.sizes_LQ = hutils.get_image_paths('img', cfgDict['dataroot_LQ'])
        assert self.paths_GT, 'Error: GT path is empty.'
        if self.paths_LQ and self.paths_GT:
            assert len(self.paths_LQ) == len(
                self.paths_GT
            ), 'GT and LQ datasets have different number of images - {}, {}.'.format(
                len(self.paths_LQ), len(self.paths_GT))
        if cfgDict['noise']:
            self.noises = noiseDataset(cfgDict['noise_data'], cfgDict['patch_size'] / cfgDict['scale'])
            print(len(self.noises))
        self.random_scale_list = [1]

    def check_cfg(self, cfgDict):
        assert cfgDict.get("dataroot_GT", False)
        assert cfgDict.get('dataroot_LQ', False) != False
        assert cfgDict.get('noise', None) != None
        assert cfgDict.get('noise_data', False)
        assert cfgDict.get('patch_size', False)
        assert cfgDict.get('scale', None)
        return True

    def __getitem__(self, index):
        GT_path, LQ_path = None, None
        scale = self.cfgDict['scale']
        GT_size = self.cfgDict['patch_size']

        # get GT image
        GT_path = self.paths_GT[index]
        img_GT = hutils.read_img(GT_path)
        img_GT = hutils.modcrop(img_GT, scale)
        img_GT = hutils.channel_convert(img_GT.shape[2], 'RGB', [img_GT])[0]

        # get LQ image
        if self.paths_LQ:
            LQ_path = self.paths_LQ[index]
            img_LQ = hutils.read_img(LQ_path)
        else:  # down-sampling on-the-fly
            H, W, _ = img_GT.shape
            # using matlab imresize
            img_LQ = hutils.imresize_np(img_GT, 1 / scale, True)
            if img_LQ.ndim == 2:
                img_LQ = np.expand_dims(img_LQ, axis=2)

        H, W, C = img_LQ.shape
        LQ_size = GT_size // scale

        # randomly crop
        rnd_h = random.randint(0, max(0, H - LQ_size))
        rnd_w = random.randint(0, max(0, W - LQ_size))
        img_LQ = img_LQ[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
        rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
        img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]
        print(rnd_h, rnd_w,rnd_h_GT, rnd_w_GT)

        # augmentation - flip, rotate
        img_LQ, img_GT = hutils.augment([img_LQ, img_GT], True, True)

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LQ = img_LQ[:, :, [2, 1, 0]]
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()

        # noise injection
        if self.cfgDict['noise']:
            print(len(self.noises))
            norm_noise, _ = self.noises[np.random.randint(0, len(self.noises))]
            img_LQ = torch.clamp(img_LQ + norm_noise, 0, 1)

        if LQ_path is None:
            LQ_path = GT_path

        # cv2.imshow("1", np.transpose(img_GT.numpy(), (2, 1, 0)))
        # cv2.imshow("2", np.transpose(img_LQ.numpy(), (2, 1, 0)))
        # cv2.waitKey()
        # exit(0)
        return {'LQ': img_LQ, 'GT': img_GT, 'LQ_path': LQ_path, 'GT_path': GT_path}

    def __len__(self):
        return len(self.paths_GT)


def testNoise():
    noiDS = noiseDataset(dataset="D:\\workroom\\tools\\image\\Real-SR\\datasets\DF2K\Corrupted_noise\\", size=256)
    print(noiDS.__len__())
    _, img = noiDS[0]
    print(img.shape)
    img = img.numpy().transpose(1, 2, 0)
    img = img[:,:,[2,1,0]]
    print(img.shape)
    cv2.imshow("1", img)
    # Image._show(img)
    cv2.waitKey()
    print(len(noiDS))


def testSRDS():
    cfgDict = {
        "dataroot_GT" : "D:\\workroom\\tools\\image\\Real-SR\\datasets\\DF2K\\generated\\tdsr\\HR",
        "dataroot_LQ" : None,
        "noise" : True,
        "noise_data" : "D:\\workroom\\tools\\image\\Real-SR\\datasets\DF2K\Corrupted_noise\\",
        "patch_size" : 256,
        'scale' : 2
    }
    ds = qnSRDataset(cfgDict)
    print(ds[0])

if __name__=="__main__":
    # testNoise()
    testSRDS()

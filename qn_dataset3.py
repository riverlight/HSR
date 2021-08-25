# -*- coding: utf-8 -*-

import torch.utils.data as data
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from PIL import Image
import glob
import cv2
import hutils
import random
import numpy as np
import h5py
import h_psnr
from preprocess.degrade import config_to_seq, degradation_pipeline, print_degrade_seg, get_blur
from qn_dataset2 import noiseDataset


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

        if self._config['hflip'] or self._config['rotate']:
            [img_GT] = hutils.augment([img_GT], self._config['hflip'], self._config['rotate'])

        H, W, C = img_GT.shape
        seq = config_to_seq(self._config)
        # print_degrade_seg(seq)
        img_Out = degradation_pipeline(img_GT, degrade_seq=seq)

        # HWC BGR -> CHW RGB
        img_GT = img_GT[:, :, [2, 1, 0]]
        img_Out = img_Out[:, :, [2, 1, 0]]
        img_GT = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_GT.astype(np.float32) / 255, (2, 0, 1)))).float()
        img_Out = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_Out.astype(np.float32) / 255, (2, 0, 1)))).float()

        return {'LQ' if self._config['scale']!=1 else 'NI' : img_Out, 'GT': img_GT}

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['hr']) // (self.interval + 1)

    def config(self, **kwargs):
        for k, v in kwargs.items():
            self._config[k] = v
        print(self._config)


class qnDataset2(data.Dataset):
    def __init__(self, h5file, interval=0, scale=2, noisedir=None):
        super(qnDataset2, self).__init__()
        self.interval = interval
        self.patch_size = 96
        self._config = {
            'scale': scale,
            'blur' : True,
            'noise' : True,
            'jpeg' : True,
            'camera' : True,
            'rotate' : True,
            'hflip' : True
        }
        self.scale = scale
        self.h5_file = h5file
        self.noises = noiseDataset(noisedir, self.patch_size / self.scale) if noisedir else None

    def __getitem__(self, idx):
        # HWC
        with h5py.File(self.h5_file, 'r') as f:
            randint = np.random.randint(0, self.interval + 1)
            img_GT = f['hr'][idx * (self.interval + 1) + randint]

        if self._config['hflip'] or self._config['rotate']:
            [img_GT] = hutils.augment([img_GT], self._config['hflip'], self._config['rotate'])

        H, W, C = img_GT.shape
        if self.scale!=1:
            img_Out = cv2.resize(img_GT, (H // self.scale, W // self.scale))
        else:
            img_Out = np.copy(img_GT)
        if self._config['jpeg']:
            ret, lr_buf = cv2.imencode(".jpg", img_Out, [int(cv2.IMWRITE_JPEG_QUALITY), np.random.randint(8, 15)])
            # ret, lr_buf = cv2.imencode(".png", img_Out)
            img_Out = cv2.imdecode(lr_buf, 1)
        if self._config['blur']:
            img_Out = get_blur(img_Out)

        # HWC BGR -> CHW RGB
        img_GT = img_GT[:, :, [2, 1, 0]]
        img_Out = img_Out[:, :, [2, 1, 0]]
        img_GT = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_GT.astype(np.float32) / 255, (2, 0, 1)))).float()
        img_Out = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_Out.astype(np.float32) / 255, (2, 0, 1)))).float()

        # noise injection
        if self.noises:
            # print(len(self.noises))
            norm_noise, _ = self.noises[np.random.randint(0, len(self.noises))]
            img_Out = torch.clamp(img_Out + norm_noise, 0, 1)

        return {'LQ' if self.scale!=1 else 'NI' : img_Out, 'GT': img_GT}

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['hr']) // (self.interval + 1)

    def config(self, **kwargs):
        for k, v in kwargs.items():
            self._config[k] = v
        print(self._config)


class qnH264Dataset(data.Dataset):
    def __init__(self, h5file, interval=0, scale=1, noisedir=None):
        super(qnH264Dataset, self).__init__()
        self.interval = interval
        self.patch_size = 96
        self.h5_file = h5file
        self.scale = scale
        self.noises = noiseDataset(noisedir, self.patch_size / self.scale) if noisedir else None

    def __getitem__(self, idx):
        # HWC BGR
        with h5py.File(self.h5_file, 'r') as f:
            randint = np.random.randint(0, self.interval + 1)
            img_GT = f['hr'][idx * (self.interval + 1) + randint]
            if self.scale==1:
                img_SRC = f['ni'][idx * (self.interval + 1) + randint]
            else:
                img_SRC = f['lr'][idx * (self.interval + 1) + randint]

        # HWC BGR -> CHW RGB
        img_GT = img_GT[:, :, [2, 1, 0]]
        img_SRC = img_SRC[:, :, [2, 1, 0]]
        img_GT = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_GT.astype(np.float32) / 255, (2, 0, 1)))).float()
        img_SRC = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_SRC.astype(np.float32) / 255, (2, 0, 1)))).float()

        # noise injection
        if self.noises:
            # print(len(self.noises))
            norm_noise, _ = self.noises[np.random.randint(0, len(self.noises))]
            img_SRC = torch.clamp(img_SRC + norm_noise, 0, 1)

        return {'NI' if self.scale==1 else 'LQ' : img_SRC, 'GT': img_GT}

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['hr']) // (self.interval + 1)


class qnVSRDataset(data.Dataset):
    def __init__(self, h5file, interval=0):
        super(qnVSRDataset, self).__init__()
        self.interval = interval
        self.patch_size = 96
        self.h5_file = h5file

    def __getitem__(self, idx):
        # HWC
        with h5py.File(self.h5_file, 'r') as f:
            randint = np.random.randint(0, self.interval + 1)
            img_GT = f['hr'][idx * (self.interval + 1) + randint]
            img_LQ = f['lr'][idx * (self.interval + 1) + randint]

        [img_GT, img_LQ] = hutils.augment([img_GT, img_LQ], True, True)
        # HWC BGR -> CHW RGB
        img_GT = img_GT[:, :, [2, 1, 0]]
        img_LQ = img_LQ[:, :, [2, 1, 0]]
        img_GT = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_GT.astype(np.float32) / 255, (2, 0, 1)))).float()
        img_LQ = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_LQ.astype(np.float32) / 255, (2, 0, 1)))).float()
        return {'LQ' : img_LQ, 'GT' : img_GT}

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['hr']) // (self.interval + 1)

def test():
    scale = 2
    ds = qnDataset2("./qn_dataset/vsr_train_hwcbgr.h5")
    ds.config(scale=scale, noise=True, blur=True, camera=True, jpeg=True)

    d0 = ds[0]
    if scale==1:
        cv2.imshow("NI", np.transpose(d0['NI'][(2, 1, 0), :, :].numpy(), (1, 2, 0)))
    else:
        cv2.imshow("LQ", np.transpose(d0['LQ'][(2, 1, 0), :, :].numpy(), (1, 2, 0)))
    cv2.imshow("GT", np.transpose(d0['GT'][(2, 1, 0), :, :].numpy(), (1, 2, 0)))
    cv2.waitKey()
    if scale==1:
        lr = d0['NI']
    else:
        lr = torch.nn.functional.interpolate(d0['LQ'].unsqueeze(0), scale_factor=scale, mode="bicubic", align_corners=False)
    print(h_psnr.calc_psnr_tensor(lr, d0['GT']))
    print(ds[1]['GT'].shape)
    print(ds[2]['GT'].shape)
    exit(0)

def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

def calc_ds_psnr():
    ds = qnDataset2("./qn_dataset/vsr_val_hwcbgr.h5")
    ds.config(scale=2, noise=True, blur=True, camera=True, jpeg=True)
    dl = DataLoader(dataset=ds, batch_size=1)
    lst_bic_psnr = list()
    for count, data in enumerate(dl):
        if count%1000 == 0:
            print("count : ", count)
            # continue
        if count>1000:
            break
        hr_img, lr_img = data['GT'], data['LQ']

        from torch.nn.functional import interpolate
        bic_img = interpolate(lr_img, scale_factor=2, mode="bicubic", align_corners=False)
        psnr = calc_psnr(hr_img, bic_img)

        # hr_img = hr_img.numpy().astype(np.float32) * 255
        # lr_img = lr_img.numpy().astype(np.float32) * 255
        # hr_img = hr_img.transpose(0, 2, 3, 1)
        # lr_img = lr_img.transpose(0, 2, 3, 1)
        # psnr = h_psnr.calc_psnr_np_upsample(hr_img[0, ...], lr_img[0, ...]).item()
        if psnr>100:
            print("count : {}, psnr : {}".format(count, psnr))
            continue
        # print(hr_img.shape, lr_img.shape, psnr)
        lst_bic_psnr.append(psnr)
        # cv2.imshow("lr", lr.astype(np.uint8))
        # cv2.imshow('hr', hr_img[0, ...].astype(np.uint8))
        # cv2.waitKey()
    print("mean psnr : ", sum(lst_bic_psnr)/len(lst_bic_psnr))
    print(sum(lst_bic_psnr), len(lst_bic_psnr))

def test_vsrds():
    ds = qnVSRDataset("./qn_dataset/vsr_dy_train_hwcbgr.h5")
    d0 = ds[0]
    cv2.imshow("LQ", np.transpose(d0['LQ'][(2, 1, 0), :, :].numpy(), (1, 2, 0)))
    cv2.imshow("GT", np.transpose(d0['GT'][(2, 1, 0), :, :].numpy(), (1, 2, 0)))
    cv2.waitKey()

def test_h264ds():
    noisedir = "d:/workroom/tools/image/Real-SR/datasets/DF2K/Corrupted_noise/"
    ds = qnH264Dataset("./qn_dataset/sr_h264_val_hwcbgr.h5", scale=2, noisedir=noisedir)
    d0 = ds[3]
    cv2.imshow("LQ", np.transpose(d0['LQ'][(2, 1, 0), :, :].numpy(), (1, 2, 0)))
    cv2.imshow("GT", np.transpose(d0['GT'][(2, 1, 0), :, :].numpy(), (1, 2, 0)))
    cv2.waitKey()

def calc_vds_psnr():
    ds = qnVSRDataset("./qn_dataset/vsr_dy_val_hwcbgr.h5")
    dl = DataLoader(dataset=ds, batch_size=1)
    lst_bic_psnr = list()
    for count, data in enumerate(dl):
        if count%1000 == 0:
            print("count : ", count)
            # continue
        if count>100000:
            break
        hr_img, lr_img = data['GT'], data['LQ']

        from torch.nn.functional import interpolate
        bic_img = interpolate(lr_img, scale_factor=2, mode="bicubic", align_corners=False)
        psnr = calc_psnr(hr_img, bic_img)

        # hr_img = hr_img.numpy().astype(np.float32) * 255
        # lr_img = lr_img.numpy().astype(np.float32) * 255
        # hr_img = hr_img.transpose(0, 2, 3, 1)
        # lr_img = lr_img.transpose(0, 2, 3, 1)
        # psnr = h_psnr.calc_psnr_np_upsample(hr_img[0, ...], lr_img[0, ...]).item()
        if psnr>100:
            print("count : {}, psnr : {}".format(count, psnr))
            continue
        # print(hr_img.shape, lr_img.shape, psnr)
        lst_bic_psnr.append(psnr)
        # cv2.imshow("lr", lr.astype(np.uint8))
        # cv2.imshow('hr', hr_img[0, ...].astype(np.uint8))
        # cv2.waitKey()
    print("mean psnr : ", sum(lst_bic_psnr)/len(lst_bic_psnr))
    print(sum(lst_bic_psnr), len(lst_bic_psnr))

if __name__=="__main__":
    # test()
    # calc_ds_psnr()
    # test_vsrds()
    # calc_vds_psnr()
    test_h264ds()


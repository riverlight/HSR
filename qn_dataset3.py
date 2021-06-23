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
from preprocess.degrade import config_to_seq, degradation_pipeline, print_degrade_seg


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
    def __init__(self, h5file, interval=0):
        super(qnDataset2, self).__init__()
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
        self.scale = 2
        self.h5_file = h5file

    def __getitem__(self, idx):
        # HWC
        with h5py.File(self.h5_file, 'r') as f:
            randint = np.random.randint(0, self.interval + 1)
            img_GT = f['hr'][idx * (self.interval + 1) + randint]

        if self._config['hflip'] or self._config['rotate']:
            [img_GT] = hutils.augment([img_GT], self._config['hflip'], self._config['rotate'])

        H, W, C = img_GT.shape
        img_Out = cv2.resize(img_GT, (H // self.scale, W // self.scale))
        ret, lr_buf = cv2.imencode(".jpg", img_Out, [int(cv2.IMWRITE_JPEG_QUALITY), np.random.randint(5, 95)])
        # ret, lr_buf = cv2.imencode(".png", img_Out)
        img_Out = cv2.imdecode(lr_buf, 1)

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



def test():
    scale = 2
    ds = qnDataset2("./qn_dataset/vsr_train_hwcbgr.h5")
    ds.config(scale=scale, noise=True, blur=True, camera=True, jpeg=True)

    d0 = ds[150]
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

if __name__=="__main__":
    # test()
    calc_ds_psnr()


# -*- coding: utf-8 -*-

import cv2
import torch as T
import h5py
import os
import numpy as np

patch_size = 96
stride = 96*2
train_dir = "D:\\workroom\\tools\\dataset\\SR\\qnSR_DS\\train"
train_h5 = "../qn_dataset/train.h5"
eval_dir = "D:\\workroom\\tools\\dataset\\SR\\qnSR_DS\\eval"
eval_h5 = "../qn_dataset/eval.h5"

def make_h5(dir, h5name, interval=0):
    h5_file = h5py.File(h5name, 'w')
    hr_patchs = list()
    lr_patchs = list()

    for count, name in enumerate(os.listdir(dir)):
        imagename = os.path.join(dir, name)
        if os.path.isdir(imagename):
            continue
        if count%(interval+1) != 0:
            continue
        hr_img = cv2.imread(imagename)
        # print(hr_img.shape)
        lr_img = cv2.resize(hr_img, (hr_img.shape[1]//2, hr_img.shape[0]//2))
        # print(lr_img.shape)
        ret, lr_buf = cv2.imencode(".JPG", lr_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        lr_img = cv2.imdecode(lr_buf, 1)
        hr_img = hr_img.transpose(2, 0, 1)
        lr_img = lr_img.transpose(2, 0, 1)

        for i in range(0, hr_img.shape[1] - patch_size + 1, stride):
            for j in range(0, hr_img.shape[2] - patch_size + 1, stride):
                hr_patchs.append(hr_img[:, i:i + patch_size, j:j + patch_size])
                lr_patchs.append(lr_img[:, i//2:(i + patch_size)//2, j//2:(j + patch_size)//2])

    hr_ds = np.array(hr_patchs, dtype=np.uint8)
    lr_ds = np.array(lr_patchs, dtype=np.uint8)
    h5_file.create_dataset('hr', data=hr_ds)
    h5_file.create_dataset('lr', data=lr_ds)
    h5_file.close()


if __name__=="__main__":
    print("Hi, this is SR qn-dataset generator program")
    make_h5(train_dir, train_h5, interval=0)
    make_h5(eval_dir, eval_h5, interval=0)

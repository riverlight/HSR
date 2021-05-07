# -*- coding: utf-8 -*-


import numpy as np
from torch.nn.functional import interpolate
import sys
sys.path.append("../")
import qn_dataset
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import cv2
import h_psnr


def test():
    ds = qn_dataset.QNDataset("..\\qn_dataset\\train.h5")
    dl = DataLoader(dataset=ds, batch_size=1)
    for data in dl:
        hr_img, lr_img = data
        bic_img = interpolate(lr_img, scale_factor=2, mode="bicubic", align_corners=False)
        print(h_psnr.calc_psnr_tensor(bic_img, hr_img))

        hr_img = hr_img.numpy() * 255
        lr_img = lr_img.numpy() * 255
        bic_img = bic_img.numpy() * 255
        hr_img = hr_img.transpose(0, 2, 3, 1)
        lr_img = lr_img.transpose(0, 2, 3, 1)
        bic_img = bic_img.transpose(0, 2, 3, 1)
        # cv2.imshow("hr", hr_img[0, ...].astype(np.uint8))
        # cv2.imshow("lr", lr_img[0, ...].astype(np.uint8))
        # cv2.imshow("bic", bic_img[0, ...].astype(np.uint8))
        # cv2.waitKey(0)
        # break

    pass

if __name__=="__main__":
    test()

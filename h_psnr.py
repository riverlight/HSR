# -*- coding: utf-8 -*-

import torch as T
import cv2
import numpy as np
import math
import os

def calc_psnr_tensor(img1, img2):
    return 10. * T.log10(1. / T.mean((img1 - img2) ** 2))

def psnr1(img1, img2):
   mse = np.mean((img1/1.0 - img2/1.0) ** 2 )
   if mse < 1.0e-10:
      return 100
   return 10 * math.log10(255.0**2/mse)

def calc_psnr_file_upsample(img1, img2):
    np1 = cv2.imread(img1)
    np2 = cv2.imread(img2)
    h1, w1, _ = np1.shape
    h2, w2, _ = np2.shape
    if h1==h2 and w1==w2:
        pass
    elif h1*w1 < h2*w2:
        np1 = cv2.resize(np1, (w2, h2), interpolation=cv2.INTER_CUBIC)
    else:
        np2 = cv2.resize(np2, (w1, h1), interpolation=cv2.INTER_CUBIC)

    psnr = calc_psnr_array(np1, np2)
    return psnr

def calc_psnr_file(img1, img2):
    np1 = cv2.imread(img1)
    np2 = cv2.imread(img2)
    h1, w1, _ = np1.shape
    h2, w2, _ = np2.shape
    if h1==h2 and w1==w2:
        pass
    elif h1*w1 > h2*w2:
        np1 = cv2.resize(np1, (w2, h2))
    else:
        # print(np1.shape, np2.shape)
        np2 = cv2.resize(np2, (w1, h1))
        # cv2.imshow("1", np2)
        # cv2.waitKey(0)
        # print(np1.shape, np2.shape)

    psnr = calc_psnr_array(np1, np2)
    return psnr

def calc_psnr_array(img1, img2):
    t1 = T.from_numpy(img1.astype(np.float32)) / 255.0
    t2 = T.from_numpy(img2.astype(np.float32)) / 255.0
    psnr = calc_psnr_tensor(t1, t2)
    return psnr


def test():
    dir = "D:\\workroom\\tools\\dataset\\SR\\SI\\"
    f1 = dir + "tmp/film-rcan.png"
    f2 = dir + "tmp/film-lr.png"
    f2 = dir + "film.png"
    psnr = calc_psnr_file_upsample(f1, f2)
    print("psnr : ", psnr)
    f1 = dir + "tmp/film-bd.jpg"
    print(calc_psnr_file_upsample(f1, f2))


def test_total():
    dir0 = "D:\\workroom\\tools\\dataset\\SR\\SI\\"
    dir1 = "D:\\workroom\\tools\\dataset\\SR\\SI\\tmp\\"
    for i, name in enumerate(os.listdir(dir0)):
        hr_name = os.path.join(dir0, name)
        if os.path.isdir(hr_name):
            continue
        print(name)
        lr_name = os.path.join(dir1, name.replace('.jpg', '-lr.jpg').replace('.png', '-lr.png'))
        rcan_name = os.path.join(dir1, name.replace('.jpg', '-rcan.jpg').replace('.png', '-rcan.png'))
        bd_name = os.path.join(dir1, name.replace('.jpg', '-bd.jpg').replace('.png', '-bd.jpg'))
        print(calc_psnr_file(bd_name, hr_name))
        print(calc_psnr_file(bd_name, lr_name))


if __name__=="__main__":
    # test()
    test_total()

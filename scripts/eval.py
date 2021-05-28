# -*- coding: utf-8 -*-


import torch as t
import sys
sys.path.append("../")
from models import HSISRNet, HRcanNet
import cv2
import numpy as np
import h_psnr
import os


def eval_np(lr_img, eval_file=None, device='cuda'):
    if eval_file is None:
        eval_file = "../weights/hsi3_epoch_117.pth"
    net = t.load(eval_file)
    net = net.to(device)
    net.eval()
    image = t.from_numpy(lr_img).to(device) / 255
    image = image.permute(2, 0, 1).unsqueeze(0)
    # print(image.shape)

    with t.no_grad():
        preds = net(image).clamp(0.0, 1.0)
        # preds = net(image)

    # print(preds.shape)
    hsi_img = preds.mul(255.0).cpu().numpy().squeeze(0).transpose(1, 2, 0)
    return hsi_img


def eval_image(lr_file=None, eval_file=None, device='cuda'):
    if lr_file is None:
        lr_file = "d:/workroom/testroom/old.png"
    out_file = lr_file.replace('.png', '_hsi.png').replace('.jpg', '_hsi.jpg')
    if eval_file is None:
        eval_file = "../weights/hsi3_noi_5_10.pth"
    net = t.load(eval_file)
    net = net.to(device)
    net.eval()
    image = cv2.imread(lr_file).astype(np.float32)
    # image = image[:,:,[2,1,0]]
    image = t.from_numpy(image).to(device) / 255
    image = image.permute(2, 0, 1).unsqueeze(0)
    # print(image.shape)

    with t.no_grad():
        preds = net(image).clamp(0.0, 1.0)
        # preds = net(image)

    # print(preds.shape)
    preds = preds.mul(255.0).cpu().numpy().squeeze(0).transpose(1, 2, 0)
    # preds = preds[:, :, [2, 1, 0]]
    cv2.imwrite(out_file, preds)
    print('done : ', out_file)
    return out_file

def eval_psnr():
    dir0 = "D:\\workroom\\tools\\dataset\\SR\\SI\\"
    dir1 = "D:\\workroom\\tools\\dataset\\SR\\SI\\tmp\\"
    # dir0 = "D:\\workroom\\tools\\dataset\\SR\\Set5\\image_SRF_2\\"
    # dir1 = "D:\\workroom\\tools\\dataset\\SR\\Set5\\image_SRF_2\\"

    lr_file = dir1 + "face-lr.jpg"
    hr_file = dir0 + 'face.jpg'
    hsi_file = eval_image(lr_file=lr_file)
    psnr = h_psnr.calc_psnr_file(hr_file, hsi_file)
    print(psnr)

    lst_file = [dir1 + "face-bd.jpg"]
    for file in lst_file:
        psnr = h_psnr.calc_psnr_file(hr_file, file)
        print(psnr)


def eval_cmp_bic_hsi_dir():
    dir = "D:\\workroom\\tools\\dataset\\SR\\srgan\\BSD100"
    dir = "D:\\workroom\\tools\\dataset\\SR\\qnSR_DS\\eval"
    # dir = "D:\\workroom\\tools\\dataset\\SR\\SI"
    lst_bic = list()
    lst_hsi = list()
    for count, name in enumerate(os.listdir(dir)):
        imagefile = os.path.join(dir, name)
        if os.path.isdir(imagefile):
            continue
        bp, hp = eval_cmp_bic_hsi(imagefile)
        lst_bic.append(bp)
        lst_hsi.append(hp)
    print("bic mean : ", sum(lst_bic)/len(lst_bic))
    print("hsi mean : ", sum(lst_hsi)/len(lst_hsi))

def eval_cmp_bic_hsi(imagefile):
    print(imagefile)
    hr_img = cv2.imread(imagefile).astype(np.float32)
    lr_img = cv2.resize(hr_img, (hr_img.shape[1]//2, hr_img.shape[0]//2))
    hsi_img = eval_np(lr_img)
    bic_psnr = h_psnr.calc_psnr_np_upsample(hr_img, lr_img)
    # return bic_psnr.item()
    hsi_psnr = h_psnr.calc_psnr_np_upsample(hr_img, hsi_img)
    print(bic_psnr.item(), hsi_psnr.item())
    return bic_psnr.item(), hsi_psnr.item()


if __name__=="__main__":
    # eval_image(lr_file="D:\\workroom\\tools\\image\\ntire20\\track1-valid-input\\0855.png")
    # eval_image()
    eval_psnr()
    # eval_cmp_bic_hsi_dir()

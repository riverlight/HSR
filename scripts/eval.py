# -*- coding: utf-8 -*-


import torch as t
import sys
sys.path.append("../")
from models import HSISRNet
import cv2
import numpy as np
import h_psnr


def eval_image(lr_file=None, eval_file=None, device='cuda'):
    if lr_file is None:
        lr_file = "d:/workroom/testroom/old.png"
    out_file = lr_file.replace('.png', '_hsi.png').replace('.jpg', '_hsi.jpg')
    if eval_file is None:
        eval_file = "../weights/hsi_best.pth"
    net = t.load(eval_file)
    net = net.to(device)
    net.eval()
    image = cv2.imread(lr_file).astype(np.float32)
    image = t.from_numpy(image).to(device) / 255
    image = image.permute(2, 0, 1).unsqueeze(0)
    # print(image.shape)

    with t.no_grad():
        preds = net(image).clamp(0.0, 1.0)
        # preds = net(image)

    # print(preds.shape)
    preds = preds.mul(255.0).cpu().numpy().squeeze(0).transpose(1, 2, 0)
    cv2.imwrite(out_file, preds)
    print('done : ', out_file)
    return out_file

def eval_psnr():
    dir0 = "D:\\workroom\\tools\\dataset\\SR\\SI\\"
    dir1 = "D:\\workroom\\tools\\dataset\\SR\\SI\\tmp\\"
    # dir0 = "D:\\workroom\\tools\\dataset\\SR\\Set5\\image_SRF_2\\"
    # dir1 = "D:\\workroom\\tools\\dataset\\SR\\Set5\\image_SRF_2\\"

    lr_file = dir1 + "car-lr.jpg"
    hr_file = dir0 + 'car.jpg'
    hsi_file = eval_image(lr_file=lr_file)
    psnr = h_psnr.calc_psnr_file(hr_file, hsi_file)
    print(psnr)

    lst_file = [dir1 + "car-bd.jpg"]
    for file in lst_file:
        psnr = h_psnr.calc_psnr_file(hr_file, file)
        print(psnr)


if __name__=="__main__":
    eval_psnr()

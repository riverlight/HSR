# -*- coding: utf-8 -*-


import torch as t
import sys
sys.path.append("../")
from models_ir import HRcanIRNet
import cv2
import numpy as np
import h_psnr
import os



default_eval_file = "../weights/qir_epoch_5.pth"


def eval_image(imagename):
    device = 'cuda'
    out_file = imagename.replace('.png', '_ir.png').replace('.jpg', '_ir.jpg')
    net = t.load(default_eval_file)
    net = net.to(device)
    net.eval()
    image = cv2.imread(imagename).astype(np.float32)
    image = image[:,:,[2,1,0]]
    image = t.from_numpy(image).to(device) / 255
    image = image.permute(2, 0, 1).unsqueeze(0)
    # print(image.shape)

    with t.no_grad():
        preds = net(image).clamp(0.0, 1.0)
        # preds = net(image)

    # print(preds.shape)
    preds = preds.mul(255.0).cpu().numpy().squeeze(0).transpose(1, 2, 0)
    preds = preds[:, :, [2, 1, 0]]
    cv2.imwrite(out_file, preds)
    print('done : \n', out_file)
    return out_file

if __name__=="__main__":
    eval_image("d:/workroom/testroom/old.png")

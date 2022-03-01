# -*- coding: utf-8 -*-


import torch as t
import sys
sys.path.append("../")
from models_ir import HRcanIRNet
from models import HRcanNet
import cv2
import numpy as np
import h_psnr
import os
import time


default_eval_file = "../weights/deblur_HRcanNet_epoch_104.pth"


class CIR:
    def __init__(self, weights_file=default_eval_file):
        self._device = 'cuda'
        # self._net = HSISRNet().to(self._device)
        self._net = t.load(weights_file).to(self._device)
        self._net.eval()

    def query(self, img):
        image = img.astype(np.float32)
        image = image[:, :, [2, 1, 0]]
        image = t.from_numpy(image).to(self._device) / 255
        image = image.permute(2, 0, 1).unsqueeze(0)
        # print(image.shape)

        with t.no_grad():
            preds = self._net(image).clamp(0.0, 1.0)
            # preds = net(image)

        # print(preds.shape)
        preds = preds.mul(255.0).cpu().numpy().squeeze(0).transpose(1, 2, 0)
        preds = preds[:, :, [2, 1, 0]]
        del image
        return preds.astype(np.uint8)

    def query_file(self, imagename):
        image = cv2.imread(imagename)
        return self.query(image)



def eval_video():
    scale = 1
    s_mp4 = "d:/workroom/testroom/fei-enc.mp4"
    cap = cv2.VideoCapture(s_mp4)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*scale)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*scale)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(s_mp4.replace('.mp4', '_ir.avi'), fourcc, fps, (width, height))
    net = CIR()

    count = 0
    starttime = time.time()
    while True:
        if count % 25 == 0:
            print('frame id ', count)
        ret, frame = cap.read()
        if ret is not True:
            break

        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        print('ts :', ts)
        print("cost time : ", time.time() - starttime)
        pred_img = net.query(frame)
        out.write(pred_img)
        count += 1
        if ts > 30:
            break
    out.release()
    cap.release()


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
    eval_image("d:/workroom/testroom/a.jpg")
    # eval_video()

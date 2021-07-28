# -*- coding: utf-8 -*-

import torch as t
import sys
sys.path.append("../")
from models import HRcanNet
import cv2
import numpy as np
import h_psnr
import os
import time


class CSISR:
    def __init__(self, weights_file="../weights/vsr_HRcanNet_best_nofea.pth"):
        self._device = 'cuda'
        # self._net = HSISRNet().to(self._device)
        self._net = t.load(weights_file).to(self._device)
        self._net.eval()

    def query(self, img):
        image = img.astype(np.float32)
        image = t.from_numpy(image).to(self._device) / 255
        image = image.permute(2, 0, 1).unsqueeze(0)
        with t.no_grad():
            preds = self._net(image).clamp(0.0, 1.0)
        preds = preds.mul(255.0).cpu().numpy().squeeze(0).transpose(1, 2, 0)
        del image
        return preds.astype(np.uint8)

    def query_file(self, imagename):
        image = cv2.imread(imagename)
        return self.query(image)



def main():
    scale = 2
    s_mp4 = "d:/workroom/testroom/jie_lr.mp4"
    cap = cv2.VideoCapture(s_mp4)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*scale)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*scale)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'I420')
    out = cv2.VideoWriter(s_mp4.replace('.mp4', '_hsr.avi'), fourcc, fps, (width, height))
    net = CSISR()

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


if __name__ == "__main__":
    print("Hi, this is video IR program!")
    main()
    print('done')

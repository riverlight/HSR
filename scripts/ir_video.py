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


class CIR:
    def __init__(self, weights_file="../weights/dejpeg_HRcanNet2_1_33_best.pth"):
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



def main(s_mp4=None):
    scale = 1
    if s_mp4 is None:
        s_mp4 = "d:/workroom/testroom/156_45.mp4"
    if '.mp4' not in s_mp4:
        raise Exception("input video must be mp4 format")
    cap = cv2.VideoCapture(s_mp4)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*scale)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*scale)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(s_mp4.replace('.mp4', '_IR.avi'), fourcc, fps, (width, height))
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
        # cv2.imshow("1", pred_img)
        # cv2.waitKey()
        out.write(pred_img)
        count += 1
        if ts > 30:
            break
    out.release()
    cap.release()


def testjpg():
    im = cv2.imread("d:/workroom/testroom/old.png")
    img_Out = np.copy(im)
    ret, lr_buf = cv2.imencode(".jpg", img_Out, [int(cv2.IMWRITE_JPEG_QUALITY), np.random.randint(15, 16)])
    img_Out = cv2.imdecode(lr_buf, 1)
    cv2.imshow("1", img_Out)
    cv2.waitKey()

if __name__ == "__main__":
    # testjpg()
    # exit(0)
    print("Hi, this is video IR program!")
    mp4s = ['d:/workroom/testroom/lowquality_video/dm_3_1920x1080.mp4',
            'd:/workroom/testroom/lowquality_video/dy_1_1920x1080.mp4',
            'd:/workroom/testroom/lowquality_video/lq_3_1920x1080.mp4']
    # for mp4 in mp4s:
    #     main(s_mp4=mp4)
    main()
    print('done')

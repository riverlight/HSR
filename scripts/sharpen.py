# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time

def clip(x, min, max):
    y = x if x>min else min
    return y if y<max else max

def sharpen(img):
    r = 25
    threshold = 0
    factor = 0.5
    h, w, c = img.shape
    print(w, h)
    blur_img = cv2.GaussianBlur(img, (r, r), 0)
    for i in range(h):
        for j in range(w):
            for s in range(c):
                value = int(img[i, j, s]) - int(blur_img[i, j, s])
                if abs(value)>threshold:
                    value = img[i, j, s] + int(factor*value)
                    img[i, j, s] = clip(value, 0, 255)
    return img


def test_video():
    scale = 1
    s_mp4 = "d:/workroom/testroom/jie_lr_hsr.mp4"
    if ".mp4" not in s_mp4:
        raise Exception("111")
    cap = cv2.VideoCapture(s_mp4)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*scale)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*scale)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'I420')
    out = cv2.VideoWriter(s_mp4.replace('.mp4', '_sp.avi'), fourcc, fps, (width, height))

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
        pred_img = sharpen(frame)
        out.write(pred_img)
        count += 1
        if ts > 3:
            break
    out.release()
    cap.release()

def test_image():
    imagename = "d:/workroom/testroom/v0.png"
    img = cv2.imread(imagename)
    img = sharpen(img)
    cv2.imwrite(imagename.replace(".png", "-sp.png").replace(".jpg", "-sp.jpg"), img)

if __name__=="__main__":
    # test_image()
    test_video()
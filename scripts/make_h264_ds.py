# -*- coding: utf-8 -*-

import os
import sys
import cv2
import random

def downsample_dir(srcdir, dstdir):
    print("start process..")
    for name in os.listdir(srcdir):
        fullname = os.path.join(srcdir, name)
        print(fullname)
        dstname = os.path.join(dstdir, name)
        img = cv2.imread(fullname)
        h, w, c = img.shape
        img = cv2.resize(img, (w//2, h//2))
        cv2.imwrite(dstname, img)
    print('done')

def image_2_h264image(imagename, qp=40, dst_dir="d:/"):
    tmp_dir = "d:/workroom/testroom/hypnos-v2/"
    tmpmp4 = os.path.join(tmp_dir, "tmp.mp4")
    image2mp4_cmd = "ffmpeg -i {} -vcodec h264 -y -qp {} {} 2>d:/1.txt".format(imagename, qp, tmpmp4)
    os.system(image2mp4_cmd)
    mp42image_cmd = "ffmpeg -i {} -r 1 -f image2 {}image-%d.png 2>d:/1.txt".format(tmpmp4, tmp_dir)
    os.system(mp42image_cmd)
    cp_cmd = "cp {} {}".format(os.path.join(tmp_dir, 'image-1.png'), os.path.join(dst_dir, os.path.basename(imagename)))
    print(cp_cmd)
    os.system(cp_cmd)

def rm_bad_png():
    dir = 'D:\\workroom\\tools\\dataset\\SR\\QIR_val_GT'
    print('start process...')
    for name in os.listdir(dir):
        fullname = os.path.join(dir, name)
        print(fullname)
        img = cv2.imread(fullname)
        h, w, c = img.shape
        if h%2 or w%2:
            rm_cmd = "rm -rf {}".format(fullname)
            print(rm_cmd)
            os.system(rm_cmd)
        # exit(0)
    print('done')

def make_h264image_dir(gt_dir, qp_dir):
    print('start process...')
    for name in os.listdir(gt_dir):
        qp = random.randint(23, 42)
        fullname = os.path.join(gt_dir, name)
        print(fullname, qp)
        image_2_h264image(fullname, qp=qp, dst_dir=qp_dir)
    print('done')

if __name__=="__main__":
    # image_2_h264image('d:/workroom/testroom/windows.png', 40)
    # rm_bad_png()
    make_h264image_dir('D:\\workroom\\tools\\dataset\\SR\\QIR_val_lr', 'D:\\workroom\\tools\\dataset\\SR\\QIR_val_lr_h264')
    # downsample_dir('D:\\workroom\\tools\\dataset\\SR\\QIR_val_GT', 'D:\\workroom\\tools\\dataset\\SR\\QIR_val_lr')
    print("done")

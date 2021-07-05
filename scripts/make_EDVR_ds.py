# -*- coding: utf-8 -*-

import sys
import os
import random
import cv2


def video_2_png(hr_video, lr_video, hr_png_dir, lr_png_dir):
    hr_cap = cv2.VideoCapture(hr_video)
    lr_cap = cv2.VideoCapture(lr_video)
    count = 0
    while True:
        print('count : ', count)
        ret_lr, lr_frame = lr_cap.read()
        ret_hr, hr_frame = hr_cap.read()
        if ret_lr is not True or ret_hr is not True:
            break
        image_name = "{:0>4d}.png".format(count)
        cv2.imwrite(os.path.join(lr_png_dir, image_name), lr_frame)
        cv2.imwrite(os.path.join(hr_png_dir, image_name), hr_frame)
        count += 1
        if count > 100:
            break
    print('done')


def make_train_val_txt(dir, txtname):
    # python3 make_EDVR_ds.py d:/workroom/tools/dataset/douyin/HR edvr.txt
    print(dir, txtname)
    filelst = os.listdir(dir)
    print(filelst)
    with open(txtname, 'wt') as ft:
        for name in filelst:
            line = "{},{}\n".format('val' if random.random() < 0.1 else 'train', name)
            print(line)
            ft.write(line)
    print('done')

def make_edvr_png(sdir, target_dir, input_dir):
    # python3 make_EDVR_ds.py d:/workroom/tools/dataset/douyin/HR
    # d:/workroom/tools/image/EDVR/datasets/train/target d:/workroom/tools/image/EDVR/datasets/train/input
    filelst = os.listdir(sdir)
    for videoname in filelst:
        print(videoname)
        dstname = "dy_{}".format(videoname.replace(".mp4", ""))
        print(dstname)
        HR_name = os.path.join(sdir, videoname)
        LR_name = './tmp.mp4'
        cmd = 'ffmpeg -i {} -vf "scale=iw/2:ih/2" -an -y {}'.format(HR_name, LR_name)
        os.system(cmd)
        hr_png_dir = os.path.join(target_dir, dstname)
        lr_png_dir = os.path.join(input_dir, dstname)
        if sys.platform=='win32':
            hr_png_dir = hr_png_dir.replace('/', '\\')
            lr_png_dir = lr_png_dir.replace('/', '\\')
        print(hr_png_dir, lr_png_dir)
        os.system('mkdir {}'.format(hr_png_dir))
        os.system('mkdir {}'.format(lr_png_dir))
        video_2_png(HR_name, LR_name, hr_png_dir, lr_png_dir)
        os.system('rm -rf {}'.format(LR_name))
        # exit(0)
    pass

if __name__=="__main__":
    # make_train_val_txt(sys.argv[1], sys.argv[2])
    make_edvr_png(sys.argv[1], sys.argv[2], sys.argv[3])

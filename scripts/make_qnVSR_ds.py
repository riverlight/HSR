# -*- coding: utf-8 -*-

import os
import cv2
import random
import h5py
import numpy as np


def HRV_2_LRV():
    LR_dir = "D:\\workroom\\tools\\dataset\\douyin\\Download\\LR"
    HR_dir = "D:\\workroom\\tools\\dataset\\douyin\\Download\\HR"
    for name in os.listdir(HR_dir):
        LR_name = os.path.join(LR_dir, name)
        HR_name = os.path.join(HR_dir, name)
        print(name, LR_name)
        cmd = 'ffmpeg -i {} -vf "scale=iw/2:ih/2" -y {}'.format(HR_name, LR_name)
        print(cmd)
        os.system(cmd)
        # exit(0)
    print('done')


def video_2_image():
    LR_dir = "D:\\workroom\\tools\\dataset\\douyin\\Download\\LR"
    HR_dir = "D:\\workroom\\tools\\dataset\\douyin\\Download\\HR"
    LR_image_dir = "D:\\workroom\\tools\\dataset\\douyin\\LR_image"
    HR_image_dir = "D:\\workroom\\tools\\dataset\\douyin\\HR_image"
    for id, name in enumerate(os.listdir(HR_dir)):
        LR_name = os.path.join(LR_dir, name)
        HR_name = os.path.join(HR_dir, name)
        print(HR_name, LR_name)
        baseid = name.replace(".mp4", "")
        hr_cap = cv2.VideoCapture(HR_name)
        lr_cap = cv2.VideoCapture(LR_name)
        count = 0
        while True:
            ret_lr, lr_frame = lr_cap.read()
            ret_hr, hr_frame = hr_cap.read()
            if ret_lr is not True or ret_hr is not True:
                break
            count += 1
            if count % 150 != 1:
                continue
            image_name = "v{}_{}.png".format(baseid, count)
            cv2.imwrite(os.path.join(LR_image_dir, image_name), lr_frame)
            cv2.imwrite(os.path.join(HR_image_dir, image_name), hr_frame)
    print('done')

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
        if count > 10000:
            break
    print('done')

def make_train_and_val():
    LR_image_dir = "D:\\workroom\\tools\\dataset\\douyin\\LR_image"
    HR_image_dir = "D:\\workroom\\tools\\dataset\\douyin\\HR_image"
    train_hr_dir = "D:\\workroom\\tools\\dataset\\douyin\\train_hr"
    train_lr_dir = "D:\\workroom\\tools\\dataset\\douyin\\train_lr"
    val_hr_dir = "D:\\workroom\\tools\\dataset\\douyin\\val_hr"
    val_lr_dir = "D:\\workroom\\tools\\dataset\\douyin\\val_lr"
    count = 0
    for id, name in enumerate(os.listdir(LR_image_dir)):
        print(id, name)
        src_lr = os.path.join(LR_image_dir, name)
        src_hr = os.path.join(HR_image_dir, name)
        flag = random.random() < 0.1
        print(flag)
        if flag:
            # dst_lr = os.path.join(val_lr_dir, name)
            # dst_hr = os.path.join(val_hr_dir, name)
            dst_lr = val_lr_dir
            dst_hr = val_hr_dir
            count += 1
        else:
            # dst_lr = os.path.join(train_lr_dir, name)
            # dst_hr = os.path.join(train_hr_dir, name)
            dst_lr = train_lr_dir
            dst_hr = train_hr_dir
        cmd = "copy {} {}".format(src_lr, dst_lr)
        os.system(cmd)
        cmd = "copy {} {}".format(src_hr, dst_hr)
        os.system(cmd)
        # exit()
    print(count)


def dir_2_h5():
    h5_file = h5py.File("../qn_dataset/vsr_dy_val_hwcbgr.h5", 'w')
    patch_size = 96
    stride = 96
    hr_patchs = list()
    lr_patchs = list()

    train_hr_dir = "D:\\workroom\\tools\\dataset\\douyin\\train_hr"
    train_lr_dir = "D:\\workroom\\tools\\dataset\\douyin\\train_lr"
    val_hr_dir = "D:\\workroom\\tools\\dataset\\douyin\\val_hr"
    val_lr_dir = "D:\\workroom\\tools\\dataset\\douyin\\val_lr"
    hr_dir = val_hr_dir
    lr_dir = val_lr_dir
    for id, name in enumerate(os.listdir(hr_dir)):
        print(id, name)
        hr_name = os.path.join(hr_dir, name)
        lr_name = os.path.join(lr_dir, name)
        # BGR HWC
        hr_img = cv2.imread(hr_name, cv2.IMREAD_UNCHANGED)
        lr_img = cv2.imread(lr_name, cv2.IMREAD_UNCHANGED)
        for i in range(0, hr_img.shape[0] - patch_size + 1, stride):
            for j in range(0, hr_img.shape[1] - patch_size + 1, stride):
                hr_np = hr_img[i:i + patch_size, j:j + patch_size, :]
                hr_patchs.append(hr_np)
                lr_np = lr_img[i // 2:(i + patch_size) // 2, j // 2:(j + patch_size) // 2, :]
                lr_patchs.append(lr_np)
        # print(hr_img.shape)

    print("scan ok")
    hr_ds = np.array(hr_patchs, dtype=np.uint8)
    lr_ds = np.array(lr_patchs, dtype=np.uint8)
    h5_file.create_dataset('hr', data=hr_ds)
    h5_file.create_dataset('lr', data=lr_ds)
    h5_file.close()
    print('done')


if __name__=="__main__":
    # HRV_2_LRV()
    # video_2_image()
    # make_train_and_val()
    # dir_2_h5()
    video_2_png('d:/workroom/testroom/jie.mp4', 'd:/workroom/testroom/jie_lr_4.mp4',
                'D:/workroom/tools/image/EDVR/datasets/test/target/jie', 'D:/workroom/tools/image/EDVR/datasets/test/input/jie')

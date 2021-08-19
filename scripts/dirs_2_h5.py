# -*- coding: utf-8 -*-

import sys
import h5py
import os
import cv2
import numpy as np


patch_size = 96
stride = patch_size * 2

def main(dir_gt, dir_ni, h5_name, mode):
    print("dir_gt : ", dir_gt)
    print("dir_ni : ", dir_ni)
    print("h5_name : ", h5_name)
    print('mode : ', mode.upper())
    h5_file = h5py.File(h5_name, 'w')
    hr_patchs = list()
    ni_patchs = list()

    for count, name in enumerate(os.listdir(dir_gt)):
        # if count>2:
        #     break
        gtname = os.path.join(dir_gt, name)
        niname = os.path.join(dir_ni, name)
        if os.path.isdir(gtname):
            continue
        print("id : ", count, gtname, niname)
        # BGR HWC
        hr_img = cv2.imread(gtname, cv2.IMREAD_UNCHANGED)
        ni_img = cv2.imread(niname, cv2.IMREAD_UNCHANGED)
        print(ni_img.shape, hr_img.shape)
        if mode=='CHWRGB':
            # BGR HWC to RGB CHW
            hr_img = hr_img[:, :, [2, 1, 0]].transpose(2, 0, 1)
            ni_img = ni_img[:, :, [2, 1, 0]].transpose(2, 0, 1)
            for i in range(0, hr_img.shape[1] - patch_size + 1, stride):
                for j in range(0, hr_img.shape[2] - patch_size + 1, stride):
                    hr_np = hr_img[:, i:i + patch_size, j:j + patch_size]
                    hr_patchs.append(hr_np)
                    ni_np = ni_img[:, i:i + patch_size, j:j + patch_size]
                    ni_patchs.append(ni_np)
        else: # HWCBGR
            for i in range(0, hr_img.shape[0] - patch_size + 1, stride):
                for j in range(0, hr_img.shape[1] - patch_size + 1, stride):
                    hr_np = hr_img[i:i + patch_size, j:j + patch_size, :]
                    hr_patchs.append(hr_np)
                    ni_np = ni_img[i:i + patch_size, j:j + patch_size, :]
                    ni_patchs.append(ni_np)

    print("loop completed...")
    hr_ds = np.array(hr_patchs, dtype=np.uint8)
    ni_ds = np.array(ni_patchs, dtype=np.uint8)
    h5_file.create_dataset('hr', data=hr_ds)
    h5_file.create_dataset('ni', data=ni_ds)
    h5_file.close()
    print('done')
    pass

if __name__=="__main__":
    if len(sys.argv)!=5:
        print("python3 dirs_2_h5.py dir_gt dir_ni h5_name mode")
        print('mode must be HWCBGR or CHWRGB')
        print("sample : python3 dirs_2_h5.py D:/workroom/tools/dataset/SR/QIR_GT D:/workroom/tools/dataset/SR/QIR_QP_45 "
              "../qn_dataset/h264_train_hwcbgr.h5 HWCBGR")
        exit(-1)
    if sys.argv[4].lower() not in ['hwcbgr', 'chwrgb']:
        print('mode must be HWCBGR or CHWRGB')
        exit(-2)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

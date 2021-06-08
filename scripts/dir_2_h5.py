# -*- coding: utf-8 -*-

import sys
import h5py
import os
import cv2
import numpy as np


patch_size = 96
stride = patch_size * 1

def main(dir, h5_name, mode):
    print("dir : ", dir)
    print("h5_name : ", h5_name)
    print('mode : ', mode.upper())
    h5_file = h5py.File(h5_name, 'w')
    hr_patchs = list()

    for count, name in enumerate(os.listdir(dir)):
        imagename = os.path.join(dir, name)
        if os.path.isdir(imagename):
            continue
        print("id : ", count, imagename)
        # BGR HWC
        hr_img = cv2.imread(imagename, cv2.IMREAD_UNCHANGED)
        if mode=='CHWRGB':
            # BGR HWC to RGB CHW
            hr_img = hr_img[:, :, [2, 1, 0]].transpose(2, 0, 1)
            for i in range(0, hr_img.shape[1] - patch_size + 1, stride):
                for j in range(0, hr_img.shape[2] - patch_size + 1, stride):
                    hr_np = hr_img[:, i:i + patch_size, j:j + patch_size]
                    hr_patchs.append(hr_np)
        else: # HWCBGR
            for i in range(0, hr_img.shape[0] - patch_size + 1, stride):
                for j in range(0, hr_img.shape[1] - patch_size + 1, stride):
                    hr_np = hr_img[i:i + patch_size, j:j + patch_size, :]
                    hr_patchs.append(hr_np)

    hr_ds = np.array(hr_patchs, dtype=np.uint8)
    h5_file.create_dataset('hr', data=hr_ds)
    h5_file.close()
    print('done')
    pass

if __name__=="__main__":
    if len(sys.argv)!=4:
        print("python3 dir_2_h5.py dir_name h5_name mode")
        print('mode must be HWCBGR or CHWRGB')
        print("sample : python3 dir_2_h5.py D:/workroom/tools/image/ntire20/Corrupted-tr-y ../qn_dataset/vsr_train_hwcbgr.h5 HWCBGR")
        exit(-1)
    if sys.argv[3].lower() not in ['hwcbgr', 'chwrgb']:
        print('mode must be HWCBGR or CHWRGB')
        exit(-2)
    main(sys.argv[1], sys.argv[2], sys.argv[3])

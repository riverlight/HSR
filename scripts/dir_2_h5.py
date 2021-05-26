# -*- coding: utf-8 -*-

import sys
import h5py
import os
import cv2
import numpy as np


patch_size = 96
stride = patch_size * 1

def main(dir, h5_name):
    print("dir : ", dir)
    print("h5_name : ", h5_name)
    h5_file = h5py.File(h5_name, 'w')
    hr_patchs = list()

    for count, name in enumerate(os.listdir(dir)):
        imagename = os.path.join(dir, name)
        if os.path.isdir(imagename):
            continue
        print(imagename)
        # BGR HWC
        hr_img = cv2.imread(imagename, cv2.IMREAD_UNCHANGED)
        # BGR HWC to RGB CHW
        hr_img = hr_img[:, :, [2, 1, 0]].transpose(2, 0, 1)

        for i in range(0, hr_img.shape[1] - patch_size + 1, stride):
            for j in range(0, hr_img.shape[2] - patch_size + 1, stride):
                hr_np = hr_img[:, i:i + patch_size, j:j + patch_size]
                hr_patchs.append(hr_np)

    hr_ds = np.array(hr_patchs, dtype=np.uint8)
    h5_file.create_dataset('hr', data=hr_ds)
    h5_file.close()
    print('done')
    pass

if __name__=="__main__":
    if len(sys.argv)!=3:
        print("python3 dir_2_h5.py dir_name h5_name")
        exit(-1)
    main(sys.argv[1], sys.argv[2])

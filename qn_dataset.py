# -*- coding: utf-8 -*-

import h5py
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import cv2
import h_psnr


class QNDataset(Dataset):
    def __init__(self, h5file):
        super(QNDataset, self).__init__()
        self.h5_file = h5file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return f['hr'][idx].astype(np.float32)/255, f['lr'][idx].astype(np.float32)/255

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['hr'])


def test():
    ds = QNDataset(".\\qn_dataset\\train.h5")
    dl = DataLoader(dataset=ds, batch_size=1)
    for data in dl:
        hr_img, lr_img = data
        hr_img = hr_img.numpy()*255
        lr_img = lr_img.numpy()*255
        hr_img = hr_img.transpose(0, 2, 3, 1)
        lr_img = lr_img.transpose(0, 2, 3, 1)
        cv2.imshow("hr", hr_img[0, ...].astype(np.uint8))
        cv2.imshow("lr", lr_img[0, ...].astype(np.uint8))
        cv2.waitKey(0)
        # break

def calc_ds_psnr():
    ds = QNDataset(".\\qn_dataset\\eval.h5")
    dl = DataLoader(dataset=ds, batch_size=1)
    lst_bic_psnr = list()
    for data in dl:
        hr_img, lr_img = data
        hr_img = hr_img.numpy()*255
        lr_img = lr_img.numpy()*255
        hr_img = hr_img.transpose(0, 2, 3, 1)
        lr_img = lr_img.transpose(0, 2, 3, 1)
        lst_bic_psnr.append(h_psnr.calc_psnr_np_upsample(hr_img[0, ...], lr_img[0, ...]).item())
    print("mean psnr : ", sum(lst_bic_psnr)/len(lst_bic_psnr))


if __name__ == "__main__":
    # test()
    calc_ds_psnr()

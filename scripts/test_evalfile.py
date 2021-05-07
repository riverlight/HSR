# -*- coding: utf-8 -*-

import sys
sys.path.append("../")
from qn_dataset import QNDataset
from torch.utils.data.dataloader import DataLoader
from torch.nn.functional import interpolate
from utils import AverageMeter, calc_psnr


def main():
    device = 'cpu'
    eval_file = "../qn_dataset/eval.h5"
    eval_dataset = QNDataset(eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    epoch_psnr = AverageMeter()
    for count, data in enumerate(eval_dataloader):
        hr_img, lr_img = data
        hr_img = hr_img.to(device)
        lr_img = lr_img.to(device)
        hsi_img = interpolate(lr_img, scale_factor=2, mode="bicubic", align_corners=False)
        print(len(lr_img))
        epoch_psnr.update(calc_psnr(hsi_img, hr_img), len(lr_img))
        if count % 10 == 0:
            print(', eval psnr: {:.2f}'.format(epoch_psnr.avg))

    print(', eval psnr: {:.2f}'.format(epoch_psnr.avg))
    pass

if __name__=="__main__":
    main()

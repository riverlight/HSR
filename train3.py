# -*- coding: utf-8 -*-

from qn_dataset import QNDataset
from qn_dataset2 import qnSRDataset, qnSRDataset2
from torch.utils.data.dataloader import DataLoader
from models import HRcanNet
import os
import torch.optim as optim
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torch as t
import torch.nn as nn
from utils import AverageMeter, calc_psnr
from HModels.discriminator_vgg_arch import VGGFeatureExtractor
import sys


def trainDSConfig():
    cfgDict = {
        # "dataroot_GT": "D:\\workroom\\tools\\image\\Real-SR\\datasets\\DF2K\\generated\\tdsr\\HR",
        'dataroot_GT': '/home/workroom/project/riverlight/datasets/DF2K/generated/tdsr/HR/',
        "dataroot_LQ": None,
        "noise": True,
        # "noise_data": "D:\\workroom\\tools\\image\\Real-SR\\datasets\DF2K\Corrupted_noise\\",
        'noise_data': '/home/workroom/project/riverlight/datasets/DF2K/Corrupted_noise/',
        "patch_size": 96,
        'scale': 2
    }
    return cfgDict

def evalDSConfig():
    cfgDict = {
        # "dataroot_GT": "D:\\workroom\\tools\\image\\ntire20\\track1-valid-gt-d2\\",
        # "dataroot_LQ": "D:\\workroom\\tools\\image\\ntire20\\track1-valid-input\\",
        "dataroot_GT": "/home/workroom/project/riverlight/datasets/track1-valid-gt-d2/",
        "dataroot_LQ": "/home/workroom/project/riverlight/datasets/track1-valid-input/",
        "noise": False,
        "noise_data": None,
        "patch_size": 96,
        'scale': 2
    }
    return cfgDict

def main():
    if t.cuda.device_count() > 1:
        use_gpus = True
    else:
        use_gpus = False
    # use_gpus = False
    eval_file = "./qn_dataset/eval.h5"
    outputs_dir = "./weights/"
    lr = 1e-4
    if use_gpus:
        batch_size = 24*24
        num_workers = 8
    else:
        batch_size = 8
        num_workers = 1
    num_epochs = 400

    seed = 1108
    best_weights = None
    best_weights = "./weights/hsi3_epoch_213.pth"
    start_epoch = 214

    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    cudnn.benchmark = True
    device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
    t.manual_seed(seed)
    cri_fea = nn.L1Loss().to(device)
    netPerc = VGGFeatureExtractor(feature_layer=34, use_bn=False, use_input_norm=True, device=device).to(device)
    netPerc.eval()
    if best_weights is not None:
        model = t.load(best_weights)
    else:
        model = HRcanNet().to(device)
    if use_gpus:
        print("Let's use", t.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        model.to(device)
        netPerc = nn.DataParallel(netPerc)

    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss().to(device)

    traindsCfg = trainDSConfig()
    train_dataset = qnSRDataset(traindsCfg)
    if sys.platform=='win32':
        train_dataset = qnSRDataset2('./qn_dataset/hr.h5', noise_dir="D:\\workroom\\tools\\image\\Real-SR\\datasets\DF2K\Corrupted_noise\\")
    else:
        train_dataset = qnSRDataset2('./qn_dataset/hr.h5', noise_dir='/home/workroom/project/riverlight/datasets/DF2K/Corrupted_noise/')
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=False,
                                  drop_last=True)
    evaldsCfg = evalDSConfig()
    eval_dataset = qnSRDataset(evaldsCfg)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_epoch = 0
    best_psnr = 0.0
    for epoch in range(num_epochs - start_epoch):
        epoch += start_epoch
        if use_gpus:
            model.module.train()
        else:
            model.train()
        epoch_losses = AverageMeter()
        pix_losses = AverageMeter()
        fea_losses = AverageMeter()
        with tqdm(total=(len(train_dataset) - len(train_dataset) % batch_size)) as tq:
            tq.set_description('epoch: {}/{}'.format(epoch, num_epochs - 1))

            for i, data in enumerate(train_dataloader):
                loss = 0
                hr_img, lr_img = data['GT'], data['LQ']
                # hr_img = hr_img.to(device)
                # lr_img = lr_img.to(device)
                hr_img = hr_img.cuda()
                lr_img = lr_img.cuda()
                hsi_img = model(lr_img)
                loss_pix = criterion(hsi_img, hr_img)
                pix_losses.update(loss_pix.item(), len(hr_img))
                loss += 1.0 * loss_pix
                real_fea = netPerc(hr_img).detach()
                fake_fea = netPerc(hsi_img)
                loss_fea = cri_fea(real_fea, fake_fea)
                fea_losses.update(loss_fea.item(), len(hr_img))
                loss += 0.2 * loss_fea

                epoch_losses.update(loss.item(), len(hr_img))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tq.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                tq.update(len(hr_img))
                print(i, epoch_losses.avg, pix_losses.ave, fea_losses.ave)

        if use_gpus:
            t.save(model.module, os.path.join(outputs_dir, 'hsi3_epoch_{}.pth'.format(epoch)))
            model.module.eval()
        else:
            t.save(model, os.path.join(outputs_dir, 'hsi3_epoch_{}.pth'.format(epoch)))
            model.eval()

        epoch_psnr = AverageMeter()
        for data in eval_dataloader:
            # hr_img, lr_img = data
            hr_img, lr_img = data['GT'], data['LQ']
            hr_img = hr_img.to(device)
            lr_img = lr_img.to(device)

            with t.no_grad():
                hsi_img = model(lr_img).clamp(0.0, 1.0)

            epoch_psnr.update(calc_psnr(hsi_img, hr_img), len(lr_img))

        print(epoch, ', eval psnr: {:.2f}'.format(epoch_psnr.avg))
        del hr_img, lr_img, hsi_img
        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            if use_gpus:
                t.save(model.module, os.path.join(outputs_dir, 'hsi3_best.pth'))
            else:
                t.save(model, os.path.join(outputs_dir, 'hsi3_best.pth'))

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))



if __name__=="__main__":
    print("Hi, this is a HSISR train3 program")
    main()

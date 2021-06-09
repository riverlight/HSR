# -*- coding: utf-8 -*-

from qn_dataset import QNDataset
from qn_dataset2 import qnSRDataset, qnSRDataset2, qnSRDataset3
from torch.utils.data.dataloader import DataLoader
from models import HRcanNet
import os
import torch.optim as optim
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torch as t
import torch.nn as nn
from utils import AverageMeter, calc_psnr
from HModels.discriminator_vgg_arch import VGGFeatureExtractor, NLayerDiscriminator
from HModels.loss import GANLoss
import sys


def main():
    if sys.platform=="win32":
        use_gpus = False
    else:
        use_gpus = True
    outputs_dir = "./weights/"

    if use_gpus:
        lr = 3e-4
        batch_size = 24*1
        num_workers = 8
        train_interval = 3
        val_interval = 7
    else:
        lr = 1e-4
        batch_size = 8
        num_workers = 1
        train_interval = 31
        val_interval = 15
    num_epochs = 400

    seed = 1108
    best_weights = None
    best_d = None
    best_weights = "./weights/hsi4_epoch_4.pth"
    best_d = "./weights/hsi4_d_4.pth"
    start_epoch = 5

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

    # 判别器相关
    if best_d is not None:
        netD = t.load(best_d)
    else:
        netD = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3).to(device)
    if use_gpus:
        netD = nn.DataParallel(netD)
    netD.train()
    cri_gan = GANLoss('ragan', 1.0, 0.0).to(device)
    l_gan_w = 0.005
    lr_D = 1e-4
    optimizer_D = optim.Adam(params=netD.parameters(), lr=lr_D)

    train_dataset = qnSRDataset3('./qn_dataset/vsr-train.h5', interval=train_interval)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=False,
                                  drop_last=True)
    eval_dataset = qnSRDataset3('./qn_dataset/vsr-val.h5', interval=val_interval)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=batch_size, num_workers=num_workers)

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
        d_losses = AverageMeter()
        with tqdm(total=(len(train_dataset) - len(train_dataset) % batch_size)) as tq:
            tq.set_description('epoch: {}/{}'.format(epoch, num_epochs - 1))

            for i, data in enumerate(train_dataloader):
                # G
                for p in netD.parameters():
                    p.requires_grad = False
                lossG = 0
                hr_img, lr_img = data['GT'], data['LQ']
                # hr_img = hr_img.to(device)
                # lr_img = lr_img.to(device)
                hr_img = hr_img.cuda()
                lr_img = lr_img.cuda()
                hr_fake = model(lr_img)
                loss_pix = criterion(hr_fake, hr_img)
                pix_losses.update(loss_pix.item(), len(hr_img))
                lossG += 0.01 * loss_pix
                real_fea = netPerc(hr_img).detach()
                fake_fea = netPerc(hr_fake)
                loss_fea = cri_fea(real_fea, fake_fea)
                fea_losses.update(loss_fea.item(), len(hr_img))
                lossG += 1.0 * loss_fea

                pred_g_fake = netD(hr_fake)
                pred_d_real = netD(hr_img).detach()
                l_g_gan = l_gan_w * (cri_gan(pred_d_real - t.mean(pred_g_fake), False) + cri_gan(pred_g_fake - t.mean(pred_d_real), True)) / 2
                lossG += l_g_gan

                epoch_losses.update(lossG.item(), len(hr_img))
                optimizer.zero_grad()
                lossG.backward()
                optimizer.step()

                # D
                for p in netD.parameters():
                    p.requires_grad = True
                optimizer_D.zero_grad()
                lossD = 0
                pred_d_real = netD(hr_img)
                pred_d_fake = netD(hr_fake.detach())  # detach to avoid BP to G
                l_d_real = cri_gan(pred_d_real - t.mean(pred_d_fake), True)
                l_d_fake = cri_gan(pred_d_fake - t.mean(pred_d_real), False)
                lossD = (l_d_real + l_d_fake) / 2
                d_losses.update(lossD.item(), len(hr_img))
                lossD.backward()
                optimizer_D.step()

                tq.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                tq.update(len(hr_img))
                print(i, epoch_losses.avg, pix_losses.avg, fea_losses.avg, d_losses.avg)

        if use_gpus:
            t.save(model.module, os.path.join(outputs_dir, 'hsi4_epoch_{}.pth'.format(epoch)))
            t.save(netD.module, os.path.join(outputs_dir, 'hsi4_d_{}.pth'.format(epoch)))
            model.module.eval()
        else:
            t.save(model, os.path.join(outputs_dir, 'hsi4_epoch_{}.pth'.format(epoch)))
            t.save(netD, os.path.join(outputs_dir, 'hsi4_d_{}.pth'.format(epoch)))
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
                t.save(model.module, os.path.join(outputs_dir, 'hsi4_best.pth'))
            else:
                t.save(model, os.path.join(outputs_dir, 'hsi4_best.pth'))

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))



if __name__=="__main__":
    print("Hi, this is a HSISR train3 program")
    main()

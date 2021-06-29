# -*- coding: utf-8 -*-

from qn_dataset3 import qnDataset2, qnVSRDataset
from torch.utils.data.dataloader import DataLoader
from models import HRcanNet, HRRDBNet, HSISRNet
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
from torch.nn.functional import interpolate
from math import sqrt


if sys.platform != 'win32':
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def prn_obj(obj):
    print('\n'.join(['%s:%s' % item for item in obj.__dict__.items()]))

class CTrain():
    def __init__(self):
        self.name = "vsr_HRRDBNet"
        self.init()
        if sys.platform == "win32":
            self.use_gpus = False
        else:
            self.use_gpus = True

        if self.use_gpus:
            self.lr = 2e-4
            self.min_lr = 5e-5
            self.batch_size = 8*8
            self.num_workers = 8
            self.train_interval = 0
            self.val_interval = 0
        else:
            self.lr = 4e-4
            self.min_lr = 1e-4
            self.batch_size = 8
            self.num_workers = 2
            self.train_interval = 63
            self.val_interval = 63
        self.lr_gamma = 0.5
        self.num_epochs = 400
        self.best_weights = None
        self.best_d = None
        # self.best_weights = "./weights/{}_epoch_208.pth".format(self.name)
        # self.best_d = "./weights/{}_d_208.pth".format(self.name)
        self.start_epoch = 0
        self.device = t.device('cuda' if t.cuda.is_available() else 'cpu')

        self.cri_fea = nn.L1Loss().to(self.device)
        self.netPerc = VGGFeatureExtractor(feature_layer=34, use_bn=False, use_input_norm=True, device=self.device).to(self.device)
        self.netPerc.eval()
        if self.best_weights is not None:
            self.model = t.load(self.best_weights)
        else:
            self.model = HSISRNet().to(self.device)
        if self.use_gpus:
            print("Let's use", t.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
            self.model.to(self.device)
            self.netPerc = nn.DataParallel(self.netPerc)
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr)
        # criterion = nn.MSELoss()
        self.cri_pix = nn.L1Loss().to(self.device)
        self.l_pix_w = 1
        self.l_fea_w = 0.5
        self.l_d_w = 0

        self.init_D()
        # self.init_dataset()
        self.init_dataset_vsr()

    def init(self):
        self.outputs_dir = "./weights/"
        if not os.path.exists(self.outputs_dir):
            os.makedirs(self.outputs_dir)

        cudnn.benchmark = True
        seed = 1108
        t.manual_seed(seed)


    def init_D(self):
        # 判别器相关
        if self.best_d is not None:
            self.netD = t.load(self.best_d)
        else:
            self.netD = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3).to(self.device)
        if self.use_gpus:
            self.netD = nn.DataParallel(self.netD)
        self.netD.train()
        self.cri_d = GANLoss('ragan', 1.0, 0.0).to(self.device)
        self.lr_D = self.lr
        self.optimizer_D = optim.Adam(params=self.netD.parameters(), lr=self.lr_D)

    def init_dataset(self):
        self.dsConf = {
            'scale': 2,
            'noise': False,
            'jpeg': False,
            'camera': False,
            'blur': False
        }
        self.train_dataset = qnDataset2('./qn_dataset/vsr_train_hwcbgr.h5', interval=self.train_interval)
        self.train_dataset.config(**self.dsConf)
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=self.num_workers,
                                      pin_memory=False,
                                      drop_last=True)
        self.eval_dataset = qnDataset2('./qn_dataset/vsr_val_hwcbgr.h5', interval=self.val_interval)
        self.eval_dataset.config(**self.dsConf)
        self.eval_dataloader = DataLoader(dataset=self.eval_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        self.trainds_len = len(self.train_dataset)
        print(len(self.train_dataset), len(self.eval_dataset))
        # exit(0)

    def init_dataset_vsr(self):
        self.train_dataset = qnVSRDataset('./qn_dataset/vsr_dy_train_hwcbgr.h5', interval=self.train_interval)
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           num_workers=self.num_workers,
                                           pin_memory=False,
                                           drop_last=True)
        self.eval_dataset = qnVSRDataset('./qn_dataset/vsr_dy_val_hwcbgr.h5', interval=self.val_interval)
        self.eval_dataloader = DataLoader(dataset=self.eval_dataset, batch_size=self.batch_size,
                                          num_workers=self.num_workers)
        self.trainds_len = len(self.train_dataset)
        print(len(self.train_dataset), len(self.eval_dataset))


    def adjust_lr(self, epoch):
        lr = self.lr * (self.lr_gamma ** (epoch // 60))
        lr = lr if lr > self.min_lr else self.min_lr
        print("adjust lr : epoch[{}] lr : {}".format(epoch, lr))
        for param_group in self.optimizer.param_groups:
            param_group['lr']= lr
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr

    def train(self):
        best_epoch = 0
        best_psnr = 0.0
        for epoch in range(self.num_epochs - self.start_epoch):
            epoch += self.start_epoch
            # 动态调整学习率
            self.adjust_lr(epoch)

            if self.use_gpus:
                self.model.module.train()
            else:
                self.model.train()
            epoch_losses = AverageMeter()
            pix_losses = AverageMeter()
            fea_losses = AverageMeter()
            d_losses = AverageMeter()
            with tqdm(total=(self.trainds_len - self.trainds_len % self.batch_size)) as tq:
                tq.set_description('epoch: {}/{}'.format(epoch, self.num_epochs - 1))

                for i, data in enumerate(self.train_dataloader):
                    # G
                    for p in self.netD.parameters():
                        p.requires_grad = False
                    lossG = 0
                    real_img, ni_img = data['GT'], data['LQ']
                    # real_img = real_img.to(device)
                    # ni_img = ni_img.to(device)
                    real_img = real_img.cuda()
                    ni_img = ni_img.cuda()
                    hr_fake = self.model(ni_img).clamp(0.0, 1.0)
                    loss_pix = self.cri_pix(hr_fake, real_img)
                    pix_losses.update(loss_pix.item(), len(real_img))
                    lossG += self.l_pix_w * loss_pix
                    real_fea = self.netPerc(real_img).detach()
                    fake_fea = self.netPerc(hr_fake)
                    loss_fea = self.cri_fea(real_fea, fake_fea)
                    fea_losses.update(loss_fea.item(), len(real_img))
                    lossG += self.l_fea_w * loss_fea

                    pred_g_fake = self.netD(hr_fake)
                    pred_d_real = self.netD(real_img).detach()
                    l_g_gan = self.l_d_w * (self.cri_d(pred_d_real - t.mean(pred_g_fake), False) + self.cri_d(
                        pred_g_fake - t.mean(pred_d_real), True)) / 2
                    lossG += l_g_gan

                    epoch_losses.update(lossG.item(), len(real_img))
                    self.optimizer.zero_grad()
                    lossG.backward()
                    self.optimizer.step()

                    # D
                    for p in self.netD.parameters():
                        p.requires_grad = True
                    self.optimizer_D.zero_grad()
                    lossD = 0
                    pred_d_real = self.netD(real_img)
                    pred_d_fake = self.netD(hr_fake.detach())  # detach to avoid BP to G
                    l_d_real = self.cri_d(pred_d_real - t.mean(pred_d_fake), True)
                    l_d_fake = self.cri_d(pred_d_fake - t.mean(pred_d_real), False)
                    lossD = (l_d_real + l_d_fake) / 2
                    d_losses.update(lossD.item(), len(real_img))
                    lossD.backward()
                    self.optimizer_D.step()

                    tq.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                    tq.update(len(real_img))
                    print('epoch:', epoch, i, epoch_losses.avg, pix_losses.avg, fea_losses.avg, d_losses.avg)

            if self.use_gpus:
                t.save(self.model.module, os.path.join(self.outputs_dir, '{}_epoch_{}.pth'.format(self.name, epoch)))
                t.save(self.netD.module, os.path.join(self.outputs_dir, '{}_d_{}.pth'.format(self.name, epoch)))
                self.model.module.eval()
            else:
                t.save(self.model, os.path.join(self.outputs_dir, '{}_epoch_{}.pth'.format(self.name, epoch)))
                t.save(self.netD, os.path.join(self.outputs_dir, '{}_d_{}.pth'.format(self.name, epoch)))
                self.model.eval()

            epoch_psnr = AverageMeter()
            epoch_bic_psnr = AverageMeter()
            for data in self.eval_dataloader:
                # real_img, ni_img = data
                real_img, ni_img = data['GT'], data['LQ']
                real_img = real_img.cuda()
                ni_img = ni_img.cuda()
                bic_img = interpolate(ni_img, scale_factor=2, mode="bicubic", align_corners=False)
                with t.no_grad():
                    hsi_img = self.model(ni_img).clamp(0.0, 1.0)
                epoch_psnr.update(calc_psnr(hsi_img, real_img), len(ni_img))
                epoch_bic_psnr.update(calc_psnr(bic_img, real_img), len(ni_img))

            print(epoch, ', eval psnr: {:.2f}, bic psnr : {:.2f}'.format(epoch_psnr.avg, epoch_bic_psnr.avg))
            del real_img, ni_img, hsi_img
            if epoch_psnr.avg > best_psnr:
                best_epoch = epoch
                best_psnr = epoch_psnr.avg
                if self.use_gpus:
                    t.save(self.model.module, os.path.join(self.outputs_dir, '{}_best.pth'.format(self.name)))
                else:
                    t.save(self.model, os.path.join(self.outputs_dir, '{}_best.pth'.format(self.name)))

        print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))

def main():
    app = CTrain()
    prn_obj(app)
    app.train()

if __name__=="__main__":
    print("Hi, this is a QIR train program")
    main()

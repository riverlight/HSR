# -*- coding: utf-8 -*-

from qn_dataset3 import qnDataset2, qnVSRDataset
from torch.utils.data.dataloader import DataLoader
from models import HRcanNet, HRRDBNet, HSISRNet, HRcanNet2
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
import random


if sys.platform != 'win32':
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def prn_obj(obj):
    print('\n'.join(['%s:%s' % item for item in obj.__dict__.items()]))

class CTrain():
    def __init__(self):
        model_name = 'HRcanNet2_1_33'
        ir_type = "deblur"
        self.scale = 1
        self.name = "{}_{}".format(ir_type, model_name)
        self.init()
        if sys.platform == "win32":
            self.use_gpus = False
        else:
            self.use_gpus = True
        self.use_gpus = False

        if self.use_gpus:
            self.lr = 4e-4
            self.min_lr = 5e-5
            self.batch_size = 8
            self.num_workers = 2
            self.train_interval = 15
            self.val_interval = 15
        else:
            self.lr = 4e-4
            self.min_lr = 5e-5
            self.batch_size = 8
            self.num_workers = 2
            self.train_interval = 15
            self.val_interval = 15
        self.lr_gamma = 0.5
        self.num_epochs = 400
        self.best_weights = None
        # self.best_weights = "./weights/{}_epoch_60.pth".format(self.name)
        self.start_epoch = 0
        self.device = t.device('cuda' if t.cuda.is_available() else 'cpu')

        self.cri_fea = nn.L1Loss().to(self.device)
        self.netPerc = VGGFeatureExtractor(feature_layer=34, use_bn=False, use_input_norm=True, device=self.device).to(self.device)
        self.netPerc.eval()
        if self.best_weights is not None:
            self.model = t.load(self.best_weights)
        else:
            self.model = HRcanNet2(scale=self.scale, resblocks=3, resgroup=3).to(self.device)
        if self.use_gpus:
            print("Let's use", t.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
            self.model.to(self.device)
            self.netPerc = nn.DataParallel(self.netPerc)
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr)
        # criterion = nn.MSELoss()
        self.cri_pix = nn.L1Loss().to(self.device)
        self.l_pix_w = 1
        self.l_fea_w = 0.0

        self.init_dataset()

    def init(self):
        self.outputs_dir = "./weights/"
        if not os.path.exists(self.outputs_dir):
            os.makedirs(self.outputs_dir)

        cudnn.benchmark = True
        seed = random.randint(1, 20000)
        t.manual_seed(seed)

    def init_dataset(self):
        self.dsConf = {
            'noise': False,
            'jpeg': False,
            'camera': False,
            'blur': True
        }
        self.train_dataset = qnDataset2('./qn_dataset/vsr_train_hwcbgr.h5', interval=self.train_interval, scale=self.scale)
        self.train_dataset.config(**self.dsConf)
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=self.num_workers,
                                      pin_memory=False,
                                      drop_last=True)
        self.eval_dataset = qnDataset2('./qn_dataset/vsr_val_hwcbgr.h5', interval=self.val_interval, scale=self.scale)
        self.eval_dataset.config(**self.dsConf)
        self.eval_dataloader = DataLoader(dataset=self.eval_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        self.trainds_len = len(self.train_dataset)
        print(len(self.train_dataset), len(self.eval_dataset))
        # exit(0)

    def adjust_lr(self, epoch):
        lr = self.lr * (self.lr_gamma ** (epoch // 15))
        lr = lr if lr > self.min_lr else self.min_lr
        print("adjust lr : epoch[{}] lr : {}".format(epoch, lr))
        for param_group in self.optimizer.param_groups:
            param_group['lr']= lr

    def train(self):
        best_epoch = 0
        best_psnr = 0.0
        for epoch in range(self.num_epochs - self.start_epoch):
            epoch += self.start_epoch
            # ?????????????????????
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
                    lossG = 0
                    real_img, ni_img = data['GT'], data['NI']
                    # real_img = real_img.to(device)
                    # ni_img = ni_img.to(device)
                    real_img = real_img.cuda()
                    ni_img = ni_img.cuda()
                    hr_fake = self.model(ni_img).clamp(0.0, 1.0)
                    loss_pix = self.cri_pix(hr_fake, real_img)
                    pix_losses.update(loss_pix.item(), len(real_img))
                    lossG += self.l_pix_w * loss_pix
                    if self.l_fea_w>0.001:
                        real_fea = self.netPerc(real_img).detach()
                        fake_fea = self.netPerc(hr_fake)
                        loss_fea = self.cri_fea(real_fea, fake_fea)
                        fea_losses.update(loss_fea.item(), len(real_img))
                    else:
                        loss_fea = 0
                        fea_losses.update(loss_fea, len(real_img))
                    lossG += self.l_fea_w * loss_fea

                    epoch_losses.update(lossG.item(), len(real_img))
                    self.optimizer.zero_grad()
                    lossG.backward()
                    self.optimizer.step()

                    # D
                    tq.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                    tq.update(len(real_img))
                    print('epoch:', epoch, i, epoch_losses.avg, pix_losses.avg, fea_losses.avg, d_losses.avg)

            if self.use_gpus:
                t.save(self.model.module, os.path.join(self.outputs_dir, '{}_epoch_{}.pth'.format(self.name, epoch)))
                self.model.module.eval()
            else:
                t.save(self.model, os.path.join(self.outputs_dir, '{}_epoch_{}.pth'.format(self.name, epoch)))
                self.model.eval()

            epoch_psnr = AverageMeter()
            epoch_bic_psnr = AverageMeter()
            for data in self.eval_dataloader:
                # real_img, ni_img = data
                real_img, ni_img = data['GT'], data['NI']
                real_img = real_img.cuda()
                ni_img = ni_img.cuda()
                if self.scale!=1:
                    bic_img = interpolate(ni_img, scale_factor=2, mode="bicubic", align_corners=False)
                else:
                    bic_img = ni_img
                with t.no_grad():
                    hsi_img = self.model(ni_img).clamp(0.0, 1.0)
                epoch_psnr.update(calc_psnr(hsi_img, real_img), len(ni_img))
                epoch_bic_psnr.update(calc_psnr(bic_img, real_img), len(ni_img))

            print(epoch, ', eval psnr: {:.2f}, bic psnr : {:.2f}, improve psnr : {:.2f}'.format(epoch_psnr.avg,
                        epoch_bic_psnr.avg, epoch_psnr.avg-epoch_bic_psnr.avg))
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

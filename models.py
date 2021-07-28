# -*- coding: utf-8 -*-


import torch.nn as nn
import common
from torch.nn.functional import interpolate
import torch as T
from torchsummary import summary
import functools
import torch.nn.functional as F


class HSISRNet(nn.Module):
    def __init__(self, scale=2):
        super(HSISRNet, self).__init__()
        self._conv = common.default_conv
        self._scale = scale
        self._n_feat = 64
        self._head = common.BasicBlock(3, self._n_feat, 9, bias=True, bn=False, act=nn.PReLU())
        self._resbody = common.ResBlock(self._conv, self._n_feat, 3, bn=False, res_scale=1, act=nn.ReLU(True), bias=False)
        self._up = common.Upsampler(self._conv, self._scale, self._n_feat, act=nn.PReLU, bias=True, bn=False)
        self._tail = common.BasicBlock(self._n_feat, 3, 9, bn=False, act=nn.Tanh(), bias=True)

    def forward(self, lr_img):
        bic_img = interpolate(lr_img, scale_factor=self._scale, mode="bicubic", align_corners=False)
        head_out = self._head(lr_img)
        x = self._resbody(head_out)
        # x = x + head_out
        x = self._up(x)
        x = self._tail(x)
        hr_img = x + bic_img
        return hr_img

class HRcanNet(nn.Module):
    def __init__(self, scale=2):
        super(HRcanNet, self).__init__()
        self._conv = common.default_conv
        self._scale = scale
        self._n_feat = 32
        self._kernel = 3
        self._reduction = 16
        self._act = nn.ReLU(True)
        self._res_scale = 1
        self._n_resblocks = 5
        self._n_resgroup = 5

        self._head = common.BasicBlock(3, self._n_feat, 5, bias=True, bn=False, act=nn.PReLU())
        lst_body = [
            common.ResidualGroup(
                self._conv, self._n_feat, self._kernel, self._reduction, act=self._act, res_scale=self._res_scale, n_resblocks=self._n_resblocks) \
            for _ in range(self._n_resgroup)]
        lst_body.append(self._conv(self._n_feat, self._n_feat, self._kernel))
        self._resbody = nn.Sequential(*lst_body)
        self._up = common.Upsampler(self._conv, self._scale, self._n_feat, act=False, bias=True, bn=False)
        self._tail = common.BasicBlock(self._n_feat, 3, 9, bn=False, act=nn.Tanh(), bias=True)

    def forward(self, lr_img):
        bic_img = interpolate(lr_img, scale_factor=self._scale, mode="bicubic", align_corners=False)
        head_out = self._head(lr_img)
        x = self._resbody(head_out)
        x = self._up(x)
        x = self._tail(x)
        hr_img = x + bic_img
        return hr_img
        # head_out = self._head(lr_img)
        # res = self._resbody(head_out)
        # res += head_out
        # x = self._up(res)
        # x = self._tail(x)
        # return x

class HRcanNet_new(nn.Module):
    def __init__(self, scale=2):
        super(HRcanNet_new, self).__init__()
        self._conv = common.default_conv
        self._scale = scale
        self._n_feat = 32
        self._kernel = 3
        self._reduction = 16
        self._act = nn.ReLU(True)
        self._res_scale = 1
        self._n_resblocks = 5
        self._n_resgroup = 5

        self._head = common.BasicBlock(3, self._n_feat, 5, bias=True, bn=False, act=nn.PReLU())
        lst_body = [
            common.ResidualGroup(
                self._conv, self._n_feat, self._kernel, self._reduction, act=self._act, res_scale=self._res_scale, n_resblocks=self._n_resblocks) \
            for _ in range(self._n_resgroup)]
        lst_body.append(self._conv(self._n_feat, self._n_feat, self._kernel))
        self._resbody = nn.Sequential(*lst_body)
        self._up = common.Upsampler(self._conv, self._scale, self._n_feat, act=False, bias=True, bn=False)
        self._tail = common.BasicBlock(self._n_feat, 3, 9, bn=False, act=nn.Tanh(), bias=True)
        self._forward = self.forward_scale_1 if self._scale==1 else self.forward_upscale

    def forward(self, lr_img):
        return self._forward(lr_img)

    def forward_scale_1(self, lr_img):
        head_out = self._head(lr_img)
        x = self._resbody(head_out)
        x = self._tail(x)
        x = x + lr_img
        return x

    def forward_upscale(self, lr_img):
        bic_img = interpolate(lr_img, scale_factor=self._scale, mode="bicubic", align_corners=False)
        head_out = self._head(lr_img)
        x = self._resbody(head_out)
        x = self._up(x)
        x = self._tail(x)
        hr_img = x + bic_img
        return hr_img


class HRRDBNet(nn.Module):
    def __init__(self):
        super(HRRDBNet, self).__init__()
        self._in_nc = 3
        self._out_nc = 3
        self._nf = 32
        self._rrdb_nb = 10
        self._gc = 64
        RRDB_block_f = functools.partial(common.RRDB, nf=self._nf, gc=self._gc)

        self.conv_first = nn.Conv2d(self._in_nc, self._nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = common.make_layer(RRDB_block_f, self._rrdb_nb)
        self.trunk_conv = nn.Conv2d(self._nf, self._nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(self._nf, self._nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(self._nf, self._nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(self._nf, self._out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        # fea = self.trunk_conv(self.RRDB_trunk(fea))

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        # x_up = F.interpolate(x, scale_factor=2, mode='bicubic')
        # out = x_up + out
        return out


def test():
    device = 'cuda'
    net = HSISRNet().to(device)
    inputs = T.rand(2, 3, 96, 96).to(device)
    outputs = net(inputs)
    print(outputs.shape)
    summary(net, input_size=(3, 96, 96), device=device)
    T.save(net, "d:/sisr.pth")
    pass

def test_hrcan():
    device = 'cuda'
    net = HRcanNet().to(device)
    inputs = T.rand(2, 3, 96, 96).to(device)
    outputs = net(inputs)
    print(outputs.shape)
    summary(net, input_size=(3, 96, 96), device=device)
    T.save(net, "d:/hrcan.pth")
    pass


def test_rrdb():
    device = 'cuda'
    net = HRRDBNet().to(device)
    inputs = T.rand(2, 3, 96, 96).to(device)
    outputs = net(inputs)
    print(outputs.shape)
    summary(net, input_size=(3, 96, 96), device=device)
    T.save(net, "d:/hrrdb.pth")

if __name__=="__main__":
    print("Hi, this is models test program")
    # test()
    test_hrcan()
    # test_rrdb()

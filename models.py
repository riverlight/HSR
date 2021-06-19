# -*- coding: utf-8 -*-


import torch.nn as nn
import common
from torch.nn.functional import interpolate
import torch as T
from torchsummary import summary


class HSISRNet(nn.Module):
    def __init__(self, scale=2):
        super(HSISRNet, self).__init__()
        self._conv = common.default_conv
        self._scale = scale
        self._n_feat = 64
        self._head = common.BasicBlock(3, self._n_feat, 9, bias=True, bn=False, act=nn.PReLU())
        self._resbody = common.ResBlock(self._conv, self._n_feat, 3, bn=False, res_scale=6, act=nn.ReLU(True), bias=False)
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
        self._n_resblocks = 10
        self._n_resgroup = 5

        self._head = common.BasicBlock(3, self._n_feat, 5, bias=True, bn=False, act=nn.PReLU())
        lst_body = [
            common.ResidualGroup(
                self._conv, self._n_feat, self._kernel, self._reduction, act=self._act, res_scale=self._res_scale, n_resblocks=self._n_resblocks) \
            for _ in range(self._n_resgroup)]
        lst_body.append(self._conv(self._n_feat, self._n_feat, self._kernel))
        self._resbody = nn.Sequential(*lst_body)
        self._up = common.Upsampler(self._conv, self._scale, self._n_feat, act=False, bias=True, bn=False)
        self._tail = common.BasicBlock(self._n_feat, 3, 9, bn=False, act=None, bias=True)

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

if __name__=="__main__":
    print("Hi, this is models test program")
    # test()
    test_hrcan()

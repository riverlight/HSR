# -*- coding: utf-8 -*-

import torch.nn as nn
import common
from torch.nn.functional import interpolate
import torch as T
from torchsummary import summary



class HRcanIRNet(nn.Module):
    def __init__(self, scale=2):
        super(HRcanIRNet, self).__init__()
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
        self._tail = common.BasicBlock(self._n_feat, 3, 9, bn=False, act=nn.Tanh(), bias=True)

    def forward(self, img):
        head_out = self._head(img)
        x = self._resbody(head_out)
        x = self._tail(x)
        out = x + img
        return out


def test_hrcanIR():
    device = 'cuda'
    net = HRcanIRNet().to(device)
    inputs = T.rand(2, 3, 96, 96).to(device)
    outputs = net(inputs)
    print(outputs.shape)
    summary(net, input_size=(3, 96, 96), device=device)
    T.save(net, "d:/hrcanIR.pth")
    pass

if __name__=="__main__":
    print("Hi, this is models test program")
    # test()
    test_hrcanIR()


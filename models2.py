# -*- coding: utf-8 -*-

import functools
import torch.nn as nn
import common
from torch.nn.functional import interpolate
import torch as T
from torchsummary import summary


class HRcanRRDBNet(nn.Module):
    def __init__(self):
        super(HRcanRRDBNet, self).__init__()
        self._in_nc = 3
        self._out_nc = 3
        self._scaler = 2
        self._nf = 32
        self._conv = common.default_conv
        self._act = nn.ReLU(True)
        self._reduction = 16
        self._res_scale = 1
        self._n_resblocks = 5
        self._rrdb_nb = 2
        self._gc = 32
        RRDB_block_f = functools.partial(common.RRDB, nf=self._nf, gc=self._gc)

        self._head = common.BasicBlock(self._in_nc, self._nf, 3, bias=True, bn=False, act=None)
        self._rcanbody = common.ResidualGroup(self._conv, self._nf, 3, self._reduction, act=self._act,
                                              res_scale=self._res_scale, n_resblocks=self._n_resblocks)
        self._RRDB_trunk = common.make_layer(RRDB_block_f, self._rrdb_nb)
        self._rcanbody2 = common.ResidualGroup(self._conv, self._nf, 3, self._reduction, act=self._act,
                                              res_scale=self._res_scale, n_resblocks=self._n_resblocks)
        self._trunk_conv = nn.Conv2d(self._nf, self._nf, 3, 1, 1, bias=True)

        #### upsampling
        self.upconv1 = nn.Conv2d(self._nf, self._nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(self._nf, self._out_nc * 4, 3, 1, 1, bias=True)
        self.upconv3 = nn.PixelShuffle(upscale_factor=2)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self._head(x)
        rcan = self._rcanbody(fea)
        trunk = self._trunk_conv(self._rcanbody2(self._RRDB_trunk(rcan)))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(fea))
        out = self.upconv3(self.upconv2(fea))

        return out



def test():
    device = 'cuda'
    net = HRcanRRDBNet().to(device)
    inputs = T.rand(2, 3, 96, 96).to(device)
    outputs = net(inputs)
    print(outputs.shape)
    summary(net, input_size=(3, 96, 96), device=device)
    T.save(net, "d:/HRcanRRDBNet.pth")


if __name__=="__main__":
    test()

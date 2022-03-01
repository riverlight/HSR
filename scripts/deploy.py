# -*- coding: utf-8 -*-

import torch


def main():
    device = 'cuda'
    weights_file = "../weights/BSRGANx2_good.pth"
    # weights_file = "d:/dejpeg_HRcanNet_best.pth"
    net = torch.load(weights_file)
    for param in net.parameters():
        param.requires_grad = False
    net = net.to(device)
    net.eval()

    # 注意模型输入的尺寸
    example = torch.rand(1, 3, 96, 96).to(device)
    traced_script_module = torch.jit.trace(net, example)
    traced_script_module.save(weights_file.replace(".pth", ".pt"))
    print('done')


if __name__=="__main__":
    main()


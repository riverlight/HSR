# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision
from torchsummary import summary



class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=34, use_bn=False, use_input_norm=True,
                 device=torch.device('cpu')):
        super(VGGFeatureExtractor, self).__init__()
        self.use_input_norm = use_input_norm
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            # model = torchvision.models.vgg19(pretrained=True)
            model = torchvision.models.vgg19(pretrained=False)
            state_dict = torch.load('/home/workroom/project/riverlight/vgg_model/vgg19-dcbb9e9d.pth')
            model.load_state_dict(state_dict)
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485 - 1, 0.456 - 1, 0.406 - 1] if input in range [-1, 1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229 * 2, 0.224 * 2, 0.225 * 2] if input in range [-1, 1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        # Assume input range is [0, 1]
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output

def test():
    device = 'cuda'
    use_bn = False
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    net = VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device).to(device)
    inputs = torch.rand(2, 3, 96, 96).to(device)
    outputs = net(inputs)
    print(outputs.shape)
    summary(net, input_size=(3, 96, 96), device=device)
    torch.save(net, "d:/VGGFeatureExtractor.pth")


if __name__=="__main__":
    test()

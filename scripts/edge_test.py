# -*- coding: utf-8 -*-


import numpy as np
import torch as t
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import cv2


def test():
    device = 'cpu'
    dir0 = "D:\\workroom\\tools\\dataset\\SR\\Set14\\image_SRF_2\\"
    dir1 = "D:\\workroom\\tools\\dataset\\SR\\Set14\\image_SRF_2\\"
    # lr_file = dir1 + "img_012_SRF_2_HR.png"
    lr_file = dir1 + "img_012_SRF_2_srcnn.png"
    image = cv2.imread(lr_file).astype(np.float32)
    image = t.from_numpy(image).to(device) / 255
    image = image.permute(2, 0, 1).unsqueeze(0)
    image = image[:,0:1,:,:]
    conv1 = nn.Conv2d(1, 1, 3, bias=False)
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    conv1.weight.data = t.from_numpy(sobel_kernel)
    edge1 = conv1(Variable(image))

    edge1 = edge1.mul(255.0).cpu().detach().numpy().squeeze(0).transpose(1, 2, 0)
    cv2.imshow("1", edge1)
    cv2.waitKey(0)
    print(edge1)
    edge1 = np.maximum(edge1, -edge1)
    edge1 = edge1.astype(np.uint8)
    print(edge1)
    img = cv2.merge([edge1, edge1, edge1])
    cv2.imwrite("d:/srcnn.jpg", img)

    print(img.shape)

    pass


if __name__=="__main__":
    test()

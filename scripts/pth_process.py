# -*- coding: utf-8 -*-

import torch as t
import sys
sys.path.append("../")
from models import HRcanNet
import cv2
import numpy as np


def obj_2_state():
    device = 'cpu'
    weight = "../weights/dejpeg_HRcanNet_best.pth"
    out_weight = weight.replace(".pth", "_state.pth")
    net = t.load(weight)
    t.save(net.state_dict(), out_weight)
    print("save done, start test..")
    test_state(out_weight)
    print('test done..')
    pass

def state_2_obj():
    state_weight = "../weights/dejpeg_HRcanNet_b_state.pth"
    weight = state_weight.replace("_state.pth", ".pth")
    if state_weight==weight:
        raise Exception("state_weight name is wrong")
    print("load state...")
    net = HRcanNet(scale=1)
    net.load_state_dict(t.load(state_weight), strict=True)
    print("save to weight..")
    t.save(net, weight)
    print("done..")

def test_state(state_weight):
    net = HRcanNet(scale=1)
    net.load_state_dict(t.load(state_weight), strict=True)
    net.eval()
    image_name = "d:/1.jpg"
    image = cv2.imread(image_name).astype(np.float32)
    image = t.from_numpy(image).to('cpu') / 255
    image = image.permute(2, 0, 1).unsqueeze(0)
    with t.no_grad():
        preds = net(image).clamp(0.0, 1.0)
    preds = preds.mul(255.0).cpu().numpy().squeeze(0).transpose(1, 2, 0)
    cv2.imwrite(image_name.replace('.jpg', '_s.jpg'), preds)

if __name__=="__main__":
    # obj_2_state()
    state_2_obj()

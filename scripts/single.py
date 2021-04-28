# -*- coding: utf-8 -*-

import cv2
import torch as T


def test():
    img = cv2.imread("d:/workroom/testroom/v0.png")
    print(img.shape)
    B, G, R = cv2.split(img)
    cv2.imshow("b", R)
    cv2.waitKey(0)
    pass


if __name__=="__main__":
    test()


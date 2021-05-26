# -*- coding: utf-8 -*-

import sys
sys.path.append("../")
import hutils
import glob
import os
import torchvision.transforms.functional as TF
from PIL import Image

def test():
    sdir = "D:\\workroom\\tools\\image\\ntire20\\track1-valid-gt\\"
    ddir = "D:\\workroom\\tools\\image\\ntire20\\track1-valid-gt-d2\\"
    sdir = "/home/workroom/project/riverlight/datasets/track1-valid-gt/"
    ddir = "/home/workroom/project/riverlight/datasets/track1-valid-gt-d2/"
    filenames = sorted(glob.glob(sdir + '*.png'))
    for name in filenames:
        print(name)
        basename = os.path.basename(name)
        input_img = Image.open(name)
        input_img = TF.to_tensor(input_img)
        resize2_img = hutils.imresize(input_img, 1.0 / 2, True)
        TF.to_pil_image(resize2_img).save(os.path.join(ddir, basename), 'PNG')
    pass


if __name__=="__main__":
    test()

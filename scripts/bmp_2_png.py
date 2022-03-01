# -*- coding: utf-8 -*-

import os
import cv2

def run(bmp_dir, png_dir):
    for count, name in enumerate(os.listdir(bmp_dir)):
        print(count, name)
        bmp_name = os.path.join(bmp_dir, name)
        png_name = os.path.join(png_dir, "p_"+name.replace(".bmp", ".png"))
        img = cv2.imread(bmp_name)
        cv2.imwrite(png_name, img)
    print("done")

if __name__=="__main__":
    bmp_dir = "D:/workroom/tools/dataset/exploration_database_and_code/pristine_images"
    png_dir = "D:/workroom/tools/dataset/exploration_database_and_code/png_images"
    run(bmp_dir, png_dir)

# -*- coding: utf-8 -*-

import os, sys, importlib, http
import subprocess, json, urllib
import platform, shutil
from urllib import request




def downloadUrl(url, localfile):
    importlib.reload(sys)  # 2
    # sys.setdefaultencoding('utf-8')  # 3

    try:
        # url = url.encode('utf-8')
        # print("url: ", url)
        urllib.request.urlretrieve(url, localfile)
    except IOError:
        return "err"
    else:
        fsize = os.path.getsize(localfile)
        if (fsize == 0):
            return "err"
        return "ok"


def test():
    urltxt = "D:\\workroom\\tools\\dataset\\qtt\\qttzsv_out_hd2.txt"
    with open(urltxt, 'rt') as ft:
        lines = ft.readlines()

    dst_dir = "D:\\workroom\\tools\\dataset\\SR\\vsr_ds"
    for n, url in enumerate(lines):
        if ".mp4" not in url:
            continue
        url = url.replace("\n", "")
        print(n, url)
        dst_name = os.path.join(dst_dir, "{}.mp4".format(n))
        downloadUrl(url, dst_name)
    print('done')

def rename():
    src_dir = "D:\\workroom\\tools\\dataset\\douyin\\Download\\src"
    dst_dir = "D:\\workroom\\tools\\dataset\\douyin\\Download\\dst"
    for n, name in enumerate(os.listdir(src_dir)):
        print(n, name)
        os.rename(os.path.join(src_dir, name), os.path.join(dst_dir, str(n)+".mp4"))

if __name__=="__main__":
    # test()
    rename()

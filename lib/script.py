import os
import shutil
from os.path import join as pjoin

root = '../../runs/v19unet_napse'

f_list = os.listdir(root)

for f in f_list:
    useful = False
    files = os.listdir(pjoin(root, f))
    for x in files:
        if 'pkl' in x:
            useful = True

    if not useful:
        shutil.rmtree(pjoin(root, f))

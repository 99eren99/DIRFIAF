## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import numpy as np
import os
import argparse
from tqdm import tqdm

import utils
import cv2

import sys
sys.path.append("..")

from glob import glob

from statistics import mean

parser = argparse.ArgumentParser(description='Single Image Motion Deblurring using Restormer')

parser.add_argument('--input_dir', default='./Datasets/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--dataset', default='COVERAGE', type=str, help='Test Dataset') # ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']

args = parser.parse_args()

sharpen_filter=np.array([[0,-1,0],
                        [-1,5,-1],
                        [0,-1,0]])


dataset = args.dataset
result_dir  = os.path.join(args.result_dir, dataset+"BlurAndSharp")
os.makedirs(result_dir, exist_ok=True)
psnrs=[]
ssims=[]

#inp_dir = "./Datasets/DSO-1"
inp_dir = "./Datasets/COVERAGE"
files = list(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.jpg')) + glob(os.path.join(inp_dir, '*.tif')))

for file_ in tqdm(files):
    img=utils.load_img(file_)
    blurred_img=cv2.GaussianBlur(img,(5,5),0)
    restored=cv2.filter2D(blurred_img,-1,sharpen_filter).clip(0,255).astype("uint8")

    psnr=utils.calculate_psnr(img,restored)
    ssim=utils.calculate_ssim(img,restored)
    print(file_,f"psnr:{psnr} , ssim:{ssim}")
    psnrs.append(psnr)
    ssims.append(ssim)
    utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0]+'.png')), restored)
    
print(f"{dataset}, blur and sharp, average psnr {mean(psnrs)}, average ssim {mean(ssims)}")

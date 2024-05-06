import torch
from piq import brisque, SSIMLoss
from PIL import Image
import os
import numpy as np
from statistics import mean

directories1=[r"C:\Users\Eren Tahir\Desktop\restoration\datasets\COVERAGE\image",
             r"C:\Users\Eren Tahir\Desktop\restoration\processedData\Gaussian_Color_Denoising\non_blind\COVERAGE\15",
             r"C:\Users\Eren Tahir\Desktop\restoration\processedData\Gaussian_Color_Denoising\non_blind\COVERAGE\25",
             r"C:\Users\Eren Tahir\Desktop\restoration\processedData\JPEGcompress_RemoveArtifacts\COVERAGE_fbcnn_color\50",
             r"C:\Users\Eren Tahir\Desktop\restoration\processedData\JPEGcompress_RemoveArtifacts\COVERAGE_fbcnn_color\70",
             r"C:\Users\Eren Tahir\Desktop\restoration\processedData\SharpAndBlur\COVERAGEBlurAndSharp",
             r"C:\Users\Eren Tahir\Desktop\restoration\processedData\ResizeAndResize\COVERAGE_resizeResize"]

directories2=[r"C:\Users\Eren Tahir\Desktop\restoration\datasets\dso-1\tifs-database\DSO-1",
             r"C:\Users\Eren Tahir\Desktop\restoration\processedData\Gaussian_Color_Denoising\non_blind\DSO-1\15",
             r"C:\Users\Eren Tahir\Desktop\restoration\processedData\Gaussian_Color_Denoising\non_blind\DSO-1\25",
             r"C:\Users\Eren Tahir\Desktop\restoration\processedData\JPEGcompress_RemoveArtifacts\DSO-1_fbcnn_color\50",
             r"C:\Users\Eren Tahir\Desktop\restoration\processedData\JPEGcompress_RemoveArtifacts\DSO-1_fbcnn_color\70",
             r"C:\Users\Eren Tahir\Desktop\restoration\processedData\SharpAndBlur\DSO-1BlurAndSharp",
             r"C:\Users\Eren Tahir\Desktop\restoration\processedData\ResizeAndResize\DSO-1_resizeResize",
             r"C:\Users\Eren Tahir\Desktop\restoration\processedData\DownscaleAndUpscale\DSO-1_SwinFIR"]

for directory in directories1+directories2:
    brisques=[]
    for imgName in os.listdir(directory):
        imgPath=os.path.join(directory,imgName)
        img=Image.open(imgPath).convert("RGB")
        img=torch.tensor(np.expand_dims(np.transpose(img,(2, 0, 1)), axis=0), dtype=torch.float) / 256.0
        img.to("cuda")

        brisque_index = brisque(img, data_range=1.0)
        brisques.append(brisque_index.item())
        del img
    print(f"{directory}\nmean brisque:{mean(brisques)}")


import numpy as np
import cv2
from PIL import Image
import os
from loadModel import trufor as model
import torch


model = model.to("cuda")

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

labels=[]
for directory in directories1+directories2:
    tp=0
    tn=0
    fp=0
    fn=0
    for imgName in os.listdir(directory):
        if "COVERAGE" in directory:
            label=1 if "t" in imgName.split(".")[0] else 0
        else:
            label=1 if "splicing" in imgName else 0

        imgPath=os.path.join(directory,imgName)
        image=Image.open(imgPath).convert("RGB")
        with torch.no_grad():
            predictedLabel,probability=model(image)

        #print(imgName,predictedLabel)

        if label==1:
            if predictedLabel==1:
                tp+=1
            else:
                fn+=1
        else:
            if predictedLabel==1:
                fp+=1
            else:
                tn+=1
                
    print(f"{directory}\ntp:{tp}, tn:{tn}, fp:{fp}, fn:{fn}\nrecall:{tp/(tp+fn)}")

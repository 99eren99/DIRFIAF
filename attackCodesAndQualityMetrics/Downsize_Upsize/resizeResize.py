from PIL import Image,ImageFilter
import numpy as np
import os
import utils
from statistics import mean

def main(dataset):

    psnrs=[]
    ssims=[]

    imagePaths=list(map(lambda x: f"./{dataset}/"+x,os.listdir(dataset)))

    for path in imagePaths:
        image=Image.open(path)
        try:
            image=image.convert("RGB")
        except:
            pass

        width, height = image.size

        result=image.copy().resize((int(width*0.5),int(height*0.75)),Image.LANCZOS)
        result=result.resize((width,height),Image.BICUBIC)
        result.save(f"./{dataset}_resizeResize/{path.split('/')[-1]}")

        psnr=utils.calculate_psnr(np.array(image),np.array(result))
        ssim=utils.calculate_ssim(np.array(image),np.array(result))
        print(path,f"psnr:{psnr} , ssim:{ssim}")
        psnrs.append(psnr)
        ssims.append(ssim)

    print(f"{dataset}, resize and resize, average psnr {mean(psnrs)}, average ssim {mean(ssims)}")


if __name__ == "__main__":
    for dataset in ["COVERAGE","DSO-1"]:
        os.makedirs(f"./{dataset}_resizeResize/",exist_ok=True)
        main(dataset)
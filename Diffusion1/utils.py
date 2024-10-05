import numpy as np 
import torch 
import torchvision
from PIL import Image
import os
from matplotlib import pyplot as plt 

def plot_imagesH(images):
    figureSize = plt.figure(figsize=(10,5))
    plt.imshow(torch.cat([image for image in images.cpu()],dim = -1).permute(1,2,0).cpu())
    plt.show()
    
def plot_imagesV(images):
    figureSize = plt.figure(figsize=(5,10))
    plt.imshow(torch.cat([image for image in images.cpu()], dim = -2).permute(1,2,0).cpu())
    plt.show()

def saveImages(images,path,normalized=False,**kwargs):
    grid = torchvision.utils.make_grid(images,**kwargs)
    images = grid.permute(1,2,0).cpu().numpy()
    if(normalized):
        images = (images*255).astype(np.uint8)
    image = Image.fromarray(images)
    image.save(path)
    


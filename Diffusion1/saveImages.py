import torchvision
import torch 
import os
from PIL import Image
import numpy as np

def saveImages(images,path,normalized=False,**kwargs):
    # make grid
    grid = torchvision.utils.make_grid(images,**kwargs)
    
    # Images
    images = grid.permute(1,2,0).cpu().numpy()
    if normalized:
        images = (images*255).astype(np.uint8)
    
    # Load Image
    image = Image.fromarray(images)
    
    # Save
    image.save(path)
    
numImages = 16
imageSize = 64

images1 = [torch.rand(3,imageSize,imageSize) for _ in range(numImages)]
images2 = [torch.randint(0,256,(3,imageSize,imageSize),dtype=torch.uint8) for _ in range(numImages)]

outputPath1 = "output_images/grid1.png"
outputPath2 = "output_images/grid2.png"

outputDirectory1 = os.path.dirname(outputPath1)
outputDirectory2 = os.path.dirname(outputPath2)

os.makedirs(outputDirectory1,exist_ok=True)
os.makedirs(outputDirectory2,exist_ok=True)

saveImages(images1,outputPath1,True,nrow=4)
saveImages(images2,outputPath2,False,nrow=4)
import torch 
from matplotlib import pyplot as plt  

def plot_imagesH(images):
    figureSize = plt.figure(figsize=(10,5))
    plt.imshow(torch.cat([image for image in images.cpu()], dim = -1).permute(1,2,0).cpu())
    plt.show()

def plot_imagesH2(images):
    figureSize = plt.figure(figsize=(10,5))
    plt.imshow(torch.cat([image for image in images.to('cpu')], dim = -1).permute(1,2,0).to('cpu'))
    plt.show()

numImages = 4
imageSize = 64

# Generate random images
images = torch.rand(numImages,3,imageSize,imageSize)

# plot the images
plot_imagesH2(images)
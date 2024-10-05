import torch 
from matplotlib import pyplot as plt  

def plot_imagesV(images):
    figureSize = plt.figure(figsize=(5,10))
    plt.imshow(torch.cat([image for image in images.cpu()], dim = -2).permute(1,2,0).cpu())
    plt.show()
    
def plot_imagesV2(images):
    figureSize = plt.figure(figsize=(5,10))
    plt.imshow(torch.dat([image for image in images.to('cpu')], dim = -2).permute(1,2,0).to('cpu'))
    plt.show()

numImages = 4
imageSize = 64

# Generate Random Noise
images = torch.rand(numImages,3,imageSize,imageSize)

# plot
plot_imagesV(images)
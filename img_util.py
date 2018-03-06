from scipy.misc import imread, imresize
import numpy as np
import matplotlib.pyplot as plt

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

def GetAndProcessImg(imgPath, n=256):
    img = imread(imgPath)
    img = rgb2gray(img)
    return ScaleImg(imresize(img, [n, n], interp='bicubic'))

def ScaleImg(img):
    return (np.float32(img) - np.min(img)) / (np.max(img) - np.min(img)) * 255.

def NonZeroMap(in_mag):
    x =np.where(in_mag>0,255,0)
    return np.uint8(x)


def ShowImg(img, title_str='temp', colormap_opt='gray'):
    img_show = np.uint8(ScaleImg(img))
    plt.figure()
    plt.imshow(img_show, cmap=plt.get_cmap(colormap_opt))
    plt.title(title_str)

    plt.show()

def PlotImg_NoBlock(img, title_str='temp', colormap_opt='gray'):

    plt.figure()
    plt.imshow(img, cmap=plt.get_cmap(colormap_opt))
    plt.title(title_str)
    # f.show()

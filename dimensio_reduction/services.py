from imageProcessing import *
import os

from skimage.io import imread, imshow
from skimage.filters import gaussian, threshold_otsu
from skimage import measure
import matplotlib.pyplot as plt

def Dataparsing_cv2(imagePath, savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    # using cv
    # read the image
    image = cv2.imread(imagePath + '.jpg')
    # Reshape image if needed- currently rotate by 90degrees regardless of the position
    # Later- consider to automatically find the orientation of the hand and rotating according to the orientation
    image = reshape_image(image)

    reducedImage = image[int(image.shape[0] * 0.05):int(image.shape[0] * 0.95),
                   int(image.shape[1] * 0.12):int(image.shape[1] * 0.82), :]
    plotimage(reducedImage)

    # convert to gray
    grayImage = cv2.cvtColor(reducedImage, cv2.COLOR_BGR2GRAY)
    plotimage(grayImage, 'gray')

    # Do mean segmentation
    binaryImageINV = mean_threshold_segmentation_cv(grayImage, True)
    plotimage(binaryImageINV, 'gray')

    binaryImage = mean_threshold_segmentation_cv(grayImage)
    plotimage(binaryImage, 'gray')

    return reducedImage, grayImage, binaryImage, binaryImageINV

def dataParsing_plt (imagePath, savepath):

    # read and plot the image
    image = plt.imread(imagePath + '.jpg')  # cv2.imread(imagePath+'.jpg')
    # Reshape image if needed- currently rotate by 90degrees regardless of the position
    # Later- consider to automatically find the orientation of the hand and rotating according to the orientation
    image = reshape_image(image)
    plotimage(image, None, savepath + 'original_image.png')
    # reduced_gray_r_Op = image[int(image.shape[0]*0.05):int(image.shape[0]*0.95),int(image.shape[1]*0.12):int(image.shape[1]*0.82),:]

    # convert to RGB
    # image = cv2.cvtColor(reduced_gray_r_Op, cv2.COLOR_BGR2RGB)
    # convert to gray scale
    # gray = cv2.cvtColor(reduced_gray_r_Op, cv2.COLOR_RGB2GRAY)
    grayImage = rgb2gray(image)
    plotimage(gray, 'gray', savepath + 'gray_scale.png')

    grayShape = gray.shape

    # Mean threshold segmentation
    binaryImage = mean_threshold_segmentation(grayImage)

    binaryImage = reduce_image_size(binaryImage, 0, 0)
    # Reduce the size of picture
    # reduced_gray_r_Op = gray_r[200:1200]  # reduce_image_size(gray_r, 0, 1)
    plot_comparison(grayImage, binaryImage, 'mean_threshold', savepath + 'Binary_reduced_size.png')

    # Do closing operation on opposite gray scale image
    binaryImageINV = negative_binary_image(grayImage)
    # Reduce the size of picture
    binaryImageINV = reduce_image_size(binaryImageINV, 1, 0)
    plot_comparison(grayImage, binaryImageINV, 'mean_threshold-negative image', savepath+'Binary_negative_reduced_size.png',)

    return image, grayImage, binaryImage, binaryImageINV

def regionprops (original):

    blurred = gaussian(original, sigma=.8)
    binary = blurred > threshold_otsu(blurred)
    labels = measure.label(binary)

    plots = {'Original': original, 'Blurred': blurred,
             'Binary': binary, 'Labels': labels}
    fig, ax = plt.subplots(1, len(plots))
    for n, (title, img) in enumerate(plots.items()):
        cmap = plt.cm.gnuplot if n == len(plots) - 1 else plt.cm.gray
        ax[n].imshow(img, cmap=cmap)
        ax[n].axis('off')
        ax[n].set_title(title)
    plt.show(fig)

    props = measure.regionprops(labels)
    for prop in props:
        print('Label: {} >> Object size: {}'.format(prop.label, prop.area))
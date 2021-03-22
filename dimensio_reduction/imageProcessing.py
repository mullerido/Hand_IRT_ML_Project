import numpy as np
import cv2
from skimage import data, filters, color
from skimage.transform import hough_circle
from skimage.feature import peak_local_max
from skimage.draw import circle_perimeter
from skimage.color import rgb2gray
from skimage import morphology
import copy
from scipy import ndimage
from plotUtils import *
# from impy import imarray
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import imread

def reduce_image_size(image, thr=0.0, ax=0.0):
    # reducedImage = image[int(np.round(image.shape[0]*thr)):int(np.round(image.shape[0]*ax)),
    #                0:int(image.shape[1]),:]

    #in case of ndarray using plotUtils to import image
    idx = np.argwhere(np.all(image == thr, axis=ax))
    reducedImage = np.delete(image, idx.tolist(), 1-ax)

    return reducedImage

def reshape_image(image):
    imageShape = image.shape
    if imageShape[0] > imageShape[1]:
        image = ndimage.rotate(image, 90)
    return image

def mean_threshold_segmentation_cv(image, isInv=False):
    meanVal = image.mean()
    t, binary = cv2.threshold(image, meanVal, meanVal, cv2.THRESH_OTSU)

    if isInv:
        _, binary = cv2.threshold(image, meanVal, meanVal, cv2.THRESH_BINARY)
    else:
        _, binary = cv2.threshold(image, meanVal, meanVal, cv2.THRESH_BINARY_INV)

    return binary

def mean_threshold_segmentation(image):
    meanVal = image.mean()
    binaryImage = np.zeros(image.shape)
    binaryImage[image >= meanVal] = 1
    return binaryImage
    #
    # meanVal = gray.mean()
    # gray_r = copy.deepcopy(image)
    # gray_r[gray >= meanVal] = 1
    # gray_r[gray < meanVal] = 0

def negative_binary_image(binaryImage):
    negativeImage = np.zeros(binaryImage.shape)
    negativeImage[binaryImage==0]=1
    return negativeImage
# def apply_binary_mask_on_image (image, mask):
#     # outImage =

def morphology_closing(image, rad):
    # Structuring Element
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (rad, rad))

    # defining the closing function
    closedImage = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    return closedImage

def morphology_opening (image, rad):
    # Structuring Element
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (rad, rad))

    # defining the opening function
    openedImage = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    return openedImage

def palm_and_fingers_identifier_close(image, max_itter=200, radDiff=4 ):

    rad = 2
    itter = 1
    foreGroundPixels = np.array(np.sum(sum(image == 0)))
    forGroundCond = True
    while itter <= max_itter and forGroundCond:  # and foreGroundPixels[-1] < (imageShape[0]*imageShape[0])*0.2: #rad<= min(imageShape[0:-1]):
        print(itter)
        closedImage = morphology_closing(image, rad)
        currentForeGround = np.sum(sum(closedImage == 0))
        foreGroundPixels = np.append(foreGroundPixels, currentForeGround)
        # if itter == 1 or itter % 50 == 0:
        #     plot_comparison(image, closedImage, 'closing' + str(rad))
        forGroundCond = foreGroundPixels[-1] > 0  # (imageShape[0]*imageShape[1])*0.1
        rad += radDiff
        itter += 1
    plotOpeningOverTime(foreGroundPixels, np.arange(2, rad+radDiff, radDiff))

    return foreGroundPixels, rad

def palm_and_fingers_identifier_open(image, max_itter=200, radDiff=4 ):
#     Perform Opening operation iteratively to find the fingers and palm
    rad=2
    itter=1
    foreGroundPixels = np.array(np.sum(sum(image >0)))
    forGroundCond=True
    while itter<=max_itter and forGroundCond: #and foreGroundPixels[-1] < (imageShape[0]*imageShape[0])*0.2: #rad<= min(imageShape[0:-1]):
        print(itter)
        openedImage = morphology_opening(image, rad)
        currentForeGround = np.sum(sum(openedImage>0))
        foreGroundPixels = np.append(foreGroundPixels,currentForeGround)
        # if itter==1 or itter%50==0:
        #      plot_comparison(image, openedImage,'opening'+str(rad))
        forGroundCond = foreGroundPixels[-1] > 0#(imageShape[0]*imageShape[1])*0.1
        rad+=radDiff
        itter+=1
    plotOpeningOverTime(foreGroundPixels, np.arange(2, rad+radDiff, radDiff))

        # Open- using skimage
        # while itter<=max_itter: #and foreGroundPixels[-1] < (imageShape[0]*imageShape[0])*0.2: #rad<= min(imageShape[0:-1]):
        #     print(itter)
        #     selem=morphology.disk(rad)
        #     openedImage = morphology.opening(cutGray_r, selem)
        #     currentForeGround = np.sum(sum(openedImage==1))
        #     foreGroundPixels = np.append(foreGroundPixels,currentForeGround)
        #     if itter==1 or itter%50==0:
        #         plot_comparison(cutGray_r, openedImage,'opening'+str(rad))
        #     rad+=4
        #     itter+=1

    return foreGroundPixels, rad

def find_contour(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    output = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # draw contours
    for i in range(0, len(contours)):
        cv2.drawContours(output, contours, i, (np.random.randint(0, 255), np.random.randint(0, 255)),
                         2)  # , random.randint(0, 255)),2)

    # create windows to display images
    cv2.namedWindow("contours", cv2.WINDOW_FULLSCREEN)

    # display images
    cv2.imshow("contours", output)

    return output

def find_edges(image, lowThreshold=30, highTHreshold=150, apertureSize=3):

    edges = cv2.Canny(image, lowThreshold, highTHreshold, apertureSize=apertureSize)
    cv2.namedWindow("edge", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("edge", edges)

    return edges

def find_fingers_circles(edges, image, impy=None):

    hough_radii = np.arange(20, 60, 2)
    hough_res = hough_circle(edges, cv2.HOUGH_GRADIENT, hough_radii)

    centers = []
    accums = []
    radii = []

    for radius, h in zip(hough_radii, hough_res):
        # For each radius, extract two circles
        peaks = peak_local_max(h, num_peaks=5)
        centers.extend(peaks - hough_radii.max())
        accums.extend(h[peaks[:, 0], peaks[:, 1]])
        radii.extend([radius, radius])
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))

    # Draw the most prominent 5 circles
    image = color.gray2rgb(edges)
    for idx in np.argsort(accums)[::-1][:5]:
        center_x, center_y = centers[idx]
        radius = radii[idx]
        cx, cy = circle_perimeter(center_y, center_x, radius)
        image[cy, cx] = (250, 0, 0)

    ax.imshow(image, cmap=plt.cm.gray)
    plt.show()

    #
    # circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20,
    #                            param1=50, param2=32, minRadius=0, maxRadius=0)
    # circles = np.uint16(np.around(circles))
    # for i in circles[0, :]:
    # # draw the outer circle
    #     cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
    #     # draw the center of the circle
    #     cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
    #     cv2.imshow('detected circles', image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #


## from GitHub
def smoothen(img, display):
    # Using a 3x3 gaussian filter to smoothen the image
    gaussian = np.array([[1 / 16., 1 / 8., 1 / 16.], [1 / 8., 1 / 4., 1 / 8.], [1 / 16., 1 / 8., 1 / 16.]])
    img.load(img.convolve(gaussian))
    if display:
        img.disp
    return img

def edge(img, threshold, display=False):
    # Using a 3x3 Laplacian of Gaussian filter along with sobel to detect the edges
    laplacian = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    # Sobel operator (Orientation = vertical)
    sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    # Generating sobel horizontal edge gradients
    G_x = img.convolve(sobel)

    # Generating sobel vertical edge gradients
    G_y = img.convolve(np.fliplr(sobel).transpose())

    # Computing the gradient magnitude
    G = pow((G_x * G_x + G_y * G_y), 0.5)

    G[G < threshold] = 0
    L = img.convolve(laplacian)
    if L is None:  # Checking if the laplacian mask was convolved
        return
    (M, N) = L.shape

    temp = np.zeros((M + 2, N + 2))  # Initializing a temporary image along with padding
    temp[1:-1, 1:-1] = L  # result hold the laplacian convolved image
    result = np.zeros((M, N))  # Initializing a resultant image along with padding
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            if temp[i, j] < 0:  # Looking for a negative pixel and checking its 8 neighbors
                for x, y in (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1):
                    if temp[i + x, j + y] > 0:
                        result[i - 1, j - 1] = 1  # If there is a change in the sign, it is a zero crossing
    img.load(np.array(np.logical_and(result, G), dtype=np.uint8))
    if display:
        img.disp
    return img

def detectCircles(img, threshold, region, radius=None):
    (M, N) = img.shape
    if radius == None:
        R_max = np.max((M, N))
        R_min = 10
    else:
        [R_max, R_min] = radius

    R = R_max - R_min
    # Initializing accumulator array.
    # Accumulator array is a 3 dimensional array with the dimensions representing
    # the radius, X coordinate and Y coordinate resectively.
    # Also appending a padding of 2 times R_max to overcome the problems of overflow
    A = np.zeros((R_max, M + 2 * R_max, N + 2 * R_max))
    B = np.zeros((R_max, M + 2 * R_max, N + 2 * R_max))

    # Precomputing all angles to increase the speed of the algorithm
    theta = np.arange(0, 180) * np.pi / 180
    edges = np.argwhere(img[:, :])  # Extracting all edge coordinates
    for val in range(R):
        r = R_min + val
        # Creating a Circle Blueprint
        bprint = np.zeros((2 * (r + 1), 2 * (r + 1)))
        (m, n) = (r + 1, r + 1)  # Finding out the center of the blueprint
        for angle in theta:
            x = int(np.round(r * np.cos(angle)))
            y = int(np.round(r * np.sin(angle)))
            bprint[m + x, n + y] = 1
        constant = np.argwhere(bprint).shape[0]
        for x, y in edges:  # For each edge coordinates
            # Centering the blueprint circle over the edges
            # and updating the accumulator array
            X = [x - m + R_max, x + m + R_max]  # Computing the extreme X values
            Y = [y - n + R_max, y + n + R_max]  # Computing the extreme Y values
            A[r, X[0]:X[1], Y[0]:Y[1]] += bprint
        A[r][A[r] < threshold * constant / r] = 0

    for r, x, y in np.argwhere(A):
        temp = A[r - region:r + region, x - region:x + region, y - region:y + region]
        try:
            p, a, b = np.unravel_index(np.argmax(temp), temp.shape)
        except:
            continue
        B[r + (p - region), x + (a - region), y + (b - region)] = 1

    return B[:, R_max:-R_max, R_max:-R_max]

def displayCircles(A, img):
    fig, ax1 = plt.subplots()
    plt.imshow(img)
    circleCoordinates = np.argwhere(A)  # Extracting the circle information
    circle = []
    for r, x, y in circleCoordinates:
        circle.append(plt.Circle((y, x), r, color=(1, 0, 0), fill=False))
        ax1.add_subplot(111).add_artist(circle[-1])
    plt.show()

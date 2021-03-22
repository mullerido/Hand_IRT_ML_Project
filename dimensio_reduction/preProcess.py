from services import *
from utils import *
import cv2 as cv
from imageProcessing import *
import pandas as pd
if __name__ == '__main__':
    imagePath =r'C:\Users\ido\Dropbox\Public\Study\Thesis\Data\7_19.11_Daniel_305328619\right\gravitation\flir_20191119T164150'
    imagePath =r'C:\Users\ido\Dropbox\Public\Study\Thesis\Data\No11\Right\No11_R_B_T0'
    savepath = r'C:\Users\ido\Google Drive\Thesis\Data\Results'
    df = pd.read_csv(imagePath+'.csv',  header=None)
    image = df.values
    minImage = image.min()
    imageRange = image.max() - image.min()
    grayImage = np.array(np.round(255*((image-minImage)/imageRange)),dtype=np.uint8)
    # a = reduce_image_size(grayImage,0,1 )
    binaryImageINV = mean_threshold_segmentation_cv(grayImage, True)
    regionprops(grayImage)
    # image, grayImage, binaryImage, binaryImageINV = Dataparsing_cv2(imagePath, savepath)
    # image, grayImage, binaryImage, binaryImageINV = dataParsing_plt(imagePath, savepath)
    fingersContours = find_contour(grayImage)

    OpFlag = 'open'
    if OpFlag == 'open':
        max_itter = 600
        radDiff = 1
        foreGroundPixels, rad = palm_and_fingers_identifier_open(binaryImageINV, max_itter, radDiff)
        np.savetxt(savepath + r'\forGroundPixels_cv2.txt', foreGroundPixels)
        allRads = np.arange(2, rad + radDiff, radDiff)
        np.savetxt(savepath + r'\allRads_cv2.txt', allRads)

    elif OpFlag == 'close':
        max_itter = 400
        radDiff = 1
        foreGroundPixels, rad = palm_and_fingers_identifier_close(binaryImage, max_itter, radDiff)
        np.savetxt(savepath + r'\forGroundPixels.txt', foreGroundPixels)
        allRads = np.arange(2, rad + radDiff, radDiff)
        np.savetxt(savepath + r'\allrad.txt', allRads)

    else:

        foreGroundPixels = np.loadtxt(r'C:\Projects\Thesis\output\forGroundPixels.txt')
        allRads = np.loadtxt(r'C:\Projects\Thesis\output\rad.txt')
        plotOpeningOverTime(foreGroundPixels, allRads, savepath + 'Close_Over_Time.png')

    # Take out fingers first
    localMaximasInds = foundTwoHils(foreGroundPixels)

    openedImage = morphology_closing(binaryImage, localMaximasInds[0]+2)
    fingers =openedImage- binaryImage
    plotimage(fingers,'gray')
    fingers[fingers < 1] = 0
    fingers = morphology_opening(fingers, 6)

    imageCopy = copy.deepcopy(image)
    imageCopy[fingers<fingers.mean(),:]=0
    fingersGray = cv.cvtColor(imageCopy, cv.COLOR_BGR2GRAY)

    # fingersContours = find_contour(fingers)

    fingersEdges = find_edges(fingers)
    FingerEdges_gray = fingers
    fingersGray[fingersEdges>0]=255
    # find_fingers_circles(fingersEdges, fingersEdges)
    ## Start Huge implementation here:
    # res = edge(fingersEdges, 50, display=True)  # set display to True to display the edge image
    # detectCircles takes a total of 4 parameters. 3 are required.
    # The first one is the edge image. Second is the thresholding value and the third is the region size to detect peaks.
    # The fourth is the radius range.
    res = detectCircles(fingersEdges, 8.1, 15, radius=[100, 10])
    displayCircles(res, fingers)

    ## Resume here

    # circles = cv.HoughCircles(fingersGray, cv.HOUGH_GRADIENT, 1.2, 100)#,param1=50, param2=30, minRadius=0, maxRadius=0)
    circles = cv.HoughCircles(fingersEdges, cv2.HOUGH_GRADIENT, 1, 20, param1=30,param2=22.5,minRadius=0,maxRadius=200)
    if circles is not None:
        # If there are some detections, convert radius and x,y(center) coordinates to integer
        circlesNP = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circlesNP:
            # Draw the circle in the output image
            cv2.circle(fingersEdges, (x, y), r, (0, 255, 0), 3)
            # Draw a rectangle(center) in the output image
            cv2.rectangle(fingersEdges, (x - 2, y - 2), (x + 2, y + 2), (0, 255, 0), -1)

    cv2.imshow("Detections", fingersEdges)
    # cv2.imwrite("CirclesDetection.jpg", fingersGray)
    cv2.waitKey()

    contours, hierarchy = cv2.findContours(fingers, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image_contour = cv2.drawContours(fingers, contours, -1, (0, 255, 0), 2)
    cv2.imshow('Contours', fingers)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    openedImage = morphology_closing(binaryImage, localMaximasInds[1])
    palm = openedImage - binaryImage
    plotimage(palm, 'gray')
    palm[fingers < 1] = 0
    palm = morphology_opening(fingers, 6)

    imageCopy = copy.deepcopy(image)
    imageCopy[palm < palm.mean(), :] = 0
    palmGray = cv.cvtColor(imageCopy, cv.COLOR_BGR2GRAY)

    x=1

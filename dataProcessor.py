# File        :   dataProcessor.py
# Version     :   1.0.0
# Description :   Script that preprocesses images...
# Date:       :   May 01, 2022
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

import numpy as np
import cv2

# Path management:
from imutils import paths

# Reads image via OpenCV:
def readImage(imagePath):
    # Loads image:
    inputImage = cv2.imread(imagePath)
    # Checks if image was successfully loaded:
    if inputImage is None:
        print("readImage>> Error: Could not load Input image.")
    return inputImage


# Defines a re-sizable image window:
def showImage(imageName, inputImage):
    cv2.namedWindow(imageName, cv2.WINDOW_NORMAL)
    cv2.imshow(imageName, inputImage)
    cv2.waitKey(0)


# Writes an PGN image:
def writeImage(imagePath, inputImage):
    imagePath = imagePath
    cv2.imwrite(imagePath, inputImage, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print("Wrote Image: " + imagePath)


# Clamps an integer to a valid range:
def clamp(val, minval, maxval):
    if val < minval: return minval
    if val > maxval: return maxval
    return val


# Image path
dataSetPath = "D://opencvImages//sudoku//dataset//test//"
pathStringLength = len(dataSetPath)

# Image counter:
imageCounter = 0

# Load each image path of the dataset:
imagePaths = sorted(list(paths.list_images(dataSetPath)))

resizedWidth = 70
resizedHeight = 70

# Loop over the input images and load em:
for imagePath in imagePaths:

    fileName = imagePath[pathStringLength+2:len(imagePath)]

    print(fileName)

    # Read the image via OpenCV:
    image = cv2.imread(imagePath)

    # Get dimensions:
    (originalHeight, originalWidth) = image.shape[:2]

    if originalWidth != resizedWidth or originalHeight != resizedHeight:
        print("Resizing: "+fileName+" to: "+str(resizedWidth)+" x "+str(resizedHeight))
        resizedImage = cv2.resize(image, (resizedWidth, resizedHeight), interpolation=cv2.INTER_AREA)
        image = resizedImage
        outPath = imagePath[0:len(imagePath)-len(fileName)]
        writeImage(outPath+fileName, image)

    # showImage("image", image)

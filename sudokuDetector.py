# File        :   sudokuDetector.py
# Version     :   1.0.0
# Description :   Main sudoku detector
# Date:       :   May 01, 2022
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

import numpy as np
import cv2
import math
from datetime import date, datetime
import keyboard
import os
import skimage.exposure


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
    imagePath = imagePath + ".png"
    cv2.imwrite(imagePath, inputImage, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print("Wrote Image: " + imagePath)


# Produces out string for image writing:
def getOutString(currentDate):
    # Format date
    currentDate = currentDate.strftime("%b-%d-%Y")

    # Get time:
    currentTime = datetime.now()
    currentTime = currentTime.strftime("%H:%M:%S")

    # Drop them nasty ":s":
    dateString = "_" + currentTime[0:2] + "-" + currentTime[3:5] + "-" + currentTime[6:8]
    print("Current Time: " + currentTime + " Date String: " + dateString)
    dateString = currentDate + dateString

    return dateString


# Orders the 4-points into tl, tr, bt, bl
def orderPoints(pts):
    # Prepare formatted corners:
    formattedCorners = np.zeros((4, 2), dtype="float32")

    # Format points:
    for p in range(len(pts)):
        currentPoint = pts[p][0]

        formattedCorners[p][0] = currentPoint[0]
        formattedCorners[p][1] = currentPoint[1]

        print(formattedCorners[p])

    # print(pts)
    # print(formattedCorners)

    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = formattedCorners.sum(axis=1)
    rect[0] = formattedCorners[np.argmin(s)]
    rect[2] = formattedCorners[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(formattedCorners, axis=1)
    rect[1] = formattedCorners[np.argmin(diff)]
    rect[3] = formattedCorners[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def rectifyImage(inputImage, corners):
    (tl, tr, br, bl) = corners
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates

    xda = (br[0] - bl[0]) ** 2
    yda = (br[1] - bl[1]) ** 2

    widthA = np.sqrt(xda + yda)

    xdb = (tr[0] - tl[0]) ** 2
    ydb = (tr[1] - tl[1]) ** 2

    widthB = np.sqrt(xdb + ydb)

    # widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    # widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = int(max(widthA, widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates

    xda = (bl[0] - tl[0]) ** 2
    yda = (bl[1] - tl[1]) ** 2

    heightA = np.sqrt(xda + yda)

    xdb = (br[0] - tr[0]) ** 2
    ydb = (br[1] - tr[1]) ** 2

    heightB = np.sqrt(xdb + ydb)

    # heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    # heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(corners, dst)
    outImage = cv2.warpPerspective(inputImage, M, (maxWidth, maxHeight))

    return outImage


def getHullImage(contour, imageSize):
    # Get image size:
    (height, width) = imageSize

    # Get the convex hull for the target contour:
    hull = cv2.convexHull(contour)

    # (Optional) Draw the hull:
    # color = (0, 0, 255)
    # cv2.polylines(resizedImage, [hull], True, color, 2)
    # showImage("Hull", resizedImage)

    # Create image for good features to track:
    # (height, width) = inputImage.shape[:2]
    # Black image same size as original input:
    outImage = np.zeros((height, width), dtype=np.uint8)

    # Draw the points:
    cv2.drawContours(outImage, [hull], 0, 255, 2)
    # showImage("hullImg", outImage)

    return outImage


# Fill 4 corners:
def fillCorners(binaryImage, offset=0, fillColor=(0, 0, 0)):
    # Get input's height and width:
    (h, w) = binaryImage.shape[:2]
    floodCorners = ((offset, offset),
                    (w - offset, offset),
                    (offset, h - offset),
                    (w - offset, h - offset))
    for c in range(len(floodCorners)):
        currentCorner = floodCorners[c]
        print("print at - x: " + str(currentCorner[0]) + ", y: " + str(currentCorner[1]))
        cv2.floodFill(binaryImage, None, currentCorner, fillColor)
        # showImage("fillCorners", binaryImage)

    return binaryImage


# Add border:
def addBorder(binaryImage, start=(0, 0,), borderColor=(0, 0, 0), borderThickness=5):
    # Apply black border:
    (h, w) = binaryImage.shape[:2]
    pt1 = start
    pt2 = (w, h)
    cv2.rectangle(binaryImage, pt1, pt2, borderColor, borderThickness)
    # showImage("binaryImage [Border]", binaryImage)

    return binaryImage


# Resize image:
def resizeImage(inputImage, resizePercent):
    # Resize at a fixed scale:
    (h, w) = inputImage.shape[:2]
    resizedWidth = int(w * resizePercent / 100)
    resizedHeight = int(h * resizePercent / 100)

    # Resize image
    resizedImage = cv2.resize(inputImage, (resizedWidth, resizedHeight), interpolation=cv2.INTER_LINEAR)

    return resizedImage


# Pre-process sample:
def preProcessSample(tempImage, newSize):
    # Set new size:
    (newWidth, newHeight) = newSize

    # Get h, w:
    (h, w) = tempImage.shape[:2]

    # Get aspect ratio:
    aRatio = w / h

    print("Image Size - H: " + str(h) + ", W: " + str(w) + " r: " + str(aRatio))

    # Minimum required size to add 2 borders
    # in vertical dimension:
    baseHeight = newHeight - 2

    # Minimum required size to add 2 borders
    # in horizontal dimension:
    baseWidth = newWidth - 2

    if h != newHeight or w != newWidth:
        # compute new temp width:
        tempHeight = int(baseHeight)
        tempWidth = math.floor(aRatio * baseHeight)
        newSize = (tempWidth, tempHeight)
        tempImage = cv2.resize(tempImage, newSize, interpolation=cv2.INTER_NEAREST)
        (w, h) = newSize
        if verbose:
            showImage("Resized Temp", tempImage)

    print("Image [Pre borders] H: " + str(h) + ", W: " + str(w))

    # Compute borders:
    left = math.floor(0.5 * (newWidth - w))
    right = left

    top = math.floor(0.5 * (newHeight - h))
    bottom = top

    print((left, right, top, bottom))

    if w + 2 * left != newWidth:
        left = left + 1

    # Apply borders:
    borderType = cv2.BORDER_CONSTANT
    borderColor = (0, 0, 0)
    outImage = cv2.copyMakeBorder(tempImage, top, bottom, left, right, borderType, None, borderColor)

    (h, w) = outImage.shape[:2]

    print("Image [Post borders] H: " + str(h) + ", W: " + str(w))

    return outImage


# Set the resources paths:
path = "D://opencvImages//sudoku//"
fileNames = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14"]

# Debug Options
verbose = True
showAugmented = True

# Sample dimensions:
numbersPerRow = 9
cellHeight = 28
cellWidth = 28

# Out options:
writeSamples = False
outDir = "samples//test//"
outPath = path + outDir

# The class dictionary:
classDictionary = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5", 5: "6", 6: "7", 7: "8", 8: "9"}

# Model dir:
modelDir = path + "model//"
modelName = "svmModel-01.xml"
svmLoaded = False

# The SVM was not previously loaded:
print("Loading SVM for the first time.")

# Load XML from path:
modelPath = modelDir + modelName
print("loadSvm>> Loading SVM model: " + modelPath)

# Load SVM:
SVM = cv2.ml.SVM_load(modelPath)

# Check if SVM is ready to classify:
svmTrained = SVM.isTrained()

# Some temp matrices:
inpaintedBoard = np.zeros((1, 1, 3), dtype="uint8")

# Store the board approximation here:
boardCorners = None
mainCorners = None

resizedHeight = 0
resizedWidth = 0

if svmTrained:
    print("loadSVM>> SVM loaded, trained and ready to test")

for f in range(len(fileNames)):

    # Create puzzle board:
    puzzleMatrix = np.zeros((9, 9, 1), dtype="uint8")

    # Get image name:
    fileName = fileNames[f] + ".png"
    print("Processing Image: " + fileName)

    inputImage = readImage(path + fileName)
    # inputImageCopy = inputImage.copy()

    showImage("Input Image", inputImage)

    # Get image dimensions
    (originalImageHeight, originalImageWidth) = inputImage.shape[:2]
    print("W: " + str(originalImageWidth) + ", H: " + str(originalImageHeight))

    # Resize at a fixed scale:
    resizedImage = resizeImage(inputImage, resizePercent=30)
    resizedInput = resizedImage.copy()
    (resizedHeight, resizedWidth) = resizedImage.shape[:2]
    if verbose:
        showImage("resizedImage", resizedImage)

    writeImage(path+"resizedImage", resizedImage)

    # To Gray
    grayscaleImage = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2GRAY)
    grayscaleImageCopy = grayscaleImage.copy()
    if verbose:
        showImage("grayscaleImage", grayscaleImage)

    # Gaussian Filter:
    sigma = (5, 5)
    grayscaleImage = cv2.GaussianBlur(grayscaleImage, sigma, 0)
    if verbose:
        showImage("grayscaleImage [Blurred]", grayscaleImage)

    # Adaptive Threshold:
    windowSize = 41
    constantValue = 10
    binaryImage = cv2.adaptiveThreshold(grayscaleImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                        windowSize, constantValue)
    # Show the result:
    if verbose:
        showImage("binaryImage [Adaptive - Gaussian]", binaryImage)

    # Apply some morphology:
    kernelSize = (3, 3)
    morphoKernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
    binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_OPEN, morphoKernel, iterations=7)
    if verbose:
        showImage("binaryImage [Morphed]", binaryImage)

    # Apply black border:
    binaryImage = addBorder(binaryImage, start=(0, 0), borderColor=(0, 0, 0), borderThickness=5)
    if verbose:
        showImage("binaryImage [Border]", binaryImage)

    # Flood fill at top left corner:
    leftCorner = (0, 0)
    fillColor = (255, 255, 255)
    cv2.floodFill(binaryImage, None, leftCorner, fillColor)
    if verbose:
        showImage("binaryImage [Filled]", binaryImage)

    # Binary to BGR:
    binaryColor = cv2.cvtColor(binaryImage, cv2.COLOR_GRAY2BGR)

    # Get contours:
    contours, hierarchy = cv2.findContours(
        binaryImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Total Contours:
    totalContours = len(contours)
    print("Total Contours: " + str(totalContours))

    # Store interesting contours here:
    targetContours = []

    # First contour filter:
    for i in range(totalContours):

        # Get contour:
        c = contours[i]

        # Get hierarchy
        h = hierarchy[0][i]

        # Get bounding rectangle:
        boundRect = cv2.boundingRect(c)

        # Get the bounding rect data:
        rectX = int(boundRect[0])
        rectY = int(boundRect[1])
        rectWidth = int(boundRect[2])
        rectHeight = int(boundRect[3])

        # Aspect ratio:
        aspectRatio = rectWidth / rectHeight

        # Get area:
        blobArea = cv2.contourArea(c)

        print("c: " + str(i) + " area: " + str(blobArea) + " aspect: " + str(aspectRatio))
        # print(h)

        # Area thresholds:
        maxArea = 500000
        minArea = 150000

        # Hierarchy filter:
        #  0     1         2            3
        # [Next, Previous, First_Child, Parent]
        if h[3] == 0:
            # if h[2] != -1 and h[3] != -1:
            if blobArea > minArea and blobArea < maxArea:
                # The target contour:
                color = (0, 255, 0)
                targetContours.append(c)
            else:
                color = (255, 255, 0)
        else:
            # Everything else:
            color = (0, 0, 255)

        # Draw contour:
        cv2.drawContours(binaryColor, [c], 0, color, 3)
        if verbose:
            showImage("Raw Contours", binaryColor)

    # rectified images here:
    rectifiedImages = []

    # Process filtered contours:
    for i in range(len(targetContours)):

        # Get the contour:
        c = targetContours[i]

        # Get Hull:
        (height, width) = binaryImage.shape[:2]
        hullImg = getHullImage(c, (height, width))

        # Set the corner detection:
        maxCorners = 4
        qualityLevel = 0.01
        minDistance = int(max(height, width) / maxCorners)

        # Store the corners:
        cornerList = []

        # Get the corners:
        corners = cv2.goodFeaturesToTrack(hullImg, maxCorners, qualityLevel, minDistance)
        corners = np.int0(corners)

        # Order corners
        # top-left, top-right, bottom-right, and bottom-left order.
        corners = orderPoints(corners)
        mainCorners = corners.copy()
        # print(corners)

        # Loop through the corner array and store/draw the corners:
        for c in range(len(corners)):
            # Get current corner:
            currentCorner = corners[c]

            # Get x, y:
            x = int(currentCorner[0])
            y = int(currentCorner[1])

            # (Optional) Draw the corner points:
            cv2.circle(resizedImage, (x, y), 5, 255, 5)
            if verbose:
                showImage("corners", resizedImage)

            writeImage(path + "corners01", resizedImage)

        # Apply unwarping:
        rectifiedImage = rectifyImage(grayscaleImageCopy, corners)
        if verbose:
            showImage("rectifiedImage", rectifiedImage)

        # Into the list:
        rectifiedImages.append(rectifiedImage)

    # Store just the numbers here:
    numbersMasks = []

    # Process rectified images:
    for r in range(len(rectifiedImages)):

        # Get rectified image:
        currentImage = rectifiedImages[r]

        # Deep Copy:
        currentImageCopy = currentImage.copy()

        # Color version:
        currentImageColor = cv2.cvtColor(currentImage, cv2.COLOR_GRAY2BGR)

        # Gaussian Filter:
        sigma = (5, 5)
        currentImage = cv2.GaussianBlur(currentImage, sigma, 0)
        if verbose:
            showImage("currentImage [Blurred]", currentImage)

        # Adaptive Thresh:
        windowSize = 31
        constantValue = 10
        binaryCrop = cv2.adaptiveThreshold(currentImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                           windowSize, constantValue)
        if verbose:
            showImage("binaryCrop", binaryCrop)

        # Fill corners:
        binaryCrop = fillCorners(binaryCrop, 5, (0, 0, 0))
        if verbose:
            showImage("binaryCrop [Corners Filled]", binaryCrop)

        # Apply some morphology:
        kernelSize = (5, 5)
        morphoKernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
        boardMask = cv2.morphologyEx(binaryCrop, cv2.MORPH_DILATE, morphoKernel, iterations=3)
        if verbose:
            showImage("boardMask [Dilate 1]", boardMask)

        boardMask = cv2.morphologyEx(boardMask, cv2.MORPH_ERODE, morphoKernel, iterations=3)
        if verbose:
            showImage("boardMask [Erode 2]", boardMask)

        # Get contours:
        contours, _ = cv2.findContours(boardMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:

            # Draw contour:
            color = (0, 0, 255)
            cv2.drawContours(currentImageColor, [c], 0, color, 3)
            if verbose:
                showImage("Board Contours", currentImageColor)

            # Get area:
            blobArea = cv2.contourArea(c)

            # Get bounding rectangle:
            boundRect = cv2.boundingRect(c)

            # Get the bounding rect data:
            rectX = int(boundRect[0])
            rectY = int(boundRect[1])
            rectWidth = int(boundRect[2])
            rectHeight = int(boundRect[3])

            # Aspect ratio:
            aspectRatio = rectWidth / rectHeight

            print((blobArea, aspectRatio))

            # Blob thresholds:
            minArea = 100000
            maxArea = 3 * minArea

            minAspectRatio = 0.60
            maxAspectRatio = 1.5
            epsilon = 0.5

            if blobArea >= minArea and blobArea < maxArea:

                print("Blob area: OK")

                # Aspect Ratio difference:
                diff = abs(minAspectRatio - aspectRatio)

                # if diff <= epsilon:
                if aspectRatio >= minAspectRatio and aspectRatio < maxAspectRatio:

                    print("Blob Aspect: OK")

                    # approximate the contour
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.05 * peri, True)

                    print(len(approx))

                    # if our approximated contour has four points, then
                    # we can assume that we have found our screen

                    if len(approx) == 4:

                        boardCorners = orderPoints(approx)

                        # Loop through the corner array and store/draw the corners:
                        for p in range(len(boardCorners)):
                            # Get current corner:
                            currentCorner = boardCorners[p]

                            # Get x, y:
                            x = int(currentCorner[0])
                            y = int(currentCorner[1])

                            # (Optional) Draw the corner points:
                            cv2.circle(currentImageColor, (x, y), 5, 255, 5)
                            if verbose:
                                showImage("Board Corners", currentImageColor)

                            writeImage(path + "corners02", currentImageColor)

                        # Apply unwarping:
                        rectifiedImage = rectifyImage(currentImageCopy, boardCorners)
                        if verbose:
                            showImage("rectifiedImage [Board]", rectifiedImage)

                        # Apply Otsu:
                        _, boardMask = cv2.threshold(rectifiedImage, 0, 255, cv2.THRESH_OTSU)
                        if verbose:
                            showImage("rectifiedImage [Binary]", boardMask)

                        # Apply black border:
                        boardMask = addBorder(boardMask, start=(0, 0), borderColor=(0, 0, 0), borderThickness=5)
                        if verbose:
                            showImage("boardMask [Border]", boardMask)

                        # Flood-fill at TL:
                        leftCorner = (0, 0)
                        fillColor = (255, 255, 255)
                        cv2.floodFill(boardMask, None, leftCorner, fillColor)
                        if verbose:
                            showImage("boardMask [Filled]", boardMask)

                        # Invert image:
                        boardMask = 255 - boardMask
                        if verbose:
                            showImage("boardMask [Inverted]", boardMask)

                        writeImage(path + "boardMask1", boardMask)

                        # Into the list:
                        numbersMasks.append(boardMask)

                        # Prepare inpaint mask:
                        kernelSize = (3, 3)
                        morphoKernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
                        inpaintMask = cv2.morphologyEx(boardMask, cv2.MORPH_DILATE, morphoKernel, iterations=2)
                        inpaintedBoard = cv2.inpaint(rectifiedImage, inpaintMask, 3, cv2.INPAINT_TELEA)

                        # Gray 2 BGR:
                        inpaintedBoard = cv2.cvtColor(inpaintedBoard, cv2.COLOR_GRAY2BGR)

                        # The "clean" board:
                        if verbose:
                            showImage("inpaintedBoard", inpaintedBoard)

    # Store numbers here:
    numbersCrops = []

    # Extract number blobs:
    for m in range(len(numbersMasks)):
        # Get current Mask:
        numbersMask = numbersMasks[m]
        # Color:
        maskColor = cv2.cvtColor(numbersMask, cv2.COLOR_GRAY2BGR)

        # Extract contours:
        contours, _ = cv2.findContours(numbersMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blobCounter = 0

        for c in contours:
            # Get the bounding rectangle:
            boundRect = cv2.boundingRect(c)

            # Get the dimensions of the bounding rect:
            rectX = int(boundRect[0])
            rectY = int(boundRect[1])
            rectWidth = int(boundRect[2])
            rectHeight = int(boundRect[3])

            # Centroid:
            bx = int(rectX + 0.5 * rectWidth)
            by = int(rectY + 0.5 * rectHeight)

            # Get area:
            blobArea = cv2.contourArea(c)

            # Aspect ratio:
            aspectRatio = rectWidth / rectHeight

            print((blobArea, aspectRatio))

            minArea = 35

            if blobArea >= minArea:
                # Set color and draw:
                color = (0, 255, 0)
                cv2.rectangle(maskColor, (rectX, rectY), (rectX + rectWidth, rectY + rectHeight), color, 2)

                # Draw the centers:
                color = (0, 0, 255)  # Red
                radius = 1
                center = (bx, by)
                cv2.circle(maskColor, center, radius, color, 3)
                if verbose:
                    showImage("Numbers Rects", maskColor)
                writeImage(path + "numberRects", maskColor)

                # Crop:
                currentNumber = numbersMask[rectY:rectY + rectHeight, rectX:rectX + rectWidth]
                if verbose:
                    showImage("currentNumber", currentNumber)

                # Store the image and its centroid on the mask:
                bx = int(bx - 0.5 * rectWidth)
                by = int(by + 0.5 * rectHeight)
                bcenter = (bx, by)
                numbersCrops.append((currentNumber, center, bcenter))

                # Get size:
                (h, w) = currentNumber.shape[:2]
                print("Blob: " + str(blobCounter) + " w: " + str(w) + ", h: " + str(h))

                blobCounter += 1

    # Send to SVM:
    for s in range(len(numbersCrops)):

        # Get Sample:
        currentSample = numbersCrops[s][0]
        if verbose:
            showImage("Current Sample", currentSample)

        # Get sample centroid:
        blobCenter = numbersCrops[s][1]
        blobTopLeft = numbersCrops[s][2]

        # Pre-process sample
        cellSize = (cellWidth, cellHeight)
        # writeImage(path + "samples//rawSamplePre", currentSample)

        currentSample = preProcessSample(currentSample, cellSize)
        # writeImage(path + "samples//rawSamplePost", currentSample)

        if verbose:
            showImage("Current Sample [Pre]", currentSample)
        writeImage(path + "svmSample", currentSample)

        # Reshape the image into a plain vector:
        # Convert data type to float 32
        testImage = currentSample.reshape(-1, cellWidth * cellHeight).astype(np.float32)

        # Classify "image" (vector)
        svmResult = SVM.predict(testImage)[1]
        svmResult = svmResult[0][0]
        print("svmResult: " + str(svmResult))

        # Get character from dictionary:
        svmLabel = classDictionary[svmResult]
        print("SVM says: " + svmLabel)

        # Draw label on image:
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 0, 255)
        fontScale = 1
        fontThickness = 2
        center = blobCenter

        cv2.putText(maskColor, svmLabel, center, font, fontScale, color, fontThickness)
        if verbose:
            showImage("Numbers Rects", maskColor)

        # Gray 2 BGR:
        currentSample = cv2.cvtColor(currentSample, cv2.COLOR_GRAY2BGR)

        # Re-size image for displaying results:
        (imageHeight, imageWidth) = currentSample.shape[:2]
        aspectRatio = imageHeight / imageWidth
        rescaledWidth = 300
        rescaledHeight = int(rescaledWidth * aspectRatio)
        newSize = (rescaledWidth, rescaledHeight)

        currentSample = cv2.resize(currentSample, newSize, interpolation=cv2.INTER_NEAREST)

        # Set text parameters:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(currentSample, "SVM: " + str(svmLabel), (3, 50), font, 0.5, (255, 0, 255), 1, cv2.LINE_8)

        # Show the classified image:
        if verbose:
            showImage("SVM Result", currentSample)

        # Write sample?
        if writeSamples:
            print("Press key to save sample..")

            # Get pressed key:
            ch = keyboard.read_key()

            # Get sample dir:
            tempPath = outPath + ch
            print("Sample Outpath: " + str(tempPath))
            sampleDir = os.listdir(tempPath)

            # Get number of files:
            filesNumber = len(sampleDir)
            filesNumber = filesNumber + 1

            sampleName = tempPath + "//s_" + str(filesNumber)
            print("Writing sample: " + sampleName)

            # get date:
            # currentDate = date.today()
            # sampleName = str(s) + "-" + getOutString(currentDate)
            writeImage(sampleName, currentSample)

        # Paste on inpainted board:
        (boardHeight, boardWidth) = inpaintedBoard.shape[:2]
        (cx, cy) = blobCenter
        # Set cell dimensions:
        # numbersPerRow = 9
        # cellHeight = boardHeight // numbersPerRow
        # cellWidth = boardWidth // numbersPerRow

        # Set matrix indices:
        numberCellWidth = int(boardWidth / numbersPerRow)
        numberCellHeight = int(boardHeight / numbersPerRow)
        colIndex = cx // numberCellWidth
        rowIndex = cy // numberCellHeight

        puzzleMatrix[rowIndex][colIndex] = int(svmResult + 1)

        # Set center:
        (tx, ty) = blobTopLeft
        print((tx, ty))
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 0, 255)
        fontScale = 1
        fontThickness = 2
        cv2.putText(inpaintedBoard, str(svmLabel), (tx, ty), font, fontScale, color, fontThickness)
        if showAugmented:
            showImage("Augmented Board", inpaintedBoard)
        writeImage(path + "augmentedBoard", inpaintedBoard)

    # Print puzzle matrix:
    print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in puzzleMatrix]))

    # Warp board:
    (bh, bw) = inpaintedBoard.shape[:2]
    (th, tw) = rectifiedImages[0].shape[:2]
    warpMask = np.ones((bh, bw, 3), dtype="uint8") * 255
    if verbose:
        showImage("warpMask 1", warpMask)

    inputPoints = np.float32([[0, 0], [boardWidth, 0], [boardWidth, boardHeight], [0, boardHeight]])
    h = cv2.getPerspectiveTransform(inputPoints, boardCorners)
    warpedBoard = cv2.warpPerspective(inpaintedBoard, h, (tw, th), borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(0, 0, 0))
    if verbose:
        showImage("warpedBoard (1)", warpedBoard)

    warpMask = cv2.warpPerspective(warpMask, h, (tw, th))
    if verbose:
        showImage("warpMask 2", warpMask)

    (boardHeight, boardWidth) = warpedBoard.shape[:2]
    inputPoints = np.float32([[0, 0], [boardWidth, 0], [boardWidth, boardHeight], [0, boardHeight]])
    h = cv2.getPerspectiveTransform(inputPoints, mainCorners)
    warpedBoard = cv2.warpPerspective(warpedBoard, h, (resizedWidth, resizedHeight), borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(0, 0, 0))

    if verbose:
        showImage("warpedBoard (2)", warpedBoard)

    warpMask = cv2.warpPerspective(warpMask, h, (resizedWidth, resizedHeight))
    if verbose:
        showImage("warpMask 3", warpMask)

    # anti-alias mask
    warpMask = cv2.GaussianBlur(warpMask, (0, 0), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_DEFAULT)
    warpMask = skimage.exposure.rescale_intensity(warpMask, in_range=(0, 128), out_range=(0, 255))

    # convert mask to float in range 0 to 1
    warpMask = warpMask.astype(np.float64) / 255

    # composite warped image over base and convert back to uint8
    result = (warpedBoard * warpMask + resizedInput * (1 - warpMask))
    result = result.clip(0, 255).astype(np.uint8)

    if verbose:
        showImage("processed warpMask", warpMask)

    showImage("Output Image", result)
    writeImage(path + "augmentedBoardFinal", result)

    # Time to let it all go:
    cv2.destroyAllWindows()

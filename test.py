import cv2 as cv
import cv2 as cv2
import numpy as np
import argparse
import random
import os
from PIL import Image
import imagehash

random.seed(12345)
eps = 1e-1
symbol_cnt = 0
symbols_list = []
symbols_label = ["shark", "Woman", "N", "Boat", "Spear", "Man", "WaterBuilding", "Dog", "Skeleton", "Compass", "U", "Anchor", "B", "Mine", "Mine", "Mine", "S", "Mine", "Skeleton", "Compass", "Compass", "Boat", "U", "Boat", "Boat", "U", "B", "O", "Spear", "Anchor", "Anchor", "Anchor", "Anchor", "Anchor", "Anchor", "Gold", "Shark", "Dog", "Shark", "Skeleton", "Anchor", "Dog", "Boat", "Anchor", "Dog", "Spear", "Anchor", "Skeleton", "Skeleton", "Woman", "Boat", "Man", "Boat", "Anchor", "Woman"]

def getHashDiff(image1, image2):
    # Convert cv2Img from OpenCV format to PIL format
    im1 = cv2.resize(image1, (128, 128))
    im2 = cv2.resize(image2, (128, 128))
    pilImg1 = Image.fromarray(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB))
    pilImg2 = Image.fromarray(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))
    
    # Get the average hashes of both images
    hash0 = imagehash.average_hash(pilImg1)
    hash1 = imagehash.average_hash(pilImg2)
    
    hashDiff = hash0 - hash1  # Finds the distance between the hashes of images
    return hashDiff


def getMSE(image1, image2):
    image1 = cv2.resize(image1, (128, 128))
    image2 = cv2.resize(image2, (128, 128))

    img1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1 = cv2.GaussianBlur(src = img1, ksize = (5, 5), sigmaX = 0)

    img2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2 = cv2.GaussianBlur(src = img2, ksize=(5, 5), sigmaX = 0)

    diff = cv2.absdiff(src1 = img1, src2 = img2)
    kernel = np.ones((5, 5))
    diff = cv2.dilate(diff, kernel, 1)
            
    thresh_frame = cv2.threshold(src = diff, thresh = 20, maxval = 255, type = cv2.THRESH_BINARY)[1]
    # Convert cv2Img from OpenCV format to PIL format
    h, w = thresh_frame.shape
    err = np.sum(thresh_frame**2)
    mse = err/(float(h * w)) * 100
    return mse

def thresh_callback(src, val):
    threshold = val
    original_image = src.copy()
    canny_output = cv.Canny(original_image, threshold, threshold * 5)
    # kernel = np.ones((5, 5))
    # canny_output = cv2.dilate(canny_output, kernel, 1)
    # thresh_frame = cv2.threshold(src=canny_output, thresh=20, maxval=255, type=cv2.THRESH_BINARY)[1]
    contours, _ = cv.findContours(canny_output, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    # centers = [None]*len(contours)
    # radius = [None]*len(contours)
    for i, c in enumerate(contours):
        peri = cv2.arcLength(c, True)
        contours_poly[i] = cv2.approxPolyDP(c, 0.02 * peri, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])
        # centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])
    
    global symbol_cnt
    global symbols_list

    # os.system('cls')

    for i in range(len(contours)):
        color = (255, 255, 255)
        # cv.rectangle(src, (int(boundRect[i][0]), int(boundRect[i][1])), \
        #     (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
        # # if boundRect[i][3] < 486 and boundRect[i][3] > 450 and boundRect[i][3] / boundRect[i][2] > 2.1 + eps and boundRect[i][3] / boundRect[i][2] < 2.7 - eps:
        #     print(f'height is {boundRect[i][3]}. ratio is {boundRect[i][3] / boundRect[i][2]}.')
        #     cv.rectangle(src, (int(boundRect[i][0]), int(boundRect[i][1])), \
        #     (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
        # if (cv2.contourArea(contours[i]) < 5000) or (cv2.contourArea(contours[i]) > 25000):
        #     # too small and too large: skip!
        #     continue
        # print(f'contour Area is {cv2.contourArea(contours[i])}.')
        # cv2.drawContours(image=src, contours=contours[i], contourIdx=-1, color=color, thickness=2, lineType=cv2.LINE_AA)
        # cv.rectangle(src, (int(boundRect[i][0]), int(boundRect[i][1])), \
        #     (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
        
        if (boundRect[i][2] / boundRect[i][3] > 0.8 and boundRect[i][2] / boundRect[i][3] < 1.2 and boundRect[i][2] > 120 and boundRect[i][2] < 145) or (boundRect[i][2] / boundRect[i][3] > 1.4 and boundRect[i][2] / boundRect[i][3] < 1.6 and boundRect[i][3] > 75 and boundRect[i][3] < 90):
            cv.rectangle(src, (int(boundRect[i][0]), int(boundRect[i][1])), \
            (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
            symbol = original_image[int(boundRect[i][1]):int(boundRect[i][1]+boundRect[i][3]), int(boundRect[i][0]):int(boundRect[i][0]+boundRect[i][2])]
            temporary_list = symbols_list.copy()
            idx = 0
            if len(symbols_list) < 1:
                temporary_list.append(symbol)
                symbol_cnt += 1
                cv.imwrite(f'./symbols/symbol{symbol_cnt}.jpg', symbol)
            else:
                flg = False
                for idx, ele in enumerate(symbols_list):
                    if getHashDiff(ele, symbol) < 20:
                        print(f'HashDiff is {getHashDiff(ele, symbol)}')
                        flg = True
                        break
                if not flg:
                    temporary_list.append(symbol)
                    symbol_cnt += 1
                    idx = len(symbols_list)
                    cv.imwrite(f'./symbols/symbol{symbol_cnt}.jpg', symbol)
            symbols_list = temporary_list.copy()

            endl = '\n'
            cv2.putText(src, f'"{symbols_label[idx]}"', (int(boundRect[i][0]) + 10, int(boundRect[i][1]) + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
    
    cv.imshow('Contours', src)


try:
      
    # creating a folder named data
    if not os.path.exists('symbols'):
        os.makedirs('symbols')
  
# if not created then raise error
except OSError:
    print ('Error: Creating directory of symbols')

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('input.mp4')
  
# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video file")
  
# Read until video is completed

flg_First = True
flg_find_contour = True

cutoff = 0.1

while(cap.isOpened()):
      
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if ret == True:

        #select ROI in frame
        # roi = cv2.selectROI(frame)
        # # print rectangle points of selected roi
        # print(roi)

        #Crop small part for movement detection
            
        movement_cropped = frame[370:515, 994:1145]
        # movement_cropped = frame[43:555, 122:1187]
        
        if flg_First:
            old_movement_cropped = movement_cropped
            flg_First = False
            continue
        # print(getHashDiff(old_movement_cropped, movement_cropped))

        #Crop selected roi from raw image
        roi_cropped = frame[43:555, 122:1187]
        cv2.imshow("ROI", roi_cropped)
        # Convert image to gray and blur it
        src_gray = cv.cvtColor(roi_cropped, cv.COLOR_BGR2GRAY)
        src_gray = cv.blur(src_gray, (2,2))
        # Create Window
        
        max_thresh = 255
        thresh = 40 # initial threshold
        # cv.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
        print(f'MSE is {getMSE(old_movement_cropped, movement_cropped)}')
        # if getMSE(old_movement_cropped, movement_cropped) < eps / 2 and flg_find_contour:
        #     flg_find_contour = False
        #     thresh_callback(src_gray, thresh)
        # else:
        #     flg_find_contour = True

        if getMSE(old_movement_cropped, movement_cropped) < eps / 2:
            thresh_callback(src_gray, thresh)
        
        old_movement_cropped = movement_cropped  
        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    # Break the loop
    else:
        break
  
# When everything done, release
# the video capture object
cap.release()
  
# Closes all the frames
cv2.destroyAllWindows()

import cv2 as cv2
import numpy as np
import os
from PIL import Image
import math
from skimage.morphology import skeletonize
# from easyocr import Reader
import pytesseract


eps = 1e-1
bet = 0.0
win = 0.0
credit = 0.0

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

def getValues(src, winName):
    original_image = src.copy()
    # src = findSkeleton(src)
    h, w = src.shape    
    # src = cv2.resize(src, (int(w / 4), int(h / 4)), interpolation = cv2.INTER_LINEAR)
    # h, w = src.shape
    
    print(f'Mean is {np.mean(src)}, Variance is {np.std(src)}')
    src = cv2.threshold(src=src, thresh=int(np.mean(src) + np.std(src)), maxval=255, type=cv2.THRESH_BINARY)[1]
    
    prepared_image = np.zeros((w, w), dtype = np.uint8)
    prepared_image[0:h, 0:w] = src.copy()
    # for i in range(h):
    #     for j in range(w):
    #         prepared_image[i][j] = src[i][j]
    print(prepared_image.shape)

    print(f'width = {w} and height = {h}')
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    
    prepared_image = skeletonize(prepared_image, method = 'lee')
    # cv2.imshow("Skeletonization", prepared_image)
    # prepared_image = findSkeleton(prepared_image)
    # prepared_image = cv2.resize(prepared_image, (int(w * 4), int(h * 4)), interpolation = cv2.INTER_LINEAR)
    src = Image.fromarray(prepared_image)
    # reader = Reader(gpu = False)
    # text = reader.readtext(prepared_image)
    text = pytesseract.image_to_string(src)
    print(f'{winName} value is {text}')
    cv2.putText(prepared_image, text, (0, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow(winName, prepared_image)
    
        

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

        # #select ROI in frame
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
        bet_cropped = frame[633:707, 285:445]
        win_cropped = frame[589:693, 576:803]
        credit_cropped = frame[607:697, 887:1065]
        cv2.imshow("Frame", frame)
        # Convert image to gray and blur it
        src_bet = cv2.cvtColor(bet_cropped, cv2.COLOR_BGR2GRAY)
        src_bet = cv2.blur(src_bet, (3,3))
        
        # Convert image to gray and blur it
        src_win = cv2.cvtColor(win_cropped, cv2.COLOR_BGR2GRAY)
        src_win = cv2.blur(src_win, (3,3))
        
        # Convert image to gray and blur it
        src_credit = cv2.cvtColor(credit_cropped, cv2.COLOR_BGR2GRAY)
        src_credit = cv2.blur(src_credit, (3,3))
        
        # Create Window
        max_thresh = 255
        thresh = 40 # initial threshold
        # cv2.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
        # print(f'MSE is {getMSE(old_movement_cropped, movement_cropped)}')
        # if getMSE(old_movement_cropped, movement_cropped) < eps / 2 and flg_find_contour:
        #     flg_find_contour = False
        #     thresh_callback(src_gray, thresh)
        # else:
        #     flg_find_contour = True

        if getMSE(old_movement_cropped, movement_cropped) < eps / 2:
            getValues(src_bet, "bet")
            getValues(src_win, "Win")
            getValues(src_credit, "Credit")
        

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

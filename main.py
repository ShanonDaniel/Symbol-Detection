# importing libraries
import cv2
import numpy as np
import random

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('input.mp4')
  
# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video file")
  
# Read until video is completed
while(cap.isOpened()):
      
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        # Display the resulting frame
        # cv2.imshow('Frame', frame)
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        cv2.imshow('Frame', img)
        canny_output = cv2.Canny(img, 70, 200)
        # kernel = np.ones((5, 5))
        # canny_output = cv2.dilate(canny_output, kernel, 1)
        # thresh_frame = cv2.threshold(src=canny_output, thresh=20, maxval=255, type=cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(canny_output, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
        contours_poly = [None]*len(contours)
        boundRect = [None]*len(contours)
        
        for i, c in enumerate(contours):
            peri = cv2.arcLength(c, True)
            contours_poly[i] = cv2.approxPolyDP(c, 0.02 * peri, True)
            boundRect[i] = cv2.boundingRect(contours_poly[i])
        
        for i in range(len(contours)):
            color = (255, 255, 255)
            # cv2.rectangle(img, (int(boundRect[i][0]), int(boundRect[i][1])), \
            #     (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
            if boundRect[i][3] < 486 and boundRect[i][3] > 450 and boundRect[i][3] / boundRect[i][2] > 2.1 and boundRect[i][3] / boundRect[i][2] < 2.7:
                print(f'height is {boundRect[i][3]}. ratio is {boundRect[i][3] / boundRect[i][2]}.')
                # cv2.rectangle(img, (int(boundRect[i][0]), int(boundRect[i][1])), \
                # (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
                cv2.drawContours(image=img, contours=contours[i], contourIdx=-1, color=color, thickness=2, lineType=cv2.LINE_AA)
                print(f"points of contour[{i}] are {len(contours_poly[i])}")
            # cv2.drawContours(image=img, contours=contours[i], contourIdx=-1, color=color, thickness=2, lineType=cv2.LINE_AA)
            # cv.rectangle(src, (int(boundRect[i][0]), int(boundRect[i][1])), \
            #     (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
            
            
        
        cv2.imshow('Contours', img)
            
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
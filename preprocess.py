from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned")
args = vars(ap.parse_args())

# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
image = cv2.imread(args["image"])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)
# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)
edged = cv2.Canny(gray, 40, 200)
# show the original image and the edge detected image
print("STEP 1: Edge Detection")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)

cv2.waitKey(0)

# Display the Contours

contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
contours_poly = [None]*len(contours)
boundRect = [None]*len(contours)

for i, c in enumerate(contours):
    peri = cv2.arcLength(c, True)
    contours_poly[i] = cv2.approxPolyDP(c, 0.02 * peri, True)
    boundRect[i] = cv2.boundingRect(contours_poly[i])

for i in range(len(contours)):
    # cv2.rectangle(img, (int(boundRect[i][0]), int(boundRect[i][1])), \
    #     (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
    if boundRect[i][3] < 486 and boundRect[i][3] > 100 and boundRect[i][3] / boundRect[i][2] > 0.8 and boundRect[i][3] / boundRect[i][2] < 2.7:
        print(f'height is {boundRect[i][3]}. ratio is {boundRect[i][3] / boundRect[i][2]}.')
        # cv2.rectangle(img, (int(boundRect[i][0]), int(boundRect[i][1])), \
        # (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
        cv2.drawContours(image=image, contours=contours[i], contourIdx=-1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        print(f"points of contour[{i}] are {len(contours_poly[i])}")
        # cv2.drawContours(image=img, contours=contours[i], contourIdx=-1, color=color, thickness=2, lineType=cv2.LINE_AA)
        cv2.rectangle(image, (int(boundRect[i][0]), int(boundRect[i][1])), \
            (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color = (0, 255, 0), thickness=2)
    
    

cv2.imshow('Contours', image)
    
cv2.waitKey(0)
cv2.destroyAllWindows()
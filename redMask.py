import cv2 as cv2
import numpy as np
import os
from PIL import Image

def distributionTransform(img, middasi, std_devdasi):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.GaussianBlur(src=img, ksize=(5, 5), sigmaX=0)
	h, w = img.shape
	print(img.shape)
	mid = np.mean(img)
	std_dev = np.std(img)
	rlt = np.zeros((h, w), dtype = np.uint8)
    
	for i in range(h):
		for j in range(w):            
			rlt[i][j] = int(middasi + (img[i][j] - mid) * std_devdasi / std_dev)    
	return rlt


img1 = cv2.imread('1.jpg')
img1 = distributionTransform(img1, 120, 30)
cv2.imshow("Image 1", img1)

img2 = cv2.imread('2.jpg')
img2 = distributionTransform(img2, 120, 30)
cv2.imshow("Image 2", img2)

diff = cv2.absdiff(img1, img2)
thresh_frame = cv2.threshold(src=diff, thresh=50, maxval=255, type=cv2.THRESH_BINARY)[1]
cv2.imshow("Difference Image", diff)

cv2.waitKey(0)


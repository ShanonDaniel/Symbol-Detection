import cv2 as cv2
import numpy as np
import pytesseract
import os
from PIL import Image
import math

def distributionTransform(img, middasi, std_devdasi):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.GaussianBlur(src=img, ksize=(3, 3), sigmaX=0)
	h, w = img.shape
	mid = np.mean(img)
	std_dev = np.std(img)
	rlt = np.zeros((h, w), dtype = np.uint8)
    
	for i in range(h):
		for j in range(w):            
			rlt[i][j] = int(middasi + (img[i][j] - mid) * std_devdasi / std_dev)
	return rlt

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

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
        original = frame.copy()
        
        roi_cropped = frame[43:555, 130:1200]
        # roi_cropped = distributionTransform(roi_cropped, 120, 30)
        dst = cv2.Canny(roi_cropped, 50, 200)
    
        # Copy edges to the images that will display the results in BGR
        cdstP = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
        
        linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 80, None, 120, 5)
        
        vLines = []
        hLines = []
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                if abs(l[0] - l[2]) < 20:
                    hLines.append(l)
                    # cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
                if abs(l[1] - l[3]) < 20:
                    vLines.append(l)
                    # cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
        vLines.sort(key=lambda x: x[1])
        hLines.sort(key=lambda x: x[0])

        cv2.line(cdstP, (vLines[0][0], vLines[0][1]), (vLines[0][2], vLines[0][3]), (0,0,255), 3, cv2.LINE_AA)
        cv2.line(cdstP, (vLines[len(vLines) - 1][0], vLines[len(vLines) - 1][1]), (vLines[len(vLines) - 1][2], vLines[len(vLines) - 1][3]), (0,0,255), 3, cv2.LINE_AA)
        cv2.line(cdstP, (hLines[0][0], hLines[0][1]), (hLines[0][2], hLines[0][3]), (0,0,255), 3, cv2.LINE_AA)
        cv2.line(cdstP, (hLines[len(hLines) - 1][0], hLines[len(hLines) - 1][1]), (hLines[len(hLines) - 1][2], hLines[len(hLines) - 1][3]), (0,0,255), 3, cv2.LINE_AA)

        p1 = line_intersection(((vLines[0][0], vLines[0][1]), (vLines[0][2], vLines[0][3])), ((hLines[0][0], hLines[0][1]), (hLines[0][2], hLines[0][3])))
        p2 = line_intersection(((vLines[len(vLines) - 1][0], vLines[len(vLines) - 1][1]), (vLines[len(vLines) - 1][2], vLines[len(vLines) - 1][3])), ((hLines[0][0], hLines[0][1]), (hLines[0][2], hLines[0][3])))
        p3 = line_intersection(((vLines[len(vLines) - 1][0], vLines[len(vLines) - 1][1]), (vLines[len(vLines) - 1][2], vLines[len(vLines) - 1][3])), ((hLines[len(hLines) - 1][0], hLines[len(hLines) - 1][1]), (hLines[len(hLines) - 1][2], hLines[len(hLines) - 1][3])))
        p4 = line_intersection(((vLines[0][0], vLines[0][1]), (vLines[0][2], vLines[0][3])), ((hLines[len(hLines) - 1][0], hLines[len(hLines) - 1][1]), (hLines[len(hLines) - 1][2], hLines[len(hLines) - 1][3])))

        # print(f'{p1}, {p2}, {p3}, {p4}')

        convex = (min(p1[0], p2[0], p3[0], p4[0]), min(p1[1], p2[1], p3[1], p4[1]), max(p1[0], p2[0], p3[0], p4[0]) - min(p1[0], p2[0], p3[0], p4[0]), max(p1[1], p2[1], p3[1], p4[1]) - min(p1[1], p2[1], p3[1], p4[1]))

        pts1 = np.float32(np.array((p1, p2, p3, p4)))
        pts2 = np.float32([[convex[0],convex[1]],[convex[0],convex[1] + convex[3]],[convex[0] + convex[2],convex[1] + convex[3]], [convex[0] + convex[2],convex[1]]])
        # print(f'Conversion Points {pts1} to Points {pts2}')
        M = cv2.getPerspectiveTransform(pts1,pts2)

        # cv2.imshow("Original Image", original)
        cv2.imshow("Original", cdstP)
        rows, cols, ch = cdstP.shape
        print(f'Image size is ({cols}, {rows})')
        final_edged = cv2.warpPerspective(cdstP,M,(cols,rows))
        cv2.imshow("Final Edge", final_edged)
                
        rows, cols, ch = original.shape
        # print(f'Image size is ({cols}, {rows})')
        final = cv2.warpPerspective(original,M,(cols,rows))
        cv2.imshow("Final", final)

        # cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
        cv2.waitKey()
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

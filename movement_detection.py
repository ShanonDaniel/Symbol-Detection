import cv2
import cv2 as cv
import numpy as np
import imagehash
from PIL import Image

def getHashDiff(image1, image2):
    # Convert cv2Img from OpenCV format to PIL format
    im1 = cv2.resize(image1, (128, 128))
    im2 = cv2.resize(image2, (128, 128))
    pilImg1 = Image.fromarray(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB))
    pilImg2 = Image.fromarray(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))
    
    # Get the average hashes of both images
    hash0 = imagehash.average_hash(pilImg1, hash_size = 16)
    hash1 = imagehash.average_hash(pilImg2, hash_size = 16)
    
    hashDiff = hash0 - hash1  # Finds the distance between the hashes of images
    return hashDiff


def motion_detector_gist():

    previous_frame = None

    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture('input.mp4')
    
    # Check if camera opened successfully
    if (cap.isOpened()== False):
        print("Error opening video file")
    while (cap.isOpened()):

        # 1. Load image; convert to RGB
        ret, img_brg = cap.read()
        if ret:
            img_rgb = cv2.cvtColor(src=img_brg, code=cv2.COLOR_BGR2RGB)


            # 2. Prepare image; grayscale and blur
            prepared_frame = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
            prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(5, 5), sigmaX=0)

            # 2. Calculate the difference
            if (previous_frame is None):
                # First frame; there is no previous one yet
                previous_frame = prepared_frame
                continue

            # 3. calculate difference and update previous frame
            # print(getHashDiff(previous_frame, prepared_frame))
            diff_frame = cv2.absdiff(src1=previous_frame, src2=prepared_frame)
            previous_frame = prepared_frame
            
            # 4. Dilute the image a bit to make differences more seeable; more suitable for contour detection
            kernel = np.ones((5, 5))
            diff_frame = cv2.dilate(diff_frame, kernel, 1)
            
            # 5. Only take different areas that are different enough (>20 / 255)
            thresh_frame = cv2.threshold(src=diff_frame, thresh=20, maxval=255, type=cv2.THRESH_BINARY)[1]
            h, w = thresh_frame.shape
            err = np.sum(thresh_frame**2)
            mse = err/(float(h*w)) * 100
            print(f'mse = {mse}')
            # cv2.imshow("Threshold Frame", thresh_frame)
            # print(imagehash.average_hash(Image.fromarray(cv2.cvtColor(thresh_frame, cv2.COLOR_BGR2RGB))))
            # 6. Find and optionally draw contours
            contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
            # Comment below to stop drawing contours
            cv2.drawContours(image=img_rgb, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            # Uncomment 6 lines below to stop drawing rectangles
            # for contour in contours:
            #   if cv2.contourArea(contour) < 50:
            #     # too small: skip!
            #       continue
            #   (x, y, w, h) = cv2.boundingRect(contour)
            #   cv2.rectangle(img=img_rgb, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)

            cv2.imshow('Motion detector', img_rgb)

            if (cv2.waitKey(30) == 27):
                # out.release()
                break
        else:
            break
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
  motion_detector_gist()
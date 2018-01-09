# importing required packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt

# Loading image
img=cv2.imread("E:/Work_Space/Python/Project/m1.jpg",1)

# Resizing to width=350
img = imutils.resize(img, width=350)
ratio = img.shape[0] / float(img.shape[0])

#cv2.imshow("Original Image", img)
#cv2.waitKey(0)
#cv2.close()

# Denoising the image
dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

#cv2.imshow("Denoised Image",dst)
#cv2.waitKey(0)
#cv2.close()

# converting RGB image to Grayscale image
img_grey = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

#cv2.imshow("Grayscale Image",img_grey)
#cv2.waitKey(0)
#cv2.close()

# converting Grayscale image to Binary Image using Otsu Method
(thresh, im_bw) = cv2.threshold(img_grey, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Finding Contours
cnts = cv2.findContours(im_bw.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

#cv2.imshow("Binary Image",im_bw)
#cv2.waitKey(0)
#cv2.close()

areas=[]
count=[]
i=1
for c in cnts:
    areas.append(cv2.contourArea(c))
    M = cv2.moments(c)
    cX = int((M["m10"] / M["m00"]) * ratio)
    cY = int((M["m01"] / M["m00"]) * ratio)
    c = c.astype("float")
    c *= ratio
    c = c.astype("int")
    cv2.drawContours(im_bw, [c], -1, (127, 255, 127), 2)
    cv2.putText(im_bw, str(i), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,0.5, (127, 127, 127), 2)
    cv2.imshow("Image", im_bw)
    count.append(i)
    i+=1
    cv2.waitKey(0)
print(areas)

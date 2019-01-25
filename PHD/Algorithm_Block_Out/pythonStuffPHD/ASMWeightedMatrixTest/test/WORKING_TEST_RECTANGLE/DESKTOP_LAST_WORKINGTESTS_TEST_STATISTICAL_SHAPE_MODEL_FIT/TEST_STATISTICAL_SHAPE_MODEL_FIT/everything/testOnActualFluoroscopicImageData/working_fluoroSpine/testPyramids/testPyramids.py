import cv2
import numpy as np
import sys

''' file name : pyramids.py
Description : This sample shows how to downsample and upsample images
This is Python version of this tutorial : http://opencv.itseez.com/doc/tutorials/imgproc/pyramids/pyramids.html#pyramids
Level : Beginner
Benefits : Learn to use 1) cv2.pyrUp and 2) cv2.pyrDown
Usage : python pyramids.py 
Written by : Abid K. (abidrahman2@gmail.com) , Visit opencvpython.blogspot.com for more tutorials '''


print " Zoom In-Out demo "
print " Press u to zoom "
print " Press d to zoom "

img = cv2.imread(sys.argv[1])

while True:
    h,w = img.shape[:2]
    
    cv2.imshow("image", img)
    k = cv2.waitKey(1) & 0xFF

    
    if k==27 :
        break

    elif k == ord("u"):  # Zoom in, make image double size
        img = cv2.pyrUp(img,dstsize = (2*w,2*h))
        cv2.imshow('image',img)

    elif k == ord("d"):  # Zoom down, make image half the size
        img = cv2.pyrDown(img,dstsize = (w/2,h/2))        
        cv2.imshow('image',img)

cv2.destroyAllWindows() 

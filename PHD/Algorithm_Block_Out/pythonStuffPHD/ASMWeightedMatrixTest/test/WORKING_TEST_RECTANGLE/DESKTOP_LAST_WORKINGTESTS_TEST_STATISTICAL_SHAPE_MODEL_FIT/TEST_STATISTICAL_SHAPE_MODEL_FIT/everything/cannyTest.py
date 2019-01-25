''' file name : canny.py

Description : This sample shows how to find edges using canny edge detection

This is Python version of this tutorial : http://opencv.itseez.com/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html

Level : Beginner

Benefits : Learn to apply canny edge detection to images.

Usage : python canny.py 

Written by : Abid K. (abidrahman2@gmail.com) , Visit opencvpython.blogspot.com for more tutorials '''


import cv2
import numpy as np
import sys

def CannyThreshold(lowThreshold):
    detected_edges = cv2.GaussianBlur(gray,(3,3),0)
    detected_edges = cv2.Canny(detected_edges,lowThreshold,lowThreshold*ratio,apertureSize = kernel_size)
    dst = cv2.bitwise_and(img,img,mask = detected_edges)  # just add some colours to edges from original image.
    cv2.imshow('canny demo',dst)
    cv2.imwrite("CANNY.jpg",dst)
    #cv2.imwrite(sys.argv[1],dst)

lowThreshold = 17
max_lowThreshold = 100
ratio = 3
kernel_size = 3

img = cv2.imread(sys.argv[1])

#smoothed before passed to canny edge detector
kernel = np.ones((3,3),np.float32)/9
img = cv2.filter2D(img,-1,kernel)
#cv2.imshow("smoothed", dst)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.namedWindow('canny demo')

CannyThreshold(17)

cv2.createTrackbar('Min threshold','canny demo',lowThreshold, max_lowThreshold, CannyThreshold)

#CannyThreshold(0)  # initialization
if cv2.waitKey(0) == 27:
	
	cv2.destroyAllWindows()

# visit for output results : http://opencvpython.blogspot.com/2012/06/image-derivatives-sobel-and-scharr.html

'''Biomedical Image Understanding: Methods and Applications'''

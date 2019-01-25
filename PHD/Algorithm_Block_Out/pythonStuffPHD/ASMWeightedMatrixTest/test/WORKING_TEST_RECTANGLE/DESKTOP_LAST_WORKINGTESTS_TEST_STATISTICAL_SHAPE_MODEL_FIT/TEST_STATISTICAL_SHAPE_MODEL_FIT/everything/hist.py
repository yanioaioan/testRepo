import cv2
import numpy as np

hbins = 180
sbins = 255
hrange = [0,180]
srange = [0,256]
ranges = hrange+srange  

#TARGET IMAGE ROI
targetImage = cv2.imread('images/grey_image_1.jpg')#iso_colour_bars.png')
rows,cols = targetImage.shape[:2]
targetImage_hsv = cv2.cvtColor(targetImage,cv2.COLOR_BGR2HSV)

#targetImage_hsv_Roi = targetImage_hsv[0:10,180:190].copy() 
targetImage_hsv_Roi = targetImage_hsv[170:180,430:440].copy() 

targetImage_hsv_Roi_hist = cv2.calcHist(targetImage_hsv_Roi,[0,1],None,[180,256],ranges)
cv2.normalize(targetImage_hsv_Roi_hist,targetImage_hsv_Roi_hist,0,255,cv2.NORM_MINMAX)


#TEST IMAGE ROI 1
testImage = cv2.imread('images/grey_image_1.jpg')#iso_colour_bars.png')
rows,cols = testImage.shape[:2]
testImage_hsv = cv2.cvtColor(testImage,cv2.COLOR_BGR2HSV)

#testImage_hsv_Roi_1 = testImage_hsv[10:20,180:190].copy() 
testImage_hsv_Roi_1 = testImage_hsv[170:180,600:610].copy() 

testImage_hsv_Roi_1_hist = cv2.calcHist(testImage_hsv_Roi_1,[0,1],None,[180,256],ranges)
cv2.normalize(testImage_hsv_Roi_1_hist, testImage_hsv_Roi_1_hist,0,255,cv2.NORM_MINMAX)


#TEST IMAGE ROI 2
testImage = cv2.imread('images/grey_image_1.jpg')#iso_colour_bars.png')
rows,cols = testImage.shape[:2]
testImage_hsv = cv2.cvtColor(testImage,cv2.COLOR_BGR2HSV)

#testImage_hsv_Roi_2 = testImage_hsv[0:10,250:260].copy() 
testImage_hsv_Roi_2 = testImage_hsv[170:180,110:120].copy() 

testImage_hsv_Roi_2_hist = cv2.calcHist(testImage_hsv_Roi_2,[0,1],None,[180,256],ranges)
cv2.normalize(testImage_hsv_Roi_2_hist, testImage_hsv_Roi_2_hist,0,255,cv2.NORM_MINMAX)


#HISTOGRAM COMPARISON
for i in xrange(1):

	a_testImage_hsv_Roi_1_hist_DIFF = cv2.compareHist(targetImage_hsv_Roi_hist,testImage_hsv_Roi_1_hist,cv2.cv.CV_COMP_CORREL)
	b_testImage_hsv_Roi_2_hist_DIFF = cv2.compareHist(targetImage_hsv_Roi_hist,testImage_hsv_Roi_2_hist,cv2.cv.CV_COMP_CORREL)
	print "Method: {0} -- a: {1} , b: {2}".format(i, a_testImage_hsv_Roi_1_hist_DIFF, b_testImage_hsv_Roi_2_hist_DIFF)

#if method 0 has been used in compareHist then->: the highest the value, the more accurate the similarity
if a_testImage_hsv_Roi_1_hist_DIFF > b_testImage_hsv_Roi_2_hist_DIFF:
	print "Roi_1 is more similar to target image roi of interest"
else:
	print "Roi_2 is more similar to target image roi of interest"
	

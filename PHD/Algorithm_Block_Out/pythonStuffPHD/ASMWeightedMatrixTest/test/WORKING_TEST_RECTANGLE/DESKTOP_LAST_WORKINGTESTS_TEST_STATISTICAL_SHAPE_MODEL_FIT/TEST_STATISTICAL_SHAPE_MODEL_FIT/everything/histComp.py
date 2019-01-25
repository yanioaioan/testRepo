''' file name : comparehist.py
Description : This sample shows how to determine how well two histograms match each other.
This is Python version of this tutorial : http://opencv.itseez.com/doc/tutorials/imgproc/histograms/histogram_comparison/histogram_comparison.html
Level : Beginner
Benefits : Learn to use cv2.compareHist and create 2D histograms
Usage : python comparehist.py
Written by : Abid K. (abidrahman2@gmail.com) , Visit opencvpython.blogspot.com for more tutorials '''


'''

import cv2
import numpy as np

base = cv2.imread('images/grey_image_1.jpg')
test1 = cv2.imread('images/grey_image_2.jpg')
test2 = cv2.imread('images/grey_image_3.jpg')

rows,cols = base.shape[:2]

basehsv = cv2.cvtColor(base,cv2.COLOR_BGR2HSV)
test1hsv = cv2.cvtColor(test1,cv2.COLOR_BGR2HSV)
test2hsv = cv2.cvtColor(test2,cv2.COLOR_BGR2HSV)

halfhsv = basehsv[rows/2:rows-1,cols/2:cols-1].copy()  # Take lower half of the base image for testing

hbins = 180
sbins = 255
hrange = [0,180]
srange = [0,256]
ranges = hrange+srange                                  # ranges = [0,180,0,256]


histbase = cv2.calcHist(basehsv,[0,1],None,[180,256],ranges)
cv2.normalize(histbase,histbase,0,255,cv2.NORM_MINMAX)

histhalf = cv2.calcHist(halfhsv,[0,1],None,[180,256],ranges)
cv2.normalize(histhalf,histhalf,0,255,cv2.NORM_MINMAX)

histtest1 = cv2.calcHist(test1hsv,[0,1],None,[180,256],ranges)
cv2.normalize(histtest1,histtest1,0,255,cv2.NORM_MINMAX)

histtest2 = cv2.calcHist(test2hsv,[0,1],None,[180,256],ranges)
cv2.normalize(histtest2,histtest2,0,255,cv2.NORM_MINMAX)

for i in xrange(4):
    base_base = cv2.compareHist(histbase,histbase,i)
    base_half = cv2.compareHist(histbase,histhalf,i)
    base_test1 = cv2.compareHist(histbase,histtest1,i)
    base_test2 = cv2.compareHist(histbase,histtest2,i)
    print "Method: {0} -- base-base: {1} , base-half: {2} , base-test1: {3}, base_test2: {4}".format(i,base_base,base_half,base_test1,base_test2)

'''


 

'''
import cv2
import numpy as np

img = cv2.imread('histogramTestImage.jpg')
h = np.zeros((400,300,3))

bins = np.arange(256).reshape(256,1)
color = [ (255,0,0),(0,255,0),(0,0,255) ]

mask = np.zeros(img.shape[:2], np.uint8)
mask[400:400, 499:499] = 255

for ch, col in enumerate(color):
	print ch,col
	
	hist_item = cv2.calcHist([img],[ch],None,[256],[0,255])
	cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
	hist=np.int32(np.around(hist_item))
	pts = np.column_stack((bins,hist))
	cv2.polylines(h,[pts],False,col)

h=np.flipud(h)

cv2.imshow('colorhist',h)
cv2.waitKey(0)

'''


'''
import cv2
import numpy as np

img = cv2.imread('histogramTestImage.jpg')

 
# create a mask
mask = np.zeros(img.shape[:2], np.uint8)
mask[400:400, 499:499] = 255
masked_img = cv2.bitwise_and(img,img,mask = mask)

# Calculate histogram with mask and without mask
# Check third argument for mask
hist_full = cv2.calcHist([img],[0],None,[256],[0,256])
hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])




h = np.zeros((300,256,3))
bins = np.arange(256).reshape(256,1)
color = [ (255,0,0),(0,255,0),(0,0,255) ]

print (hist_mask)


for ch, col in enumerate(color):
	print ch,col
	hist_item = cv2.calcHist([img],[0],None,[256],[0,256])#cv2.calcHist([img],[ch],None,[256],[0,255])
	
	cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
	hist=np.int32(np.around(hist_mask))
	pts = np.column_stack((bins,hist))
	cv2.polylines(h,[pts],False,col)
h=np.flipud(h)
cv2.imshow('colorhist',h)
cv2.waitKey(0)

#plt.subplot(221), plt.imshow(img, 'gray')
#plt.subplot(222), plt.imshow(mask,'gray')
#plt.subplot(223), plt.imshow(masked_img, 'gray')
#plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
#plt.xlim([0,256])
#
#plt.show()
'''


'''
import cv2
import cv
import numpy as np
import time


img1 = cv2.imread('red.png')
#img1= cv2.cvtColor(img1,cv.CV_BGR2HSV)
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

mask = np.zeros(img1.shape[:2], np.uint8)
mask[0:0, 100:100] = 255


img2 = cv2.imread('blue.png')
#img2= cv2.cvtColor(img2,cv.CV_BGR2HSV)
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

mask2 = np.zeros(img2.shape[:2], np.uint8)
mask2[0:0, 100:100] = 255


h = np.zeros((300,256,3))
bins = np.arange(256).reshape(256,1)
color = [ (255,0,0),(0,255,0),(0,0,255) ]

for ch, col in enumerate(color):
	#print ch,col
	hist_item1 = cv2.calcHist([img1],[0],None,[256],[0,256])#[180,256],[0, 180, 0, 256])
	hist_item2 = cv2.calcHist([img2],[0],None,[256],[0,256])#[180,256],[0, 180, 0, 256])
	cv2.normalize(hist_item1,hist_item1,0,255,cv2.NORM_MINMAX)
	cv2.normalize(hist_item2,hist_item2,0,255,cv2.NORM_MINMAX)
	sc= cv2.compareHist(hist_item1, hist_item2, cv.CV_COMP_CORREL)
	print sc

	#hist=np.int32(np.around(hist_item2))
	#pts = np.column_stack((bins,hist))    
	#cv2.polylines(h,[pts],False,col)

#h=np.flipud(h)
#cv2.imshow('hist.png',h)
#cv2.waitKey(0)

'''


''' file name : comparehist.py
Description : This sample shows how to determine how well two histograms match each other.
This is Python version of this tutorial : http://opencv.itseez.com/doc/tutorials/imgproc/histograms/histogram_comparison/histogram_comparison.html
Level : Beginner
Benefits : Learn to use cv2.compareHist and create 2D histograms
Usage : python comparehist.py
Written by : Abid K. (abidrahman2@gmail.com) , Visit opencvpython.blogspot.com for more tutorials '''

import cv2
import numpy as np
'''
base = cv2.imread('images/grey_image_1.jpg')
test1 = cv2.imread('images/grey_image_2.jpg')
test2 = cv2.imread('images/grey_image_3.jpg')

rows,cols = test2.shape[:2]

basehsv = cv2.cvtColor(base,cv2.COLOR_BGR2HSV)
test1hsv = cv2.cvtColor(test1,cv2.COLOR_BGR2HSV)
test2hsv = cv2.cvtColor(test2,cv2.COLOR_BGR2HSV)

halfhsv = test2hsv[rows/2:rows-1,cols/2:cols-1].copy()  # Take lower half of the test2 image for testing
#halfhsv = test2hsv[350:499,350:499].copy() 

hbins = 180
sbins = 255
hrange = [0,180]
srange = [0,256]
ranges = hrange+srange                                  # ranges = [0,180,0,256]


histbase = cv2.calcHist(basehsv,[0,1],None,[180,256],ranges)
cv2.normalize(histbase,histbase,0,255,cv2.NORM_MINMAX)

histhalf = cv2.calcHist(halfhsv,[0,1],None,[180,256],ranges)
cv2.normalize(histhalf,histhalf,0,255,cv2.NORM_MINMAX)

histtest1 = cv2.calcHist(test1hsv,[0,1],None,[180,256],ranges)
cv2.normalize(histtest1,histtest1,0,255,cv2.NORM_MINMAX)

histtest2 = cv2.calcHist(test2hsv,[0,1],None,[180,256],ranges)
cv2.normalize(histtest2,histtest2,0,255,cv2.NORM_MINMAX)

for i in xrange(1):
    base_base = cv2.compareHist(histbase,histbase,i)
    base_half = cv2.compareHist(histbase,histhalf,i)
    base_test1 = cv2.compareHist(histbase,histtest1,i)
    base_test2 = cv2.compareHist(histbase,histtest2,i)
    print "Method: {0} -- base-base: {1} , base-half: {2} , base-test1: {3}, base_test2: {4}".format(i,base_base,base_half,base_test1,base_test2)
   
'''


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

#if method 0 (CV_COMP_CORREL) has been used in compareHist then->: the highest the value, the more accurate the similarity
if a_testImage_hsv_Roi_1_hist_DIFF > b_testImage_hsv_Roi_2_hist_DIFF:
	print "Roi_1 is more similar to target image roi of interest"
else:
	print "Roi_2 is more similar to target image roi of interest"
	


import cv2
import numpy as np
import sys

#http://docs.opencv.org/3.1.0/d1/db7/tutorial_py_histogram_begins.html
#http://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_calculation/histogram_calculation.html#histogram-calculation

hbins = 180
sbins = 255
hrange = [0,180]
srange = [0,256]
ranges = hrange+srange  


def hist_lines(im,x1,x2,y1,y2):	 	
    h = np.zeros((300,256,3))	 	   
    #if len(im.shape)!=2:	 	    
    #    print "hist_lines applicable only for grayscale images"	 
    #    #print "so converting image to grayscale for representation"	
    #    im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    print "im.shape=%s %s"%(im.shape)

    im = im[y1:y2,x1:x2].copy() 
    
    
    
    
    hist_item = cv2.calcHist([im],[0],None,[256],[0,256])	 	  
    
    cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
    hist=np.int32(np.around(hist_item))	 	    
    for x,y in enumerate(hist):	 	   
        cv2.line(h,(x,0),(x,y),(255,255,255))	 	        
    y = np.flipud(h)
    cv2.namedWindow('histogram_target[%d:%d,%d:%d]'%(y1,y2,x1,x2),cv2.WINDOW_NORMAL)
    cv2.imshow('histogram_target[%d:%d,%d:%d]'%(y1,y2,x1,x2),y)	 	  
    
    return hist_item

#TARGET IMAGE ROI
targetImage = cv2.imread(sys.argv[1],cv2.IMREAD_GRAYSCALE)#Make SURE YOU READ GRAYSCALE OTHERWISE HISTOGRAM DOESN'T WORK	 iso_colour_bars.png')

target_x1=450
target_x2=500
target_y1=150
target_y2=200
targetRoi = hist_lines(targetImage,target_x1,target_x2,target_y1,target_y2)#x1x2 y1y2

ROI_1_x1=650
ROI_1_x2=700
ROI_1_y1=150
ROI_1_y2=200
more_similar = hist_lines(targetImage,ROI_1_x1,ROI_1_x2,ROI_1_y1,ROI_1_y2)#x1x2 y1y2

ROI_2_x1=150
ROI_2_x2=200
ROI_2_y1=150
ROI_2_y2=200
less_similar = hist_lines(targetImage,ROI_2_x1,ROI_2_x2,ROI_2_y1,ROI_2_y2)#x1x2 y1y2

#After having calculated histograms based on grayscaled image, convert to color so as to show rectangle ROIs in color
targetImage = cv2.cvtColor(targetImage,cv2.COLOR_GRAY2BGR)

#draw target region
topLeftCorner=(target_x1,target_y1)
bottomRightCorner=(target_x2,target_y2)
cv2.rectangle(targetImage, topLeftCorner , bottomRightCorner,(0,0,255),1)

#draw more similar region
topLeftCorner=(ROI_1_x1,ROI_1_y1)
bottomRightCorner=(ROI_1_x2,ROI_1_y2)
cv2.rectangle(targetImage, topLeftCorner , bottomRightCorner,(0,255,0),1)


#draw target region
topLeftCorner=(ROI_2_x1,ROI_2_y1)
bottomRightCorner=(ROI_2_x2,ROI_2_y2)
cv2.rectangle(targetImage, topLeftCorner , bottomRightCorner,(255,0,0),1)

cv2.imshow('targetImage',targetImage)


#for i in xrange(1):

a_testImage_hsv_Roi_1_hist_DIFF = cv2.compareHist(more_similar,targetRoi,cv2.cv.CV_COMP_CORREL)
b_testImage_hsv_Roi_2_hist_DIFF = cv2.compareHist(less_similar,targetRoi,cv2.cv.CV_COMP_CORREL)
print "Method: {0} -- a: {1} , b: {2}".format(cv2.cv.CV_COMP_CORREL, a_testImage_hsv_Roi_1_hist_DIFF, b_testImage_hsv_Roi_2_hist_DIFF)

#if method 0 grey_image_1 has been used in compareHist then->: the highest the value, the more accurate the similarity
if a_testImage_hsv_Roi_1_hist_DIFF > b_testImage_hsv_Roi_2_hist_DIFF:
	print "more_similar is more similar to target image roi of interest (RED target is more similar to Green target)"
else:
	print "less_similar is more similar to target image roi of interest (RED target is more similar to Blue target)"
	

cv2.waitKey(0)

'''
rows,cols = targetImage.shape[:2]
# convert the image to grayscale and create a histogram
targetImage_hsv = cv2.cvtColor(targetImage, cv2.COLOR_BGR2GRAY)
targetImage_hsv_Roi = targetImage_hsv[100:150,400:500].copy() 

targetImage_hsv_Roi_hist = cv2.calcHist(targetImage_hsv_Roi,[0], None, [256], [0, 256])
cv2.normalize(targetImage_hsv_Roi_hist,targetImage_hsv_Roi_hist,0,255,cv2.NORM_MINMAX)


#TEST IMAGE ROI 1
testImage = cv2.imread('images/lion_greyscale.jpg')#iso_colour_bars.png')
rows,cols = testImage.shape[:2]
# convert the image to grayscale and create a histogram
testImage_hsv = cv2.cvtColor(testImage,cv2.COLOR_BGR2GRAY)
testImage_hsv_Roi_1 = testImage_hsv[100:150,250:350].copy()  

testImage_hsv_Roi_1_hist = cv2.calcHist(testImage_hsv_Roi_1,[0], None, [256], [0, 256])
cv2.normalize(testImage_hsv_Roi_1_hist, testImage_hsv_Roi_1_hist,0,255,cv2.NORM_MINMAX)


#TEST IMAGE ROI 2
testImage = cv2.imread('images/lion_greyscale.jpg')#iso_colour_bars.png')
rows,cols = testImage.shape[:2]
# convert the image to grayscale and create a histogram
testImage_hsv = cv2.cvtColor(testImage,cv2.COLOR_BGR2GRAY)
testImage_hsv_Roi_2 = testImage_hsv[100:150,500:599].copy() 

testImage_hsv_Roi_2_hist = cv2.calcHist(testImage_hsv_Roi_2,[0], None, [256], [0, 256])
cv2.normalize(testImage_hsv_Roi_2_hist, testImage_hsv_Roi_2_hist,0,255,cv2.NORM_MINMAX)


#HISTOGRAM COMPARISON
for i in xrange(1):

	a_testImage_hsv_Roi_1_hist_DIFF = cv2.compareHist(targetImage_hsv_Roi_hist,testImage_hsv_Roi_1_hist,cv2.cv.CV_COMP_INTERSECT)
	b_testImage_hsv_Roi_2_hist_DIFF = cv2.compareHist(targetImage_hsv_Roi_hist,testImage_hsv_Roi_2_hist,cv2.cv.CV_COMP_INTERSECT)
	print "Method: {0} -- a: {1} , b: {2}".format(i, a_testImage_hsv_Roi_1_hist_DIFF, b_testImage_hsv_Roi_2_hist_DIFF)

#if method 0 has been used in compareHist then->: the highest the value, the more accurate the similarity
if a_testImage_hsv_Roi_1_hist_DIFF > b_testImage_hsv_Roi_2_hist_DIFF:
	print "Roi_1 is more similar to target image roi of interest"
else:
	print "Roi_2 is more similar to target image roi of interest"
	
'''

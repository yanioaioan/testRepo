#!/usr/bin/env python

##ADDITION
import itertools

import sys
import os
import cv
import cv2
import glob
import math
import numpy as np
from random import randint
import random
import array
import copy

import scipy.spatial.distance;
from scipy import linalg

def exitONKeyPress():
    c=cv.WaitKey()
    #whichever integer key code makes the app exit
    if c==1048603:
        exit()

def getAngleBetween2Lines(l1x1,l1y1,l1x2,l1y2, l2x1,l2y1,l2x2,l2y2):
    angle1 = math.atan2(l1y1 - l1y2,  l1x1 - l1x2);
    angle2 = math.atan2(l2y1 - l2y2,  l2x1 - l2x2);
    return angle1-angle2;

def showCVImage(image):
    #ShapeBeforeResizeToFitImage = cv.CreateImage(cv.GetSize(image), self.g_image[scale].depth, 1)
    #cv.Copy(self.g_image[scale], ShapeBeforeResizeToFitImage)
    #for i, pt in enumerate(self.shape.pts):
    #    cv.Circle(ShapeBeforeResizeToFitImage, ( int(pt.x), int(pt.y) ), 2, (100,100,100))
    cv.NamedWindow("showCVImage", cv.CV_WINDOW_NORMAL)
    cv.ShowImage("showCVImage",image)


def showRectOnImage(image,x1,x2,y1,y2, color):
    #After having calculated histograms based on grayscaled image, convert to color so as to show rectangle ROIs in color
    image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)

    #draw target region
    topLeftCorner=(int(x1), int(y1))
    bottomRightCorner=(int(x2), int(y2))
    cv2.rectangle(image, topLeftCorner, bottomRightCorner, color, 1)
    cv2.imshow('rectOnImage',image)


def sobel(img):

    #Description : This sample shows how to find derivatives of an image

    #This is Python version of this tutorial : http://opencv.itseez.com/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html#sobel-derivatives

    #Level : Beginner

    #Benefits : Learn to use Sobel and Scharr derivatives

    #Usage : python sobel.py

    #Written by : Abid K. (abidrahman2@gmail.com) , Visit opencvpython.blogspot.com for more tutorials '''




    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    ConvertBackToRgbAfterSobelizing=False

    #make sure we read as grayscale
    #img = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    #img = cv2.imread(sys.argv[2],cv2.IMREAD_GRAYSCALE)

    gray = img

    if len(img.shape)==3:
        ConvertBackToRgbAfterSobelizing=True
        gray = cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY)

    #blur it
    gray = cv2.fastNlMeansDenoising(gray,None,10,7,21)

    gray = cv2.GaussianBlur(gray,(3,3),0)

    # Gradient-X
    grad_x = cv2.Sobel(gray,ddepth,1,0,ksize = 3, scale = scale, delta = delta,borderType = cv2.BORDER_DEFAULT)
    #grad_x = cv2.Scharr(gray,ddepth,1,0)

    # Gradient-Y
    grad_y = cv2.Sobel(gray,ddepth,0,1,ksize = 3, scale = scale, delta = delta, borderType = cv2.BORDER_DEFAULT)
    #grad_y = cv2.Scharr(gray,ddepth,0,1)

    abs_grad_x = cv2.convertScaleAbs(grad_x)   # converting back to uint8
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    dst = cv2.addWeighted(abs_grad_x,0.5,abs_grad_y,0.5,0)
    #dst = cv2.add(abs_grad_x,abs_grad_y)

    cv2.imshow('dst',dst)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # If img parameter was and RGB image, then after taking the grayscaling & 1st derivative on it, we need to convert it back to RGB at the end before returning
    if ConvertBackToRgbAfterSobelizing == True:
        dst = cv2.cvtColor(dst,cv2.COLOR_GRAY2BGR)

    return dst



    # To see the results, visit : http://opencvpython.blogspot.com/2012/06/image-derivatives-sobel-and-scharr.html



def Sobelize_1st_Der(img):
        #Sobel way of calculating 1st derivatives-->are they normalized(so as to avoid being affected by noise)
        scale = 1
        delta = 0
        ddepth = cv2.CV_16S


        img = cv2.GaussianBlur(img,(5,5),0)

        gray = img
        #Check if the image type not grayscale as when it is it returns a tuple of 3 elements ex.(618, 1024, 3)
        if len(img.shape)==3:
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Gradient-X
        grad_x = cv2.Sobel(gray,ddepth,1,0,ksize = 3, scale = scale, delta = delta,borderType = cv2.BORDER_DEFAULT)
        #grad_x = cv2.Scharr(gray,ddepth,1,0)

        # Gradient-Y
        grad_y = cv2.Sobel(gray,ddepth,0,1,ksize = 3, scale = scale, delta = delta, borderType = cv2.BORDER_DEFAULT)
        #grad_y = cv2.Scharr(gray,ddepth,0,1)

        abs_grad_x = cv2.convertScaleAbs(grad_x)   # converting back to uint8
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        dst = cv2.addWeighted(abs_grad_x,0.5,abs_grad_y,0.5,0)
        #dst = cv2.add(abs_grad_x,abs_grad_y)

	cv2.imshow('gradientOutput',dst)
        cv2.waitKey(0)
        return dst


#http://stackoverflow.com/questions/36637400/how-to-normalize-opencv-sobel-filter-given-kernel-size
#https://github.com/abidrahmank/OpenCV2-Python/blob/master/Official_Tutorial_Python_Codes/3_imgproc/sobel.py
def gradient(img, dx, dy, ksize):


    ##deriv_filter = cv2.getGaussianKernel(5, 2, cv2.CV_32F)



    #    scale = 1

    #    delta = 1
    #    ddepth = cv2.CV_16S

    #    kernelSize=3
    #    blurKernel=3

    #    #img = cv2.GaussianBlur(img,(blurKernel,blurKernel),0)
    #    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #    #gray=img

    #    #DENOISE BEFORE GRADIENT - as derivatives a re sensitive to noise
    #    #gray = cv2.fastNlMeansDenoisingColored(gray,None,10,10,7,21)



    #    # Gradient-X
    #    grad_x = cv2.Sobel(img,ddepth,1,0,ksize = kernelSize, scale = scale, delta = delta,borderType = cv2.BORDER_DEFAULT)
    #    #grad_x = cv2.Scharr(img,ddepth,1,0)

    #    # Gradient-Y
    #    grad_y = cv2.Sobel(img,ddepth,0,1,ksize = kernelSize, scale = scale, delta = delta, borderType = cv2.BORDER_DEFAULT)
    #    #grad_y = cv2.Scharr(img,ddepth,0,1)

    #    abs_grad_x = cv2.convertScaleAbs(grad_x)   # converting back to uint8
    #    abs_grad_y = cv2.convertScaleAbs(grad_y)

    #    dst = cv2.addWeighted(abs_grad_x,0.5,abs_grad_y,0.5,0)



    #dst = cv2.add(abs_grad_x,abs_grad_y)


    #Calculate X gradient
    #    w,h,c=img.shape
    #    gradX = np.zeros((w,h,c), 'uint16')
    #    gradY = np.zeros((w,h,c), 'uint16')
    #    for i in range(w-1):
    #        for j in range(h-1):
    #            gradX.itemset((i,j,0), img[i+1,j,0] -img[i,j,0] )
    #            gradX.itemset((i,j,1), img[i+1,j,1] -img[i,j,1] )
    #            gradX.itemset((i,j,2), img[i+1,j,2] -img[i,j,2] )
    #    print gradX
    #    cv.WaitKey(0)
    #    #Calculate Y gradient
    #    for i in range(w-1):
    #        for j in range(h-1):
    #            gradX.itemset((i,j,0), img[i,j+1,0] -img[i,j,0] )
    #            gradX.itemset((i,j,1), img[i,j+1,1] -img[i,j,1] )
    #            gradX.itemset((i,j,2), img[i,j+1,2] -img[i,j,2] )
    #    print gradY
    #    cv.WaitKey(0)



    #return dst

    #    cv2.imshow('dst',dst)
    #    cv2.waitKey(0)
    #    cv2.destroyAllWindows()
    #    exit()



    #old way of calculating 1st derivatives of the image fed into the function
    #deriv_filter = cv2.getDerivKernels(dx=dx, dy=dy, ksize=ksize, normalize=True)
    #return cv2.sepFilter2D(img, -1, deriv_filter[0], deriv_filter[1])




    lowThreshold = 17
    max_lowThreshold = 100
    ratio = 3
    kernel_size = 3

    #img = cv2.imread(sys.argv[2])

    #smoothed before passed to canny edge detector
    kernel = np.ones((3,3),np.float32)/9
    img = cv2.filter2D(img,-1,kernel)
    #cv2.imshow("smoothed", dst)

    gray=img
    if len(img.shape)==3:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    cv2.namedWindow('canny %s'%(sys.argv[2]))


    detected_edges = cv2.GaussianBlur(gray,(3,3),0)
    detected_edges = cv2.Canny(detected_edges,lowThreshold,lowThreshold*ratio,apertureSize = kernel_size)
    dst = cv2.bitwise_and(img,img,mask = detected_edges)  # just add some colours to edges from original image.
    #cv2.imshow('Cannied: %s'%(sys.argv[2]),dst)

    print "Gradient() function called"
    #c=cv.WaitKey()
    #if c ==1048603:
    #    exit()
    return dst

    #cv2.imshow('test custom filter',img)
    #    c=cv.WaitKey()
    #    if c == 1048603:
    #        exit()











import cv2
import numpy as np
import sys


def showHistogram(histogram,histogramName):
    cv2.normalize(histogram,histogram,0,255,cv2.NORM_MINMAX)
    hist=np.int32(np.around(histogram))

    h = np.zeros((300,256,3))
    for x,y in enumerate(hist):
        cv2.line(h,(x,0),(x,y),(255,255,255))

    y = np.flipud(h)

    #draw histogram
    cv2.namedWindow('%s histogram_target'%(histogramName),cv2.WINDOW_NORMAL)
    cv2.imshow('%s  histogram_target'%(histogramName),y)



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

    #draw histogram
    #cv2.namedWindow('histogram_target[%d:%d,%d:%d]'%(y1,y2,x1,x2),cv2.WINDOW_NORMAL)
    #cv2.imshow('histogram_target[%d:%d,%d:%d]'%(y1,y2,x1,x2),y)

    return hist_item


# Compare two histogram areas along the whisker, and present the results to help decide
# which one (either x1-y1 or x2-t2) is closer to the average color intensity of the target vertebrae itself.
# All histogram calculations are being performed the original image, NOT the derivatives
def hist2(im, x1,x2,y1,y2, prev_x1,prev_x2,prev_y1,prev_y2 ):


    #http://docs.opencv.org/3.1.0/d1/db7/tutorial_py_histogram_begins.html
    #http://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_calculation/histogram_calculation.html#histogram-calculation

    #hbins = 180
    #sbins = 255
    #hrange = [0,180]
    #srange = [0,256]
    #ranges = hrange+srange

    #TARGET IMAGE ROI
    #Make SURE YOU READ GRAYSCALE OTHERWISE HISTOGRAM DOESN'T WORK	 iso_colour_bars.png')

    tmpTargetImage = sys.argv[2]

    targetImage = cv2.imread( tmpTargetImage,cv2.IMREAD_GRAYSCALE)

    #vertebra ROI window to test against (between new possible point to update to, and the previous one we were currently at)
    target_x1=235#175
    target_x2=250#200
    target_y1=125#100
    target_y2=140#125
    targetRoi = hist_lines(targetImage,target_x1,target_x2,target_y1,target_y2)#x1x2 y1y2

    #assign new possible update point ROI window
    ROI_1_x1=x1
    ROI_1_x2=x2
    ROI_1_y1=y1
    ROI_1_y2=y2
    more_similar = hist_lines(targetImage,ROI_1_x1,ROI_1_x2,ROI_1_y1,ROI_1_y2)#x1x2 y1y2

    #assign previous-saved point ROI window, we were currently at
    ROI_2_x1=prev_x1
    ROI_2_x2=prev_x2
    ROI_2_y1=prev_y1
    ROI_2_y2=prev_y2
    less_similar = hist_lines(targetImage,ROI_2_x1,ROI_2_x2,ROI_2_y1,ROI_2_y2)#x1x2 y1y2

    #After having calculated histograms based on grayscaled image, convert to color so as to show rectangle ROIs in color
    targetImage = cv2.cvtColor(targetImage,cv2.COLOR_GRAY2BGR)

    #draw target region
    topLeftCorner=( int(target_x1), int(target_y1))
    bottomRightCorner=( int(target_x2), int(target_y2))
    cv2.rectangle(targetImage, topLeftCorner , bottomRightCorner,(0,0,255),1)

    #draw newer similar region
    topLeftCorner=( int(ROI_1_x1), int(ROI_1_y1))
    bottomRightCorner=( int(ROI_1_x2), int(ROI_1_y2))
    cv2.rectangle(targetImage, topLeftCorner , bottomRightCorner,(0,255,0),1)


    #draw older region
    topLeftCorner=( int(ROI_2_x1), int(ROI_2_y1))
    bottomRightCorner=( int(ROI_2_x2), int(ROI_2_y2))
    cv2.rectangle(targetImage, topLeftCorner , bottomRightCorner,(255,0,0),1)

    cv2.imshow('targetImage',targetImage)
    #cv2.waitKey(0)


    #for i in xrange(1):

    a_testImage_hsv_Roi_1_hist_DIFF = cv2.compareHist(more_similar,targetRoi,cv2.cv.CV_COMP_CORREL)
    b_testImage_hsv_Roi_2_hist_DIFF = cv2.compareHist(less_similar,targetRoi,cv2.cv.CV_COMP_CORREL)
    print "Method: {0} -- a: {1} , b: {2}".format(cv2.cv.CV_COMP_CORREL, a_testImage_hsv_Roi_1_hist_DIFF, b_testImage_hsv_Roi_2_hist_DIFF)

    #if method 0 grey_image_1 has been used in compareHist then->: the highest the value, the more accurate the similarity
    if a_testImage_hsv_Roi_1_hist_DIFF > b_testImage_hsv_Roi_2_hist_DIFF:
	    print "more_similar is more similar to target image roi of interest (RED target is more similar to Green target)"
	    print 'So.. update to newer point as is more simlar than the previous one texturewise!!'
            return True
    else:
	    print "less_similar is more similar to target image roi of interest (RED target is more similar to Blue target)"
	    print 'So.. DONT\'T update to newer point as is LESS simlar than the previous one texturewise!!'
            return False





##https://gist.github.com/mweibel/bd2d6c2271e42ed97b97
##http://docs.opencv.org/trunk/da/d6e/tutorial_py_geometric_transformations.html
##http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html
##getRectSubPix
##http://answers.opencv.org/question/497/extract-a-rotatedrect-area/
def rotatedRectExtractionArea(image, centerX, centerY, npx, npy):

    width=13
    height=7

    #find angle between a line horizontal to the x axis starting from the origin &
    # a line starting from the origin and ending at the best point chosen (npx,npy)
    angle=getAngleBetween2Lines(centerX,centerY,centerX+10,centerY, centerX,centerY,npx,npy) * (180.0/math.pi)
    print angle
    rect = ((centerX,centerY),(width,height),angle)
    box = cv2.cv.BoxPoints(rect)
    box = np.int0(box)

    ## Draw the contour window REC RED
    #cv2.drawContours(image,[box],0,(0,0,255),2)


    #img=cv2.imread("grey_image_1.jpg")

    '''
    details = image.shape
    print "details="
    print details
    cv.WaitKey()
    if len(details)==3:
        rows,cols,ch = details
	print "rows=%s ,cols=%s, ch=%s"%(rows,cols,ch)

    else:
        #ch is ommited as tuple now contains only (width,height)
        rows,cols = details
	print "rows=%s ,cols=%s"%(rows,cols)
        cv.WaitKey()
    '''

    # rect is the RotatedRect (I got it from a contour...)
    rect = ((centerX,centerY),(width,height),angle)
    # matrices we'll use
    #Mat M, rotated, cropped;
    # get angle and size from the bounding box

    print "rot angle=%f"%(rect[2])
    angle = rect[2]
    rect_size_width = rect[1][0]
    rect_size_height = rect[1][1]

    # thanks to http://felix.abecassis.me/2011/10/opencv-rotation-deskewing/
    ## if the skew angle is positive,
    ## the angle of the bounding box is below -45 degrees because the angle is given by taking as a reference a "vertical" rectangle, i.e. with the height greater than the width.
    ## Therefore, if the angle is positive, we swap height and width before cropping

    if (angle < -45.):
        angle += 90.0
        rect_size_width, rect_size_height = rect_size_height, rect_size_width

    # get the rotation matrix

    Cx=rect[0][0]
    Cy=rect[0][1]
    M = cv2.getRotationMatrix2D((Cx,Cy), angle, 1.0);
    print M

    # perform the affine transformation
    extractedRotatedRectangle = cv2.warpAffine(image,M,(image.shape[:2])) # (cols.rows) instead of (image.shape[:2])

    # crop the resulting image

    deskewed_image=cv2.getRectSubPix(extractedRotatedRectangle, (rect_size_width,rect_size_height), (Cx,Cy));

    cv2.namedWindow("img", cv.CV_WINDOW_NORMAL)
    cv2.imshow("img",image)
    #cv.WaitKey()

    cv2.namedWindow("RotatedRectangleExtractionAreaToHistogramTest", cv.CV_WINDOW_NORMAL)
    cv2.imshow("RotatedRectangleExtractionAreaToHistogramTest",deskewed_image)
    #cv.WaitKey()

    return deskewed_image




def ConvertFromCv2ToCvImage(i, imagepath):
    size=cv.GetSize(i)
    grey_imageFor = cv.CreateImage(size, 8, 1)
    grey_imageFormattedToOldCvVersion = cv.LoadImage(imagepath)
    cv.CvtColor(grey_imageFormattedToOldCvVersion, grey_imageFor, cv2.COLOR_BGR2GRAY)
    return grey_imageFor

def drange(start, stop, step):
  r = start
  while r < stop:
    yield r
    r += step

class Point ( object ):
  """ Class to represent a point in 2d cartesian space """
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __add__(self, p):
    """ Return a new point which is equal to this point added to p
    :param p: The other point
    """
    return Point(self.x + p.x, self.y + p.y)

  def __div__(self, i):
    return Point(self.x/i, self.y/i)

  def __eq__(self, other):
    return self.x == other.x and self.y == other.y

  def __ne__(self, other):
    return not self.__eq__(other)

  def __repr__(self):
    """return a string representation of this point. """
    return '(%f, %f)' % (self.x, self.y)

  def dist(self, p):
    """ Return the distance of this point to another point

    :param p: The other point
    """
    return math.sqrt((p.x - self.x)**2 + (p.y - self.y)**2)

class Shape ( object ):
  """ Class to represent a shape.  This is essentially a list of Point
  objects
  """
  def __init__(self, pts = []):
    self.pts = pts
    self.num_pts = len(pts)

  def __add__(self, other):
    """ Operator overloading so that we can add one shape to another
    """
    s = Shape([])
    for i,p in enumerate(self.pts):
      s.add_point(p + other.pts[i])
    return s

  def __div__(self, i):
    """ Division by a constant.
    Each point gets divided by i
    """
    s = Shape([])
    for p in self.pts:
      s.add_point(p/i)
    return s

  def __eq__(self, other):
    for i in range(len(self.pts)):
      if self.pts[i] != other.pts[i]:
        return False
    return True

  def __ne__(self, other):
    return not self.__eq__(other)

  def add_point(self, p, degreesOfRotation=0, CenterOfMass=Point(0,0)):


    if CenterOfMass.x != 0 and CenterOfMass.y != 0 and degreesOfRotation!=0:
        #degreesOfRotation=-15
        f=degreesOfRotation*(math.pi/180)
        #p.x = p.x * math.cos(f) - p.y *math.sin(f)
        #p.y = p.y * math.cos(f) + p.x *math.sin(f)
        p.x = CenterOfMass.x + (p.x-CenterOfMass.x)*math.cos(f)-(p.y-CenterOfMass.y)*math.sin(f)
        p.y = CenterOfMass.y + (p.x-CenterOfMass.x)*math.sin(f)+(p.y-CenterOfMass.y)*math.cos(f)

    self.pts.append(p)
    self.num_pts += 1

  def transform(self, t, degreesOfRotation=0, CenterOfMass=Point(0,0) ):
    s = Shape([])
    for p in self.pts:
      s.add_point(p + t, degreesOfRotation, CenterOfMass)
    return s

  """ Helper methods for shape alignment """
  def __get_X(self, w):
    return sum([w[i]*self.pts[i].x for i in range(len(self.pts))])
  def __get_Y(self, w):
    return sum([w[i]*self.pts[i].y for i in range(len(self.pts))])
  def __get_Z(self, w):
    return sum([w[i]*(self.pts[i].x**2+self.pts[i].y**2) for i in range(len(self.pts))])
  def __get_C1(self, w, s):
    return sum([w[i]*(s.pts[i].x*self.pts[i].x + s.pts[i].y*self.pts[i].y) \
        for i in range(len(self.pts))])
  def __get_C2(self, w, s):
    return sum([w[i]*(s.pts[i].y*self.pts[i].x - s.pts[i].x*self.pts[i].y) \
        for i in range(len(self.pts))])

  def get_alignment_params(self, s, w):
    """ Gets the parameters required to align the shape to the given shape
    using the weight matrix w.  This applies a scaling, transformation and
    rotation to each point in the shape to align it as closely as possible
    to the shape.

    This relies on some linear algebra which we use numpy to solve.

    [ X2 -Y2   W   0][ax]   [X1]
    [ Y2  X2   0   W][ay] = [Y1]
    [ Z    0  X2  Y2][tx]   [C1]
    [ 0    Z -Y2  X2][ty]   [C2]

    We want to solve this to find ax, ay, tx, and ty

    :param shape: The shape to align to
    :param w: The weight matrix
    :return x: [ax, ay, tx, ty]
    """

    X1 = s.__get_X(w)
    X2 = self.__get_X(w)
    Y1 = s.__get_Y(w)
    Y2 = self.__get_Y(w)
    Z = self.__get_Z(w)
    W = sum(w)
    C1 = self.__get_C1(w, s)
    C2 = self.__get_C2(w, s)

    a = np.array([[ X2, -Y2,   W,  0],
                  [ Y2,  X2,   0,  W],
                  [  Z,   0,  X2, Y2],
                  [  0,   Z, -Y2, X2]])

    b = np.array([X1, Y1, C1, C2])
    # Solve equations
    # result is [ax, ay, tx, ty]
    return np.linalg.solve(a, b)

  def apply_params_to_shape(self, p):
    new = Shape([])
    # For each point in current shape
    for pt in self.pts:
      new_x = (p[0]*pt.x - p[1]*pt.y) + p[2]
      new_y = (p[1]*pt.x + p[0]*pt.y) + p[3]
      new.add_point(Point(new_x, new_y))
    return new

  def align_to_shape(self, s, w):
    p = self.get_alignment_params(s, w)
    return self.apply_params_to_shape(p)

  def get_vector(self):
    vec = np.zeros((self.num_pts, 2))
    for i in range(len(self.pts)):
      vec[i,:] = [self.pts[i].x, self.pts[i].y]
    return vec.flatten()

  def get_normal_to_point(self, p_num):
    # Normal to first point
    x = 0; y = 0; mag = 0
    if p_num == 0:
      x = self.pts[1].x - self.pts[0].x
      y = self.pts[1].y - self.pts[0].y
    # Normal to last point
    elif p_num == len(self.pts)-1:
      x = self.pts[-1].x - self.pts[-2].x
      y = self.pts[-1].y - self.pts[-2].y
    # Must have two adjacent points, so...
    else:

      #http://www.gamedev.net/topic/367293-finding-the-normal-to-a-2d-line/
      '''
      in general..
      Given a 2d vector (a,b), the normal is (x,y):
      x = b
      y = -a

      So basically, flip a and b, and negate the y. Two assignments and one negation. Can't be cheaper!
      '''

      x = self.pts[p_num+1].x - self.pts[p_num-1].x
      y = self.pts[p_num+1].y - self.pts[p_num-1].y

    mag = math.sqrt(x**2 + y**2)#normalize as well
    return (-y/mag, x/mag)

  @staticmethod
  def from_vector(vec):
    s = Shape([])
    for i,j in np.reshape(vec, (-1,2)):
      s.add_point(Point(i, j))
    return s

class ShapeViewer ( object ):
  """ Provides functionality to display a shape in a window
  """
  @staticmethod
  def show_shapes(shapes):
    """ Function to show all of the shapes which are passed to it
    """
    cv.NamedWindow("Shape Model Variations", cv.CV_WINDOW_NORMAL)
    # Get size for the window
    max_x = int(max([pt.x for shape in shapes for pt in shape.pts]))
    max_y = int(max([pt.y for shape in shapes for pt in shape.pts]))
    min_x = int(min([pt.x for shape in shapes for pt in shape.pts]))
    min_y = int(min([pt.y for shape in shapes for pt in shape.pts]))

    i = cv.CreateImage((max_x-min_x+20, max_y-min_y+20), cv.IPL_DEPTH_8U, 3)



    cv.Set(i, (0, 0, 0))
    for shape in shapes:
      r = 180#randint(0, 255)
      g = 150#randint(0, 255)
      b = 240#randint(0, 255)
      #r = 0
      #g = 0
      #b = 0
      for pt_num, pt in enumerate(shape.pts):
        #if pt_num>=1:
        #    break
        # Draw normals
        #norm = shape.get_normal_to_point(pt_num)
        #cv.Line(i,(pt.x-min_x,pt.y-min_y), \
        #    (norm[0]*10 + pt.x-min_x, norm[1]*10 + pt.y-min_y), (r, g, b))
        cv.Circle(i, (int(pt.x-min_x), int(pt.y-min_y)), 3, (b, g, r), -1)
      #####print "pt=%d,%d"%(pt.x,pt.y)
      cv.NamedWindow("Active shape Model", cv.CV_WINDOW_NORMAL)
      cv.ShowImage("Active shape Model",i)

  @staticmethod
  def show_modes_of_variation(model, mode):
    # Get the limits of the animation

    ##print "model.evals[mode]",model.evals[mode]
    ###cv.WaitKey()

    start=0
    if model.evals[mode] > 0:
        start = -3*math.sqrt(model.evals[mode])
    stop = -start
    step = (stop - start) / 100

    b_all = np.zeros(model.modes)
    b = start
    while True:
      b_all[mode] = b
      s = model.generate_example(b_all)
      ShapeViewer.show_shapes([s])
      # Reverse direction when we get to the end to keep it running
      if (b < start and step < 0) or (b > stop and step > 0):
        step = -step

        #this break is here to break when the first mode positions has been shown
        break

      b += step
      c = cv.WaitKey(10)
      if c == 1048603:
          #exit()
          break


  @staticmethod
  def draw_model_fitter(f):
    cv.NamedWindow("Fit Model ASM", cv.CV_WINDOW_NORMAL)

    #c = cv.WaitKey(10)
    #####print "f.shape.pts",f.shape.pts
    #c = cv.WaitKey(10)

    # Copy


    size = cv.GetSize(f.greyTargetImage[0])

    print "len of greyTargetImage as many as the pyramid level"
    print len(f.greyTargetImage)
    #cv.WaitKey()

    #create an image which will be the grayscaled of the original
    grey_image = cv.CreateImage(size, 8, 3)

    cv.CvtColor(f.greyTargetImage[0], grey_image, cv.CV_GRAY2RGB)
    backtorgb=grey_image

    i = cv.CreateImage(cv.GetSize(backtorgb), backtorgb.depth, 3)
    cv.Copy(backtorgb, i)

    prevPoint=-1
    nextPoint=-1
    for pt_num, pt in enumerate(f.shape.pts):
      # Draw normals
      #cv.Circle(i, (int(pt.x), int(pt.y)), 1, (0,255,0), -1)

      nextPoint = (int(pt.x), int(pt.y))
      if prevPoint != -1:
          cv.Line(i, prevPoint , nextPoint ,(0,255,0),1)
      prevPoint = nextPoint

      ##print "pt=%d,%d"%(pt.x,pt.y)
    #Draw the original targeted image with cyan landmark points
    ######cv.WaitKey()
    cv.ShowImage("Fit Model ASM",i)
    ####cv.WaitKey()
    #####print 'STAGE - key pressed\n'

class PointsReader ( object ):
  """ Class to read from files provided on Tim Cootes's website."""
  @staticmethod
  def read_points_file(filename):
    """ Read a .pts file, and returns a Shape object """
    s = Shape([])
    num_pts = 0
    with open(filename) as fh:


      # Get expected number of points from file
      first_line = fh.readline()
      if first_line.startswith("version"):
        # Then it is a newer type of file...
        num_pts = int(fh.readline().split()[1])
        # Drop the {
        fh.readline()
      else:
        # It is an older file...
	#####print "FFFFFfirst_line=%s"%first_line
        num_pts = int(first_line)
      for line in fh:
        if not line.startswith("}"):
          pt = line.strip().split()
	  #####print "line",line
          s.add_point(Point(float(pt[0]), float(pt[1])))
    if s.num_pts != num_pts:
      #####print "Unexpected number of points in file.  "\
      "Expecting %d, got %d" % (num_pts, s.num_pts)
    return s

  @staticmethod
  def read_directory(dirname):


    """ Reads an entire directory of .pts files and returns
    them as a list of shapes
    """
    fileNumber=1
    pts = []

    #print "WOWO %d"%(len(os.listdir(dirname)))
    ###cv.WaitKey()

    totalPtsFiles=0
    for file in os.listdir(dirname):
        if file.endswith(".pts"):
            totalPtsFiles=totalPtsFiles+1

    for i in range(0,totalPtsFiles,1):
        file = glob.glob(os.path.join(dirname, "*%d*.pts"%(fileNumber) ))
	#print file

        ####cv.WaitKey()

        f=("%d.pts")%(fileNumber)
        f=file[0]

	##print f
        pts.append(PointsReader.read_points_file(f))
        fileNumber+=1
    return pts

class FitModel:
  """
  Class to fit a model to an image

  :param asm: A trained active shape model
  :param image: An OpenCV image
  :param t: A transformation to move the shape to a new origin
  """

  def __init__(self, asm, image, targetImageName, originalShapes, t=Point(0,0)):#0,0 was working
    #the image target we are testing against

    ##DENOISE ADDITION HERE CONTINUE
    #image = cv2.fastNlMeansDenoising(np.asarray(image[:,:]),None,10,7,21)

    #source = image # source is numpy array
    #image = cv.CreateImageHeader((source.shape[1], source.shape[0]), cv.IPL_DEPTH_8U, 3)
    #cv.SetData(image, source.tostring(),
    #           source.dtype.itemsize * 3 * source.shape[1])
    ##DENOISE ADDITION HERE CONTINUE

    #For each landmark p_num of the shape - maintain a list of previous best points chosen (based on the their window histograms)
    #each element of this list will contain a pair of "topleft, rightBottom" coordinates representing the window ROI
    #of the previous histogram for this particular landmark of the shape
    self.prevBestHistShapePointsList=[]


    self.image = image


    #the array of grey_scaled target image at different resolutions (if implemented)
    self.g_image = []

    #grey target image
    self.greyTargetImage = []

    scale=0

    #in the case WE HAVE 3 PYRAMID LEVELS INDEXED AT 0,1,2
    self.currentSearchPyramidLevel=2#was and SHOULD BE 2
    self.pyramidLevels=3#was and SHOULD BE 3

    #vector containing the covariance matrices calculated once, when forming the grey-level profile (and it is about to be used during "image search")
    self.CovarianceMatricesVec=[]

    #append  (totalnumberofpyramidlevels) empty lists, then fill each one accordingly & use as needed
    for i in range(self.currentSearchPyramidLevel+1):
        self.CovarianceMatricesVec.append([])


    self.tmpCovar=0

    self.COVARIANCE_CALCULATION_ONCE = False

    #this encapsulates all images' normalized derivative profiles for all their landmarks
    self.AllImages_NormalizedtmpLandmarkDerivativeIntensityVector = []

    #for all 3 images' vectors of landmark points in the training set
    self.testImageNameCounter=0

    self.trainingGmean = []

    #self.gaussianMatrix=np.array([
    #                        [0.000001      ,0.000005      ,0.000019      ,0.000055      ,0.000116      ,0.000183      ,0.000213      ,0.000183      ,0.000116      ,0.000055      ,0.000019      ,0.000005          ,0.000001],
    #                        [0.000005      ,0.000026      ,0.0001          ,0.000288      ,0.00061         ,0.000958      ,0.001114      ,0.000958      ,0.00061         ,0.000288      ,0.0001        ,0.000026     ,0.000005],
    #                        [0.000019      ,0.0001          ,0.000389      ,0.001115      ,0.002365      ,0.003713          ,0.004316      ,0.003713      ,0.002365      ,0.001115      ,0.000389      ,0.0001        ,0.000019],
    #                        [0.000055      ,0.000288      ,0.001115      ,0.003196      ,0.00678         ,0.010646      ,0.012374      ,0.010646      ,0.00678       ,0.003196      ,0.001115      ,0.000288      ,0.000055],
    #                        [0.000116      ,0.00061         ,0.002365      ,0.00678         ,0.014384      ,0.022586      ,0.026251      ,0.022586      ,0.014384      ,0.00678           ,0.002365      ,0.00061         ,0.000116],
    #                        [0.000183      ,0.000958      ,0.003713      ,0.010646      ,0.022586      ,0.035465      ,0.04122         ,0.035465      ,0.022586      ,0.010646      ,0.003713      ,0.000958      ,0.000183],#
    #                        [0.000213      ,0.001114      ,0.004316      ,0.012374      ,0.026251      ,0.04122       ,0.04791         ,0.04122         ,0.026251      ,0.012374      ,0.004316      ,0.001114      ,0.000213],
    #                        [0.000183      ,0.000958      ,0.003713      ,0.010646      ,0.022586      ,0.035465      ,0.04122           ,0.035465      ,0.022586      ,0.010646      ,0.003713      ,0.000958      ,0.000183],
    #                        [0.000116      ,0.00061         ,0.002365      ,0.00678         ,0.014384      ,0.022586      ,0.026251      ,0.022586      ,0.014384      ,0.00678       ,0.002365      ,0.00061       ,0.000116],
    #                        [0.000055      ,0.000288      ,0.001115      ,0.003196      ,0.00678         ,0.010646      ,0.012374      ,0.010646      ,0.00678         ,0.003196      ,0.001115      ,0.000288      ,0.000055],
    #                        [0.000019      ,0.0001          ,0.000389      ,0.001115      ,0.002365      ,0.003713      ,0.004316      ,0.003713      ,0.002365      ,0.001115      ,0.000389      ,0.0001        ,0.000019],
    #                        [0.000005      ,0.000026      ,0.0001          ,0.000288      ,0.00061         ,0.000958      ,0.001114      ,0.000958      ,0.00061         ,0.000288      ,0.0001          ,0.000026      ,0.000005],
    #                        [0.000001      ,0.000005      ,0.000019      ,0.000055      ,0.000116      ,0.000183      ,0.000213      ,0.000183      ,0.000116      ,0.000055      ,0.000019      ,0.00005      ,0.000001]
    #                                                                ])



    self.gaussianMatrix=np.array([
                                                                [0.000158,	0.000608,        0.00183,	0.004312,	0.007953,	0.011483,	0.012978,	0.011483,	0.007953,	0.004312,	0.00183,	0.000608,	0.000158],
                                                                [0.000291,	0.001121,	0.003375,	0.007953,	0.014669,	0.021179,	0.023938,	0.021179,	0.014669,	0.007953,	0.003375,	0.001121,	0.000291],
                                                                [0.000421,	0.001618,	0.004873,	0.011483,	0.021179,	0.030579,	0.034561,	0.030579,	0.021179,	0.011483,	0.004873,	0.001618,	0.000421],

                                                                [0.000476,	0.001829,	0.005508,	0.012978,	0.023938,	0.034561,	0.039062,	0.034561,	0.023938,	0.012978,	0.005508,	0.001829,	0.000476],

                                                                [0.000421,	0.001618,	0.004873,	0.011483,	0.021179,	0.030579,	0.034561,	0.030579,	0.021179,	0.011483,	0.004873,	0.001618,	0.000421],
                                                                [0.000291,	0.001121,	0.003375,	0.007953,	0.014669,	0.021179,	0.023938,	0.021179,	0.014669,	0.007953,	0.003375,	0.001121,	0.000291],
                                                                [0.000158,	0.000608,	0.00183,	0.004312,	0.007953,	0.011483,	0.012978,	0.011483,	0.007953,	0.004312,	0.00183,	0.000608,	0.000158]
                                                                ])


    #        self.gaussianMatrix=np.array([
    #                                                            [0.005084,	0.009377,	0.013539,	0.015302,	0.013539,	0.009377,	0.005084],
    #                                                            [0.009377,	0.017296,	0.024972,	0.028224,	0.024972,	0.017296,	0.009377],
    #                                                            [0.013539,	0.024972,	0.036054,	0.040749,	0.036054,	0.024972,	0.013539],
    #                                                            [0.015302,	0.028224,	0.040749,	0.046056,	0.040749,	0.028224,	0.015302],
    #                                                            [0.013539,	0.024972,	0.036054,	0.040749,	0.036054,	0.024972,	0.013539],
    #                                                            [0.009377,	0.017296,	0.024972,	0.028224,	0.024972,	0.017296,	0.009377],
    #                                                            [0.005084,	0.009377,	0.013539,	0.015302,	0.013539,	0.009377,	0.005084]
    #                                                            ])




    #creates 4 different size grayscale images, of the target-to-segment image
    #for i in range(0,4):
    #GrayScale image and save it
    self.g_image.append(self.__produce_gradient_image(image, 2**0))



    self.STOP_SEARCH_FITTING=False
    self.savedBestIndicesPreviousIteration=[]
    self.totallandmarksAdjustedCounterPrev=0

    self.asm = asm
    totalNumberOfLandmarks = self.asm.shapes[0].num_pts
    self.savedBestIndicesCurrentIteration=[0]*totalNumberOfLandmarks#initialize totalNumberOfLandmarks elements to zero
    


    for imagelevel in range(self.pyramidLevels):

        imageNameDownSampled=0

        ##Save a the 3 subsampled training set images for each pyramid level. (if 3 levels then [0,1,2] level 0, [3,4,5] level 1, [6,7,8] level 2 )
        if imagelevel == 0:

            #convert image to numpy array
            gradim = np.asarray(image[:,:])#cv2.imread(cv2TocvConvertedImage)

            gradim=sobel(gradim)#gradient(gradim, 1, 1 ,5)

            #cv2 derivative image to iplimage
            source = gradim # source is numpy array
            bitmap = cv.CreateImageHeader((source.shape[1], source.shape[0]), cv.IPL_DEPTH_8U, 3)
            cv.SetData(bitmap, source.tostring(),
                       source.dtype.itemsize * 3 * source.shape[1])

            ##DENOISE ADDITION HERE CONTINUE
            #bitmap = cv2.fastNlMeansDenoising(np.asarray(bitmap[:,:]),None,10,7,21)

            #source = bitmap # source is numpy array
            #bitmap = cv.CreateImageHeader((source.shape[1], source.shape[0]), cv.IPL_DEPTH_8U, 3)
            #cv.SetData(bitmap, source.tostring(),
            #           source.dtype.itemsize * 3 * source.shape[1])
            ##DENOISE ADDITION HERE CONTINUE

            #for the first level we add to greyTargetImage the grayscale gradient (without any downsampling/upsampling)
            self.greyTargetImage.append(self.__produce_gradient_image(bitmap, 2**imagelevel))
	    print "imagelevel %d"%(imagelevel)
            #cv.WaitKey()

        elif imagelevel > 0:#scale image to pyramidlevel blur & downsample

            #if we are creating the 1 pyramid level sampling
            if imageNameDownSampled==0:
                img = cv2.imread(targetImageName,cv2.IMREAD_GRAYSCALE)

                gradim=sobel(img)#gradient(img, 1, 1 ,5)
                img=gradim
		print "imageNameDownSampled %d"%(imageNameDownSampled)
		print "1...imagelevel = %s "%(imagelevel)
                #cv.WaitKey()

            #sub-sample iteratively the original target image 'imagelevel number of times'
            for iterativeSamplingIndex in range(imagelevel):
		print "imageNameDownSampled %d"%(imageNameDownSampled)
                #cv.WaitKey()

                h,w = img.shape[:2]

                #gradim=sobel(img,1,1,5)
                #img=gradim
                #smoothed&downsampled & upsampled back
                img = cv2.pyrDown( img,dstsize = (w/2, h/2) )
                h,w = img.shape[:2]
                img = cv2.pyrUp(img,dstsize = (2*w,2*h))
		cv2.imshow("tmp_target_grey_imageDownSampled_%d.jpg"%(imagelevel),img)
                cv2.imwrite("tmp_target_grey_imageDownSampled.jpg",img)
                test_grey_image = cv.LoadImage("tmp_target_grey_imageDownSampled.jpg")
                #cv.WaitKey()

		cv2.imshow('dst',img)
		print "subsample imagelevel %d"%(imagelevel)
		print "2...imagelevel = %s "%(imagelevel)
                #cv.WaitKey()


		print 'scaled Down & up by..%d'%(2**imagelevel)
                #cv.WaitKey()


            ##DENOISE ADDITION HERE CONTINUE
            #test_grey_image = cv2.fastNlMeansDenoising(np.asarray(test_grey_image[:,:]),None,10,7,21)

            #source = test_grey_image # source is numpy array
            #test_grey_image = cv.CreateImageHeader((source.shape[1], source.shape[0]), cv.IPL_DEPTH_8U, 3)
            #cv.SetData(test_grey_image, source.tostring(),
            #           source.dtype.itemsize * 3 * source.shape[1])
            ##DENOISE ADDITION HERE CONTINUE


            #imagelevel doesn't do much here apart from numbering the pyramidlevelImages grey_imageToBeScaled_level_1,2,3 etc.
            self.greyTargetImage.append(self.__produce_gradient_image(test_grey_image, 2**imagelevel))
	    print 'test_grey_image downsampled added to greyTargetImage Vector'
            #cv.WaitKey()


        #self.greyTargetImage.append(self.__produce_gradient_image(image, 2**i))


    print "target image vector of pyramid levels to use while image-searching %d"%len(self.greyTargetImage)
    #cv.WaitKey()

    ##print "G_IMAGE="
    width, height = cv.GetSize(self.g_image[scale])
    ##print width,height
    #####cv.WaitKey()

    '''transform the mean shape manually to match the orientation of an image (in this case) grey_image_3.jpg'''

    meanshapeVec=Shape.from_vector(asm.mean)
    sumx=0
    sumy=0
    for i, pt in enumerate(meanshapeVec.pts):
        sumx+=pt.x
        sumy+=pt.y
    sumx/=len(meanshapeVec.pts)
    sumy/=len(meanshapeVec.pts)
    CenterOfMass = Point(sumx,sumy)
    degreesOfRotation=0

    #degreesOfRotation=0
    #CenterOfMass=Point(0,0)
    if CenterOfMass.__eq__(Point(0,0)):
        t= Point(0,0)

    #self.asm = asm
    # Copy mean shape as starting shape and transform it to where the target shape to detect is manually (later on with template matching)
    ###print "asm.mean=%s"%(asm.mean)
    #####cv.WaitKey()
    self.shape = Shape.from_vector(asm.mean).transform(t, degreesOfRotation, CenterOfMass)
    ###print "asm.mean transformed=%s"%(self.shape.pts)
    #####cv.WaitKey()


    '''now modifly each of the asm shapes as well, based on the desired transform'''

    '''the original shapes read, not the asm shapes. Before any transform applied to them, for use in the getCorrectedLandmarkPoint'''
    self.originalShapes = originalShapes


    '''
    testImageNameCounter=0
    #create a save original shapes as marked on original images
    for s in (originalShapes):

        #convert this test_grey_image to grayscale
        greyImage = []
        testImageNameCounter=testImageNameCounter+1
        #so for shape 1..load  gray_image_1, for shape 2..load  gray_image_2 etc

        test_grey_image = cv.LoadImage("grey_image_"+str(testImageNameCounter)+".jpg")
        currentImageName="grey_image_"+str(testImageNameCounter)+".jpg"
        #greyImage will have the greyscale image marked with landmarks
        greyImage.append(self.__produce_gradient_image(test_grey_image, 2**0))

        #create a copy of the greyscaled image to put the new landmarks onto
        trainingSetImage = cv.CreateImage(cv.GetSize(greyImage[0]), greyImage[0].depth, 1)
        cv.Copy(greyImage[0], trainingSetImage)

        for i,pt in enumerate(s.pts):
	    ##print "%d, %s"%(i,pt)

            cv.Circle(trainingSetImage, ( int(pt.x), int(pt.y) ), 4, (100,100,100))
            cv.NamedWindow("TEST", cv.CV_WINDOW_NORMAL)
	    cv.ShowImage("trainingSetImage",trainingSetImage)

            ####cv.WaitKey()
    '''






    '''Transform the shape to where the target vertebra is / should be done with Template Matching'''
    shapecounter=0
    #now modifly each of the asm shapes as well, based on the desired transform
    for shape in self.asm.shapes:
	###print "self.shapes[0].pts=%s"%(shape.pts[0])
        #tmpShapeToTransform=shape.pts
        #shape.pts[0]=Point(1,1)
        #shape = np.array([shape.get_vector()])

        shapeCoordinatesList=[]
        for i, pt in enumerate(shape.pts):
            shapeCoordinatesList.append([pt.x,pt.y])

        shapeCoordinatesList = np.array(shapeCoordinatesList)
	###print "shapeCoordinatesList=%s"%(shapeCoordinatesList)

        s= shapeCoordinatesList.flatten()
	###print "shapeCoordinatesList=%s"%(s)

	##print "%d, s=%s points each vector"%(shapecounter,len(s))
        ####cv.WaitKey()

        tmpShapeManuallyTransformed=Shape.from_vector(s).transform(t,CenterOfMass)
	###print "tmpShapeManuallyTransformed=%s"%(tmpShapeManuallyTransformed.pts)
        #####cv.WaitKey()


	###print "asm.shapes[%d]=%s"%(shapecounter,asm.shapes[shapecounter].pts)
        #####cv.WaitKey()

        #now transform each asm read in shape (manually) to where the target vertebra in the image target is (just to visualize it there ..it may be placed based on tracking at a later stage or by an somehow automated procedure)
        shapePoints=asm.shapes[shapecounter].pts
        transformShapePoints=tmpShapeManuallyTransformed.pts
        for i in range(len(shapePoints)):
	    ###print "before=%s"%(shapePoints[i])
            #####cv.WaitKey()

            shapePoints[i].x=transformShapePoints[i].x
            shapePoints[i].y=transformShapePoints[i].y
	    ###print "after=%s"%(shapePoints[i])
            #####cv.WaitKey()

	###print "asm.shapes[%d] transformed=%s"%(shapecounter,asm.shapes[shapecounter])
        #####cv.WaitKey()

        shapecounter+=1




    ###print "self.asm.shapes[0][0]=%s"%self.asm.shapes[0][0]
    #####cv.WaitKey()

    ##print "scale=%d"%(scale)
    #####cv.WaitKey()

    ShapeBeforeResizeToFitImage = cv.CreateImage(cv.GetSize(self.image), self.g_image[scale].depth, 1)
    cv.Copy(self.g_image[scale], ShapeBeforeResizeToFitImage)
    for i, pt in enumerate(self.shape.pts):
        cv.Circle(ShapeBeforeResizeToFitImage, ( int(pt.x), int(pt.y) ), 2, (100,100,100))
    cv.NamedWindow("ShapeBeforeResizeToFitImage", cv.CV_WINDOW_NORMAL)
    cv.ShowImage("ShapeBeforeResizeToFitImage",ShapeBeforeResizeToFitImage)
    #####cv.WaitKey()


    # And resize shape to fit image if required
    if self.__shape_outside_image(self.shape, self.image):
      self.shape = self.__resize_shape_to_fit_image(self.shape, self.image)

    #replace init mahalanobis


    ##########################################################################################
    ##########################################################################################

    '''
    covariance_xy = np.cov(x,y, rowvar=0)
    inv_covariance_xy = np.linalg.inv(covariance_xy)
    xy_mean = np.mean(x),np.mean(y)
    x_diff = np.array([x_i - xy_mean[0] for x_i in x])
    y_diff = np.array([y_i - xy_mean[1] for y_i in y])
    diff_xy = np.transpose([x_diff, y_diff])

    md = []
    for i in range(len(diff_xy)):
        md.append(np.sqrt(np.dot(np.dot(np.transpose(diff_xy[i]),inv_covariance_xy),diff_xy[i])))
    ####print "mahalanobis distance between all g profiles of this landmark and the g mean profile is :%s"%(md)
    '''

    #X = np.vstack((IthImage_NormalizedtmpLandmarkDerivativeIntensityVector[i]), np.array(y_j_mean[0])))
    #covMat=np.cov( IthImage_NormalizedtmpLandmarkDerivativeIntensityVector[i] , np.array(y_j_mean[0]) )

    #####print "covariance matrix of  =%s"%(covMat)
    #######cv.WaitKey()

    #calculate for the ith image , each jth landmark which profile g fits best (based on mahalanobis distance measurement)
    #when done with all landmarks of this ith image..then move onto the next image to fin best fits
    #covMatInverse = np.linalg.inv(covMat)
    #mahalanobisDist = scipy.spatial.distance.mahalanobis(  np.array(IthImage_NormalizedtmpLandmarkDerivativeIntensityVector[i]).flatten(), np.array(y_j_mean[0]).flatten(), covMatInverse )
    #mahalanobisDist = scipy.spatial.distance.mahalanobis(  np.array(IthImage_NormalizedtmpLandmarkDerivativeIntensityVector[i]).flatten(), np.array(y_j_mean[0]).flatten(), covMatInverse )

    #####print "mahalanobis=%s"%(mahalanobisDist)
    ######cv.WaitKey()
    '''
    for ithImageLandmarksVector in IthImage_NormalizedtmpLandmarkDerivativeIntensityVector:#3 images x 30 landmarks each x 6 normalizedDerivativeProfiles..so 450 element vector
        landmarkAtWhiskerXMeanIndex=0
        for yij_ithLandmarkVector in ithImageLandmarksVector:#1 images x 30 landmarks x 6 normalizedDerivativeProfiles..so 180 element vector
            yj_mean=githLandmarkMeanAcrossAllImagesVector[landmarkAtWhiskerXMeanIndex]
	    ####print "Calculate the Covariance Matrix between\n: yij_ithLandmarkVector=%s and yj_mean[%d]=%s \n"%(np.array(yij_ithLandmarkVector),landmarkAtWhiskerXMeanIndex, yj_mean)
            covMat=np.cov(np.array(yij_ithLandmarkVector),yj_mean)
            landmarkAtWhiskerXMeanIndex+=1
	    ####print "covMat=%s"%(covMat)
            #######cv.WaitKey()
            covMatDeterminant=np.linalg.det(covMat)
	    ####print "covMatDeterminant=%s"%(covMatDeterminant)
            #######cv.WaitKey()

            covMatInverse = np.linalg.inv(covMat)
            #mahalanobisDist = scipy.spatial.distance.mahalanobis(  yij_ithLandmarkVector, yj_mean, covMatInverse )
	    #####print "mahalanobis=%s"%(mahalanobisDist)
            #######cv.WaitKey()
    '''


    '''
    ####print "Calculate the Covariance Matrix between\n: IthImage_NormalizedtmpLandmarkDerivativeIntensityVector[0][0]=%s and githLandmarkMeanAcrossAllImagesVector[0]=%s \n"%(np.array(IthImage_NormalizedtmpLandmarkDerivativeIntensityVector[0][0]),githLandmarkMeanAcrossAllImagesVector[0])
    covMat=np.cov(np.array(IthImage_NormalizedtmpLandmarkDerivativeIntensityVector[0][0]),githLandmarkMeanAcrossAllImagesVector[0])
    ####print "covMat=%s"%(covMat)
    ######cv.WaitKey()
    covMatDeterminant=np.linalg.det(covMat)
    ####print "covMatDeterminant=%s"%(covMatDeterminant)
    ######cv.WaitKey()
    '''



    '''
        #Now Time for the derivative of this currentLandmarkProfiles vector

        #(equation 12 of the 'Subspace Methods for Pattern Recognition in Intelligent Environment' book)
        normalizedBySumofDifference=0
        for intensity in range(len(currentLandmarkProfiles)-1):

            #calculate derivative profile intensity g_image[i] - g_image[i-1]...and so on.... dg vector
            #....................
            #....................
            difference=currentLandmarkProfiles[intensity+1]-currentLandmarkProfiles[intensity]

	    ####print "%f-%f .. difference=%f"%(currentLandmarkProfiles[intensity+1], currentLandmarkProfiles[intensity], difference)

            #store derivative gray-level profile vector of all whisker point (-3 points on the one side  +landmark Itseflf + 3 points on the other side
            tmpLandmarkDerivativeIntensityVector.append( difference )

            normalizedBySumofDifference=normalizedBySumofDifference+difference

	####print "tmpLandmarkDerivativeIntensityVector: %s"%(tmpLandmarkDerivativeIntensityVector)
	####print "normalizedBy: %s"%(normalizedBySumofDifference)
        #######cv.WaitKey()

        #(equation 13 of the 'Subspace Methods for Pattern Recognition in Intelligent Environment' book)
        #normalize tmpProfileDerivativeIntensityVector
        NormalizedtmpLandmarkDerivativeIntensityVector=[]
        for x in tmpLandmarkDerivativeIntensityVector:
            if normalizedBySumofDifference!=0:
                x = x / normalizedBySumofDifference
                NormalizedtmpLandmarkDerivativeIntensityVector.append(x)
            else:
		####print "tmpLandmarkDerivativeIntensityVector is: %s\n"%(tmpLandmarkDerivativeIntensityVector)
                x = x / 1
                NormalizedtmpLandmarkDerivativeIntensityVector.append(x)
                ######cv.WaitKey()

	####print "NormalizedtmpLandmarkDerivativeIntensityVector %s\n"%(NormalizedtmpLandmarkDerivativeIntensityVector) ###.. must be for this point along the whisker
        #######cv.WaitKey()

        #now store this Normalized tmp Landmark Derivative Intensity Vector for this image for THIS landmark
        # and move onto caclulating the NormalizedtmpLandmarkDerivativeIntensityVector for this image for the NEXT landmark

        #This vector contains : for each image, all normalized derivative profile of ALL ith's image's landmarks
	####print "image=%d\n"%(testImageNameCounter-1)
        IthImage_NormalizedtmpLandmarkDerivativeIntensityVector[testImageNameCounter-1].append(NormalizedtmpLandmarkDerivativeIntensityVector)
	####print "Here the following vector is filled  with the  NormalizedtmpLandmarkDerivativeIntensityVector \nor each of the landmarks for this ith image in the training set"
	####print "IthImage_NormalizedtmpLandmarkDerivativeIntensityVector %s\n"%(IthImage_NormalizedtmpLandmarkDerivativeIntensityVector[testImageNameCounter-1])
        #######cv.WaitKey()
	####print "Total %s's landmarks tested=%d"%(currentImageName,len(IthImage_NormalizedtmpLandmarkDerivativeIntensityVector[testImageNameCounter-1]))
        totalLandmarksLeftToBeTested-=1
	####print "now %d landmarks of %s are left to be tested..before we move the next image's landmark points\n\n"%(totalLandmarksLeftToBeTested, currentImageName)
        #######cv.WaitKey()

        #The IthImage_NormalizedtmpLandmarkDerivativeIntensityVector has been updated with IthImag's landmark profile values
	####print "The IthImage_NormalizedtmpLandmarkDerivativeIntensityVector has been updated with %s's landmark profile values:\n\n%s\n"%(currentImageName,IthImage_NormalizedtmpLandmarkDerivativeIntensityVector)
    '''

    '''
    ####print "The IthImage_NormalizedtmpLandmarkDerivativeIntensityVector has been updated with %d image point shapes: \n\n%s\n"%(len(self.asm.shapes),IthImage_NormalizedtmpLandmarkDerivativeIntensityVector)
    #######cv.WaitKey()

    #We are going to iterate all image's vectors and sum the corresponding landmark vectors to. Every one on the top of the preexisting other vector.
    sumOflandMarksProfileAccrossAllImages=[]
    testImageNameCounter=0
    landmarkCounter=0
    for normalizedDerivativeProfileLandmarksVector in IthImage_NormalizedtmpLandmarkDerivativeIntensityVector:

        #assign the correct name to the image
        testImageNameCounter+=1
        landmarkCounter=0
        currentImageName="grey_image_"+str(testImageNameCounter)+".jpg"

	####print "%s's landmarks' profiles=\n\n%s\n\n"%(currentImageName,normalizedDerivativeProfileLandmarksVector)
        ######cv.WaitKey()

        for landmarkNormalizedDerivativeProfile in normalizedDerivativeProfileLandmarksVector:
            landmarkCounter+=1
	    ####print "%s's landmark profile%d=%s\n\n"%(currentImageName,landmarkCounter,landmarkNormalizedDerivativeProfile)
            ######cv.WaitKey()

            #calculate g-mean for this ith landmark of this ith image by iterating through all values of g vector
            #for i in landmarkNormalizedDerivativeProfile:
            #modified

            if testImageNameCounter==1:#the first time just append all 30 landmark related vectors
                sumOflandMarksProfileAccrossAllImages.append(np.array(landmarkNormalizedDerivativeProfile))
		####print "sumOflandMarksProfileAccrossAllImages=%s"%(sumOflandMarksProfileAccrossAllImages)
		####print "shit happened"
            else:#then just update each of the vectors' values
                sumOflandMarksProfileAccrossAllImages[landmarkCounter-1]=np.array(sumOflandMarksProfileAccrossAllImages[landmarkCounter-1]) + (np.array(landmarkNormalizedDerivativeProfile))
		####print "sumOflandMarksProfileAccrossAllImages=%s"%(sumOflandMarksProfileAccrossAllImages)
		####print "all good"


    #Now caclulate the mean for each one (eq. 14 of the 'Subspace Methods for Pattern Recognition in Intelligent Environment' book )
    #and put it in a vector
    githLandmarkMeanAcrossAllImagesVector=[]
    for sumYith in sumOflandMarksProfileAccrossAllImages:
        githLandmarkMeanAcrossAllImagesVector.append(sumYith / len(self.asm.shapes))#so we end up calculating 30 mean (6 element) vectors, which are each landmarks' mean across all images' in the training set
	####print "githLandmarkMeanAcrossAllImagesVector=\n\n%s\n\n"%(githLandmarkMeanAcrossAllImagesVector)
        #######cv.WaitKey()
    ######cv.WaitKey()

    ####print "githLandmarkMeanAcrossAllImagesVector=\n\n%s\n\n"%(githLandmarkMeanAcrossAllImagesVector)
    ######cv.WaitKey()
    ####print "IthImage_NormalizedtmpLandmarkDerivativeIntensityVector[0][0].size=%d elements"%(len(IthImage_NormalizedtmpLandmarkDerivativeIntensityVector[0][0]))
    ######cv.WaitKey()
    ####print "githLandmarkMeanAcrossAllImagesVector.size=%d elements"%(len(githLandmarkMeanAcrossAllImagesVector[0]))
    ######cv.WaitKey()

    '''

    #find  covariance matrix
    '''
    We want to find the covariance matrix with minimum volume encompassing some % of the data set.
    Let us assume we have a set of observations X = {x1, x2, ... xN} of size N.  Further let us assume 10% of samples are outliers.
    One could construct a brute-force algorithm as follows:

    determine all unique sample subsets of size h = 0.9N
    for each subset S
    compute the covariance matrix of Cs = S
    compute Vs = det(Cs)
    choose the subset where Vs is a minimal
    Geometrically, the determinant is the volume of the N dimensional vector space implied by the covariance matrix.  Minimizing the ellipsoid is equivalent to minimizing the volume.
    '''

    '''
    for ithImageLandmarksVector in IthImage_NormalizedtmpLandmarkDerivativeIntensityVector:#3 images x 30 landmarks each x 6 normalizedDerivativeProfiles..so 450 element vector
        landmarkAtWhiskerXMeanIndex=0
        for yij_ithLandmarkVector in ithImageLandmarksVector:#1 images x 30 landmarks x 6 normalizedDerivativeProfiles..so 180 element vector
            yj_mean=githLandmarkMeanAcrossAllImagesVector[landmarkAtWhiskerXMeanIndex]
	    ####print "Calculate the Covariance Matrix between\n: yij_ithLandmarkVector=%s and yj_mean[%d]=%s \n"%(np.array(yij_ithLandmarkVector),landmarkAtWhiskerXMeanIndex, yj_mean)
            covMat=np.cov(np.array(yij_ithLandmarkVector),yj_mean)
            landmarkAtWhiskerXMeanIndex+=1
	    ####print "covMat=%s"%(covMat)
            #######cv.WaitKey()
            covMatDeterminant=np.linalg.det(covMat)
	    ####print "covMatDeterminant=%s"%(covMatDeterminant)
            #######cv.WaitKey()

            covMatInverse = np.linalg.inv(covMat)
            #mahalanobisDist = scipy.spatial.distance.mahalanobis(  yij_ithLandmarkVector, yj_mean, covMatInverse )
	    #####print "mahalanobis=%s"%(mahalanobisDist)
            #######cv.WaitKey()

    '''


    '''
    ####print "Calculate the Covariance Matrix between\n: IthImage_NormalizedtmpLandmarkDerivativeIntensityVector[0][0]=%s and githLandmarkMeanAcrossAllImagesVector[0]=%s \n"%(np.array(IthImage_NormalizedtmpLandmarkDerivativeIntensityVector[0][0]),githLandmarkMeanAcrossAllImagesVector[0])
    covMat=np.cov(np.array(IthImage_NormalizedtmpLandmarkDerivativeIntensityVector[0][0]),githLandmarkMeanAcrossAllImagesVector[0])
    ####print "covMat=%s"%(covMat)
    ######cv.WaitKey()
    covMatDeterminant=np.linalg.det(covMat)
    ####print "covMatDeterminant=%s"%(covMatDeterminant)
    ######cv.WaitKey()
    '''



    '''
    #LandMark X Derivatives Vector Out of All Images In The Training Set
    LandMarkX_DerivativesVectorSum=nparray()

    imageCounter=0
    landmarkCounter=0
    # Loop over rows. (each tested image)
    for row in IthImage_NormalizedtmpLandmarkDerivativeIntensityVector:
        imageCounter=imageCounter+1
	####print "Training Image:%d"%(imageCounter)

        # Loop over columns. (each landmark's normalized derivative profile)
        landmarkCounter=0
        for column in row:
            landmarkCounter=landmarkCounter+1
	    ####print "derivative profile: landmark %d = %s"%(landmarkCounter,column)

            #######cv.WaitKey()
	####print("\n")
    ######cv.WaitKey()




    imageCounter=0
    # Loop over rows. (each tested image)
    for row in IthImage_NormalizedtmpLandmarkDerivativeIntensityVector:
        imageCounter=imageCounter+1
	####print "Training Image:%d"%(imageCounter)

        # Loop over columns. (each landmark's normalized derivative profile)
        landmarkCounter=0
        for column in row:
            landmarkCounter=landmarkCounter+1
	    ####print "derivative profile: landmark %d = %s"%(landmarkCounter,column)

            #create an element for each landmark added,
            LandMarkX_DerivativesVectorSum[landmarkCounter]=LandMarkX_DerivativesVectorSum[LandMarkX_DerivativesVector] + column
	    ####print

            #######cv.WaitKey()
	####print("\n")

    '''


#####print ("x=%d,y=%d")%(x,y)
#Gives the landmark points of 1 image out of the training set
#####print "Gives the landmark points of this-each image out of the training set"
#####print ("oneTrainingImageVector=%s")%(oneTrainingImageVector.pts)
#######cv.WaitKey()






    '''

    #im_array = np.asarray( self.g_image[scale][:,:] )

    #for each mean shape points


    #Calculate their mean g profile and covariance matreix
    #Gmean = Shape([])


    for i, pt in enumerate(self.shape.pts):

                #create a list of Shapes - the g profiles for each landmark point
                currentLandmarkProfiles = []

                #store point
                p = self.shape.pts[i]

                #get normal to the point
                norm = self.shape.get_normal_to_point(i)

		####print "\n\n\n\n\n  !!!!!!!!!!!!!!!!!!!!!! New landmark point mahalanobis calculation !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

                currentLandmarkProfilesNormalizedDerivativesVector=[]

                #along normal (whisker)
                for t in drange(-3, 3, 1):
                        # Normal to normal...
			#####print "norm",norm


			#####print "p",(p.x,p.y)
			#####print "new_p",(new_p)
                        tmpProfileIntensityVector = []
                        tmpProfileDerivativeIntensityVector = []

                        # Look 6 pixels to each side along the whisker normal too (seach profile)
                        for side in range(-6,6):#tangent width

                                new_p = Point(p.x + side*-norm[1], p.y + side*norm[0])#why ..the other way around? best way to form the search window 12 x 7 = 84 pixels wide window

                                x = int((norm[0]*t + new_p.x))#*math.sin(t*(math.pi/180)))
                                y = int((norm[1]*t + new_p.y))#*math.cos(t*(math.pi/180)))

                                #(equation 11 of the 'Subspace Methods for Pattern Recognition in Intelligent Environment' book)

                                #add to g (the inensity value of next pixel along the perpendicular to the whisker.. search profile)
                                gradientIntensity=self.g_image[scale][y-1,x-1]

                                #store gray-level profile in tmpProfileIntensityVector ..g vector
                                tmpProfileIntensityVector.append(gradientIntensity)

			####print "tmpProfileIntensityVector:%s"%tmpProfileIntensityVector


                        #(equation 12 of the 'Subspace Methods for Pattern Recognition in Intelligent Environment' book)
                        normalizedBy=0
                        for intensity in range(len(tmpProfileIntensityVector)-1):

                                #calculate derivative profile intensity g_image[i] - g_image[i-1]...and so on.... dg vector
                                #....................
                                #....................
                                difference=tmpProfileIntensityVector[intensity+1]-tmpProfileIntensityVector[intensity]

				####print "%d-%d"%(tmpProfileIntensityVector[intensity+1],tmpProfileIntensityVector[intensity])

                                #store derivative gray-level profile vector
                                tmpProfileDerivativeIntensityVector.append( difference )
                                CovarianceMatricesVec
                                normalizedBy=normalizedBy+difference

			####print "tmpProfileDerivativeIntensityVector: %s"%(tmpProfileDerivativeIntensityVector)
			####print "normalizedBy: %s"%(normalizedBy)

                        #(equation 13 of the 'Subspace Methods for Pattern Recognition in Intelligent Environment' book)
                        #normalize tmpProfileDerivativeIntensityVector
                        NormalizedTmpProfileDerivativeIntensityVector=[]
                        for x in tmpProfileDerivativeIntensityVector:
                                if normalizedBy!=0:
                                        x = x / normalizedBy
                                        NormalizedTmpProfileDerivativeIntensityVector.append(x)
                                else:
                                        x = x / 1
                                        NormalizedTmpProfileDerivativeIntensityVector.append(x)

			####print "NormalizedTmpProfileDerivativeIntensityVector %s\n"%(NormalizedTmpProfileDerivativeIntensityVector)
                        cv.WaitKey(10)


                        #fill in each normalized derivative profile vector for this landmark and create the currentLandmarkProfilesNormalizedDerivativesVector
                        #essentially containing all normalized derivative profiles for current landmark point, of which we should calculate the mean

                        currentLandmarkProfilesNormalizedDerivativesVector.append(NormalizedTmpProfileDerivativeIntensityVector)
			####print "currentLandmarkProfilesNormalizedDerivativesVector total: %s"%(len(currentLandmarkProfilesNormalizedDerivativesVector))
			####print "currentLandmarkProfilesNormalizedDerivativesVector: %s"%(currentLandmarkProfilesNormalizedDerivativesVector)

                        #(equation 14 of the 'Subspace Methods for Pattern Recognition in Intelligent Environment' book)

                # now sum all currentLandmarkProfilesNormalizedDerivativesVector elements together and divide by len(currentLandmarkProfilesNormalizedDerivativesVector)
                # which is the total number of dormalized derivative profile vectors for this landmark point
                #/////////////currentLandmarkProfilesNormalizedDerivativesVector/len(currentLandmarkProfilesNormalizedDerivativesVector)
        '''

    '''
			#####print "Gmean",Gmean.pts
                        currentLandmarkProfiles.append(tmpProfileIntensityVector)

                for j in range(len(currentLandmarkProfiles)):
			####print "profile %d with %d points: %s"%(j, len(currentLandmarkProfiles[j]), currentLandmarkProfiles[j])
                cv.WaitKey(1000)
         '''




		#####print "total profiles",len(profiles)
                #cv.WaitKey(1000)
                #now calculate the derivative profile dg

    '''
                for i in range(len(profiles)):
			####print "i",profiles[0].pts
                        ######cv.WaitKey()
    '''

                #now calculate the meanG profile for this landmark , as well as the covariance matrix for this landmark



		#####print "profile number:",len(profiles)
                #cv.WaitKey(100)




    ###################################################
    ###################################################


  def __shape_outside_image(self, s, i):
    for p in s.pts:
      if p.x > i.width or p.x < 0 or p.y > i.height or p.y < 0:
        return True
    return False

  def __resize_shape_to_fit_image(self, s, i):
    # Get rectagonal boundary of shape
    min_x = min([pt.x for pt in s.pts])
    min_y = min([pt.y for pt in s.pts])
    max_x = max([pt.x for pt in s.pts])
    max_y = max([pt.y for pt in s.pts])

    # If it is outside the image then we'll translate it back again
    #ex. if minX or minY point of init shape is beyond image's width
    #ratioX & rationY is adjusted accordingly
    if min_x > i.width: min_x = 0
    if min_y > i.height: min_y = 0
    ratio_x = (i.width-min_x) / (max_x - min_x)
    ratio_y = (i.height-min_y) / (max_y - min_y)
    new = Shape([])
    for pt in s.pts:
      new.add_point(Point(pt.x*ratio_x if ratio_x < 1 else pt.x,  pt.y*ratio_y if ratio_y < 1 else pt.y))
    return new

  '''
      This function takes an input images, and grayscale's them!
      Addiitonally, it saves out different target-image downsampled pyramid levels under pyramidLevelImages folder
  '''
  def __produce_gradient_image(self, i, scale):

        #old way of producing gradient
    '''
        #get the size of the original image
    size = cv.GetSize(i)

    #create an image which will be the grayscaled of the original
    grey_image = cv.CreateImage(size, 8, 1)

    size = [s/scale for s in size]

    grey_image_small = cv.CreateImage(size, 8, 1)


    cv.CvtColor(i, grey_image, cv.CV_RGB2GRAY)

    df_dx = cv.CreateImage(cv.GetSize(i), cv.IPL_DEPTH_16S, 1)#create a 16 bit image to avoid overflow while performing sobel filtering on it
    cv.Sobel( grey_image, df_dx, 1, 1)
    #cv2.Sobel( grey_image, 8, 1, 1, df_dx)
    cv.Convert(df_dx, grey_image)#make it an 8 bit image
    cv.Resize(grey_image, grey_image_small)#, interpolation=cv.CV_INTER_NN)
    cv.Resize(grey_image_small, grey_image)#, interpolation=cv.CV_INTER_NN)

    return grey_image
    '''

    #get the size of the original image
    width, height = cv.GetSize(i)
    ####print "width=%d,height=%d"%(width,height)

    #Save out the image and move on to the new cv2 way of doing things so as to gray scale it
    cv.SaveImage("InputToGrayScale.jpg", i)
    InputToGrayScaleImg=cv2.imread("InputToGrayScale.jpg")#,cv2.IMREAD_GRAYSCALE)
    #######cv.WaitKey()

    grey_image = np.zeros((width,height,1), np.uint8)


    #cv.WaitKey(1000)

    width=width/scale
    height=height/scale

    grey_image_small = np.zeros((width,height,1), np.uint8)

    #CONVERTS INPUT IMAGE TO GRAYSCALE
    grey_image = cv2.cvtColor(InputToGrayScaleImg, cv2.COLOR_BGR2GRAY)

    #cv.Resize(grey_image, grey_image_small)#, interpolation=cv.CV_INTER_NN)
    #cv.Resize(grey_image_small, grey_image)#, interpolation=cv.CV_INTER_NN)
    #grey_image = cv2.GaussianBlur(grey_image,(3,3),0)

    #####print grey_image[0][:]
    '''
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    # Gradient-X
    grad_x = cv2.Sobel(grey_image,ddepth,1,0,ksize = 3, scale = 0.5, delta = delta,borderType = cv2.BORDER_DEFAULT)

    # Gradient-Y
    grad_y = cv2.Sobel(grey_image,ddepth,0,1,ksize = 3, scale = 0.5, delta = delta, borderType = cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)   # converting back to uint8
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    dst = cv2.addWeighted(abs_grad_x,0.5,abs_grad_y,0.5,0)



    cv2.imwrite("grey_image.jpg",dst)
    cv2.imshow("grey_image.jpg", dst)
    '''

    cv2.imwrite("grey_image.jpg",grey_image)
    cv2.imshow("grey_image.jpg", grey_image)



    '''
    #Contrast Streching
    constrastStrechedimage = cv2.imread("grey_image.jpg")
    constrastStrechedimage = cv2.cvtColor(constrastStrechedimage, cv2.COLOR_BGR2GRAY)
    constrastStrechedimage = cv2.equalizeHist(constrastStrechedimage)
    #constrastStrechedimage = cv2.cvtColor(constrastStrechedimage, cv2.COLOR_GRAY2BGR)
    cv2.imshow("equalizeHist", constrastStrechedimage)
    cv2.imwrite("grey_image-equalizeHist-Contrast-Streched.jpg",constrastStrechedimage)
    '''

    size=cv.GetSize(i)

    grey_imageFor = cv.CreateImage(size, 8, 1)
    grey_imageFormattedToOldCvVersion = cv.LoadImage("grey_image.jpg")

    #make sure we convert it to grayscale too
    cv.CvtColor(grey_imageFormattedToOldCvVersion, grey_imageFor, cv2.COLOR_BGR2GRAY)

    #save and
    cv.SaveImage("pyramidLevelImages/grey_imageToBeScaled_level_%d.jpg"%(scale),grey_imageFor )

    #return it
    return grey_imageFor


  def calculateIthImageLandmarkDerivIntensityVec(self, p, norm, greyImage, gaussianMatrix, _searchProfiles, IthImage_NormalizedtmpLandmarkDerivativeIntensityVector, gaussianWindow):

      scale=0
      ##print 'greyImage.channels=%s'%(greyImage.channels)

      #c=###cv.WaitKey()
      #if c==1048603 :#whichever integer key code makes the app exit
      #    exit()

      #along the whisker
      for side in range(-3,4):#along normal profile
          # Normal to normal...
	  #####print"norm",norm

          #CHANGE BACK to the FOLLOWING LINE if it does not work
          #new_p = Point(p.x + side*-norm[1], p.y + side*norm[0])#why ..the other way around? best way to form the search window 12 x 7 = 84 pixels wide window
          new_p = Point(p.x + side*norm[0], p.y + side*norm[1])


	  #####print"p",(p.x,p.y)
	  #####print"new_p",(new_p)

          tmpProfileIntensityVector = []
          tmpProfileDerivativeIntensityVector = []



          #for t in drange(min_t, max_t, (max_t-min_t)/l):

          # Look 6 pixels to each side too
          for t in drange(-6, 7, 1):#tangent MATRIX CHANGED

              #horizontalProfileFormed = Shape([])
              #counter=1


              #distributed = drange(-search if -search > min_t else min_t, search if search < max_t else max_t , 1)
              #for t in distributed:

              #counter=counter+1

              #CHANGE BACK the FOLLOWING 2 LINES if it does not work
              #x = int((norm[0]*t + new_p.x))#*math.sin(t*(math.pi/180)))
              #y = int((norm[self.shape1]*t + new_p.y))#*math.cos(t*(math.pi/180)))
              x = int((new_p.x + t*-norm[1]))#*math.sin(t*(math.pi/180)))
              y = int((new_p.y + t*norm[0]))#*math.cos(t*(math.pi/180)))



              cv.Circle(gaussianWindow, ( int(x), int(y) ), 2, (255,0,50))
              cv.NamedWindow("gaussianWindow", cv.CV_WINDOW_NORMAL)
	      cv.ShowImage("gaussianWindow",gaussianWindow)

              #c=###cv.WaitKey()
              #if c==1048603 :#whichever integer key code makes the app exit
              #    exit()



              #(equation 11 of the 'Subspace Methods for Pattern Recognition in Intelligent Environment' book)
              scale=0




              #http://stackoverflow.com/questions/13104161/fast-conversion-of-iplimage-to-numpy-array
              gradimg = np.asarray(greyImage[:,:])#cv2.imread(cv2TocvConvertedImage)
	      cv2.imshow('greyImage - down/up sampled',gradimg)






              #add to g (the intensity value of next pixel along the perpendicular to the whisker.. search profile)

              #CHECK IF ..GETS PASSED THEY IMAGE BORDERS
              #width, height=cv.GetSize(greyImage)
	      #print "width=%d"%(width)
	      #print "height=%d"%(height)
	      #print x,y

	      #print greyImage.width, greyImage.height
	      #print y,x
              #cv.WaitKey(0)
              #if c == 1048603:
              #    exit()
              gradientIntensity=greyImage[y,x] * self.gaussianMatrix[t+6][side+3] #MATRIX CHANGED

              #Add weights properly based on gaussian distribution generated matrix
	      ####print"gaussianMatrix[%d][%d]=%s gaussian matrix component was multiplied with.."%(t+6,side+3,gaussianMatrix[t+6][side+3])
	      ####print"greyImage[%d][%d,%d]=%s"%(scale,y,x,greyImage[scale][y,x])

              #c=cv.WaitKey(1)
	      #####print"EXIT when ESC keycode is pressed=%d"%(c)
              #if c == 1048603:
              #    exit()

              #store gray-level profile in tmpProfileIntensityVector ..g vector
              tmpProfileIntensityVector.append(gradientIntensity)
	      ####print"tmpProfileIntensityVector=%f"%(tmpProfileIntensityVector[-1])

	  ####print"point =%d along the whisker:\n , tmpProfileIntensityVector:%s\n"%(side,tmpProfileIntensityVector)

          '''store g# intensity profile, there are 7 for each point'''
          _searchProfiles.append(tmpProfileIntensityVector)
	  #print"_searchProfiles =%s\n"%(_searchProfiles)
          #cv.WaitKey()


          #Landmark Search Profile with Weights is calculated
	  ####print"Landmark Search Profile with Weights was calculated\n"
	  ####print"tmpProfileIntensityVector:%s\n"%(tmpProfileIntensityVector)
          #######cv.WaitKey()

          '''
          sumGith=0
          averageGith=0
          #now average each gith and save it to currentLandmarkProfiles vector
          for i in tmpProfileIntensityVector:
              sumGith=sumGith + i
	  ####print"sumGith :%f\n"%(sumGith)
          #######cv.WaitKey()


          #save the averageGith in currentLandmarkProfiles vector, which corresponds to this point along the whisker
          averageGith=sumGith/len(tmpProfileIntensityVector)
	  ####print"averageGith=%s"%(averageGith)
          #######cv.WaitKey()
          currentLandmarkProfiles.append(averageGith)
	  ####print"currentLandmarkProfiles=%s"%(currentLandmarkProfiles)
          #######cv.WaitKey()
          '''
      #find g mean profile vector for this landmark
      g_mean=[]

      #PROBABLY NOT NEEDED THE FOLLOWING

      ##print"_searchProfiles=%s"%(len(_searchProfiles))
      ####cv.WaitKey()
      for profile in _searchProfiles:
          if not g_mean:
              g_mean.append(np.array(profile))
          else:
              g_mean[0]=g_mean[0]+np.array(profile)
	  #####print"np.array(profile)=%s"%(np.array(profile))
	  ####print"g_mean=%s"%(g_mean)


      #Calculate g_mean, mean of all _searchprofiles
      for i in g_mean:#just one element since its 1 numpy array
          g_mean[0]=i/len(_searchProfiles)#this is executed only once
	  ####print"I: %s=%s/%s"%(i,i,len(_searchProfiles))#divide all numpy array elements in place
	  ####print"g_mean averaged=%s"%(g_mean)

       #PROBABLY NOT NEEDED THE PREVIOUS


      '''
      #now calculate dg- subtract each profile from the other to find the derivative profile
      dg_landmarkDerivativeProfiles=[]
      #for profileIndex in range(len(_searchProfiles)-1):
      for profileIndex in range(len(_searchProfiles)):

          #calculate derivative profile intensity g_image[i] - g_image[i-1]...and so on.... dg vector
          #....................
          #....................
          DG__searchProfiles=[]
          for i in range (len(_searchProfiles[profileIndex])-1):
              difference=_searchProfiles[profileIndex][i+1] - _searchProfiles[profileIndex][i]

              DG__searchProfiles.append(difference)


	  print "_searchProfiles[profileIndex] = %s"%(_searchProfiles[profileIndex])
          ##cv.WaitKey()
	  print "DG__searchProfiles = %s"%(DG__searchProfiles)
          ##cv.WaitKey()


          #difference=np.array(_searchProfiles[profileIndex+1])-np.array(_searchProfiles[profileIndex])
	  ####print"\n_searchProfiles[profileIndex+1]=%s\n - _searchProfiles[profileIndex]=%s\n .. difference=%s\n"%(_searchProfiles[profileIndex+1], _searchProfiles[profileIndex], difference)
          dg_landmarkDerivativeProfiles.append( (DG__searchProfiles) )#np.fabs
          #######cv.WaitKey()
       '''



      '''
      NOT NEEDED
      ###print"\dg vector=%s\n"%(dg_landmarkDerivativeProfiles)
      ####cv.WaitKey()

      ###print"dg_landmarkDerivativeProfiles=%s"%(len(dg_landmarkDerivativeProfiles))
      ####cv.WaitKey()
      '''

      #since we sobeled
      dg_landmarkDerivativeProfiles=_searchProfiles

      '''
      #calculate normalized dg FIX THIS - THE FOLLOWING FOR LOOP IS REDUNDANT THEN SINCE sum_dg is not used anywhere
      sum_dg=[]#contains only one numpy array
      for profileIndex in range(len(dg_landmarkDerivativeProfiles)):
          if not sum_dg:
              #sum_dg.append( np.fabs(np.array(dg_landmarkDerivativeProfiles[profileIndex])) )
              sum_dg.append( np.array(dg_landmarkDerivativeProfiles[profileIndex]) )
          else:
              #sum_dg[0]=sum_dg[0]+np.fabs(np.array(dg_landmarkDerivativeProfiles[profileIndex]))
              sum_dg[0]=sum_dg[0]+ np.array(dg_landmarkDerivativeProfiles[profileIndex])
	  #####print"np.array(profile)=%s"%(np.array(profile))
	  #####print"\sum_dg=%s\n"%(sum_dg)
          #######cv.WaitKey()

      ####print"\sum_dg vector=%s\n"%(sum_dg[0])
      #######cv.WaitKey()
      '''

      #an array of 6 by13 elements derived by the normalization of each dg
      yij_normalizedDerivProfile=[]


      for profileIndex in range(len(dg_landmarkDerivativeProfiles)):

	  #print "dg_landmarkDerivativeProfiles[%d]=%s"%(profileIndex, dg_landmarkDerivativeProfiles[profileIndex])
          #cv.WaitKey()
          #Note when dg_landmarkDerivativeProfiles[profileIndex] is copied to l, it is copied as a reference. so any changes to l will directly affect dg_landmarkDerivativeProfiles as well
          l=dg_landmarkDerivativeProfiles[profileIndex]
          #make sure we take the absolute element values of the list
          l=[ abs(x) for x in l ]
          horizontalElementVectorSum = sum(l)

	  #print "sum(dg_landmarkDerivativeProfiles[%d])=%s"%(profileIndex, horizontalElementVectorSum)
          #cv.WaitKey()

	  #print "divide vector with horizontalElementVectorSum=%s, so as to normalize it"%(horizontalElementVectorSum)
          #cv.WaitKey()
          normalizedDg = dg_landmarkDerivativeProfiles[profileIndex]/horizontalElementVectorSum
	  #print "normalizedDg=%s"%(normalizedDg)
          #cv.WaitKey()

          #add it
          yij_normalizedDerivProfile.append(normalizedDg.tolist())






      #Calculate dg magnitude based on: |a| = sqrt((ax * ax) + (ay * ay) + (az * az))

      #totalDGsum to divide dg with..
      totalDGsum=0


      '''
      OLD WAY

      #for the 6 derivative profile vectors , sum up their individual magnitudes/lengths
      for profileIndex in range(len(dg_landmarkDerivativeProfiles)):

          #square all elements of dg (derivative profiles)
          squared=np.power(dg_landmarkDerivativeProfiles[profileIndex] , 2)
	  ##print "math.pow(dg_landmarkDerivativeProfiles[profileIndex] , 2) = %s"%(np.power(dg_landmarkDerivativeProfiles[profileIndex] , 2))
          #c=####cv.WaitKey()
          #sum them up
          summed=np.sum(squared)
	  ##print "summed = %s"%(summed)
          #c=####cv.WaitKey()
          #sqrt them to derive this dg's magnitude
          sqrooted=np.sqrt(summed)
	  ##print "sqrooted = %s"%(sqrooted)
          #c=####cv.WaitKey()

          dg_magnitude=sqrooted

          ##add them all together so as to GET the normalized derivative of this landmark
          totalDGsum+=dg_magnitude

	  ##print "dg_landmarkDerivativeProfiles[profileIndex] = %s"%(dg_landmarkDerivativeProfiles[profileIndex])
          #c=####cv.WaitKey()
          #normalizedDg =            dg_landmarkDerivativeProfiles[profileIndex]/dg_magnitude
	  ##print "normalizedDg = %s"%(normalizedDg)
          #c=####cv.WaitKey()

          #yij_normalizedDerivProfile.append(totalDGsum)


      #for the 6 derivative profile vectors , divide up with the sum of their individual magnitudes/lengths (NORMALIZE DG for this landmark in this image)
      for profileIndex in range(len(dg_landmarkDerivativeProfiles)):
          yij_normalizedDerivProfile.append(dg_landmarkDerivativeProfiles[profileIndex]/dg_magnitude)
      '''


      '''
      NEW WAY
      #for the 6 derivative profile vectors , sum up their individual magnitudes/lengths
      for profileIndex in range(len(dg_landmarkDerivativeProfiles)):

          #square all elements of dg (derivative profiles)
          squared=np.power(dg_landmarkDerivativeProfiles[profileIndex] , 2)
	  #print "math.pow(dg_landmarkDerivativeProfiles[profileIndex] , 2) = %s"%(np.power(dg_landmarkDerivativeProfiles[profileIndex] , 2))
          #c=#cv.WaitKey()
          #sum them up
          summed=np.sum(squared)
	  #print "summed = %s"%(summed)
          #c=#cv.WaitKey()
          #sqrt them to derive this dg's magnitude
          sqrooted=np.sqrt(summed)
	  #print "sqrooted = %s"%(sqrooted)
          #c=#cv.WaitKey()

	  #print dg_landmarkDerivativeProfiles[profileIndex]
          ##cv.WaitKey()

          yij_normalizedDerivProfile.append(dg_landmarkDerivativeProfiles[profileIndex]/sqrooted)

	  #print dg_landmarkDerivativeProfiles[profileIndex]/sqrooted
          ##cv.WaitKey()

          ##add them all together so as to GET the normalized derivative of this landmark
          #totalDGsum+=dg_magnitude

	  ##print "dg_landmarkDerivativeProfiles[profileIndex] = %s"%(dg_landmarkDerivativeProfiles[profileIndex])
          #c=###cv.WaitKey()
          #normalizedDg =            dg_landmarkDerivativeProfiles[profileIndex]/dg_magnitude
	  ##print "normalizedDg = %s"%(normalizedDg)
          #c=###cv.WaitKey()

          #yij_normalizedDerivProfile.append(totalDGsum)


      #for the 6 derivative profile vectors , divide up with the sum of their individual magnitudes/lengths (NORMALIZE DG for this landmark in this image)
      #for profileIndex in range(len(dg_landmarkDerivativeProfiles)):
      #    yij_normalizedDerivProfile.append(dg_landmarkDerivativeProfiles[profileIndex])



      #print "yij_normalizedDerivProfile = %s"%(np.asarray(yij_normalizedDerivProfile))
      #c=#cv.WaitKey()
      yij_normalizedDerivProfile=np.asarray(yij_normalizedDerivProfile)
      '''



      #yij_normalizedDerivProfile.append(_searchProfiles)



      #for this ith image for this jth landmark derive calculate the normalized derivative profile
      ###################################################################################################################################################
      ###################################################################################################################################################
      '''TOOK THE NEXT LINE OUT as now Normalization of each of the derivative profiles for this landmark is calculated properly above in the for loop'''
      #yij_normalizedDerivProfile=np.array(dg_landmarkDerivativeProfiles) / np.array(sum_dg[0]) #equation 13 of 'Subspace Methods for Pattern Recognition in Intelligent Environment' book
      ###################################################################################################################################################
      ###################################################################################################################################################

      ##print"\y_ij vector=%s\n"%(yij_normalizedDerivProfile)
      #c=###cv.WaitKey()
      #if c==1048603 :#whichever integer key code makes the app exit
      #   exit()


      '''save this & every other landmark's normalized derivative profile for this image '''

      #print "yij_normalizedDerivProfile=%s"%(yij_normalizedDerivProfile)
      #cv.WaitKey()
      #print "yij_normalizedDerivProfile[0]=%s"%(yij_normalizedDerivProfile[0])
      #cv.WaitKey()



      IthImage_NormalizedtmpLandmarkDerivativeIntensityVector.append(yij_normalizedDerivProfile)#yij_normalizedDerivProfile[0]
      #print"\n%d , IthImage_NormalizedtmpLandmarkDerivativeIntensityVector=%s\n"%(len(IthImage_NormalizedtmpLandmarkDerivativeIntensityVector),IthImage_NormalizedtmpLandmarkDerivativeIntensityVector)
      ##cv.WaitKey()


  #find the G mean for all landmarks in the training set of shapes
  def findLandmarkNormIIntesityVec(self, gMeanFlag, oneTrainingImageVector, shape, trainingSetImage, IthImage_NormalizedtmpLandmarkDerivativeIntensityVector,gaussianMatrix, greyImage):

      # 'trainingSetImag'e' and  'greyImage' are the same as we pass the same argument at the point of function call

      gaussianWindow = cv.CreateImage(cv.GetSize(greyImage), greyImage.depth, greyImage.channels)
      cv.Copy(greyImage, gaussianWindow)

      '''for each of each shape's landmarks'''
      for i,p in enumerate(shape.pts):

	  ###print "p.x=%d ,p.y=%d"%(p.x,p.y)
          #####cv.WaitKey()

          ##if g mean of training set
          if gMeanFlag==1:
              '''draw them in a window named trainingSetImage'''
              tmpP = Point(p.x, p.y)
              #cv.Circle(trainingSetImage, ( int(tmpP.x), int(tmpP.y) ), 4, (100,100,100))
              cv.NamedWindow("trainingSetImage", cv.CV_WINDOW_NORMAL)
	      cv.ShowImage("trainingSetImage",trainingSetImage)
              ####cv.WaitKey()
          elif gMeanFlag==0:
              '''draw them in a window named targetImage'''
              tmpP = Point(p.x, p.y)
              #PathTrails : change trainingSetImage to self.image
              #cv.Circle(trainingSetImage, ( int(tmpP.x), int(tmpP.y) ), 1, (i*10,i*20,i*30))
              cv.NamedWindow("targetImage", cv.CV_WINDOW_NORMAL)
	      cv.ShowImage("targetImage",trainingSetImage)
              ####cv.WaitKey()

          #this image's current landmark point; for each of these landmarks calculate a 2d windows search profile
          #x=(p.x)
          #y=(p.y)

          #create a list of the g profiles for each landmark point
          currentLandmarkProfiles = []

          #store point
          p = p

          #for each point.. get normal direction of this the point based on the 2 adjacent point
          norm =  shape.get_normal_to_point(i)#??? self.shape

	  ####print"\n\n\n\n\n  !!!!!!!!!!!!!!!!!!!!!! New landmark point SEARCH PROFILE calculation !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

          tmpLandmarkDerivativeIntensityVector=[]
          currentLandmarkProfilesNormalizedDerivativesVector=[]

          #g vectors
          _searchProfiles=[]

          '''calculate, update-append to IthImage_NormalizedtmpLandmarkDerivativeIntensityVector (for each landmark for this shape)'''
          self.calculateIthImageLandmarkDerivIntensityVec(p, norm, greyImage, gaussianMatrix, _searchProfiles, IthImage_NormalizedtmpLandmarkDerivativeIntensityVector, gaussianWindow)


      print "XAXA=%s"%(IthImage_NormalizedtmpLandmarkDerivativeIntensityVector)
      ##print "IthImage_NormalizedtmpLandmarkDerivativeIntensityVector LENGTH %d"%(len(IthImage_NormalizedtmpLandmarkDerivativeIntensityVector[0]))
      #c=###cv.WaitKey()
      #if c==1048603 :#whichever integer key code makes the app exit
      #   exit()

      #this vector contains the normalized derivative profiles for all landmarks of this shape
      return IthImage_NormalizedtmpLandmarkDerivativeIntensityVector



  def findtrainingGmean(self, testImageNameCounter, AllImages_NormalizedtmpLandmarkDerivativeIntensityVector, gaussianMatrix):

      '''for each image/shape'''
      gMeanFlag=1;

      PyramidLevel_AllImages_NormalizedtmpLandmarkDerivativeIntensityVector=[]

      '''At each level..
          The procedure of getting image pyramid is as follows:
          1.
              Smooth the original image with a Gaussian filter, which is linearly
              decomposed into two 1-5-8-5-1 convolutions as Figure 9 shows. The reason of
              using Gaussian filter is that jagged edge will be produced in the sub-image if we
              directly sample the original image without a filter, and it will go against the
              gray-level modeling of landmarks.
          2.
            Sub-sample the image every other pixel in each dimension. Then we get a new
          image of level 1, which is 1/4 of the original image.
          3.
            From level 1, repeat step 1 and step 2 to obtain the higher level of the pyramid.
          4.
              Until we have got the highest level pre-defined for all the training images,
              terminate the process.
      '''


      ##at each level
      for imagelevel in range(self.pyramidLevels):
	  #print 'Pyramid Level %d'%(imagelevel)
          ###cv.WaitKey()

          testImageNameCounter=0
          imageNameDownSampled=0

          for asmshape in self.originalShapes:#self.asm.shapes !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 3 for now

              #add another list of derivative profile vectors for all landmarks which correspond to each image
              #IthImage_NormalizedtmpLandmarkDerivativeIntensityVector.append([])
              IthImage_NormalizedtmpLandmarkDerivativeIntensityVector=[]

              #1)for each image, convert to grayscale, iterate over the  corresponding points, and multiply profile pixel intensity with corresponding kernel gaussian distr. matrix element,
              #..then get the average of them for each tangential search profile and store it in the gith position of the g vector of elements along the whisker.


              #just test pixel value
	      #####print'my greyImage_gradient:'
	      #####printgreyImage[0][207,282]


              #######cv.WaitKey()
	      #print"loading image=%s"%("grey_image_"+str(testImageNameCounter+1)+".jpg")
              #######cv.WaitKey()


              #convert this test_grey_image to grayscale
              greyImage = []
              testImageNameCounter=testImageNameCounter+1
              '''so for shape 1..load  gray_image_1, for shape 2..load  gray_image_2 etc'''


              dirname="images"
              """ Reads an entire directory of .jpg files"""
              currentImageName = glob.glob(os.path.join(dirname, "grey_image_%d.jpg"%(testImageNameCounter) ))
	      ##print currentImageName
              #currentImageName="all5000images"+ "\\" +"franck_"+"%05d"%(testImageNameCounter)+".jpg"#grey_image_

              test_grey_image = cv.LoadImage(currentImageName[0])


              ##BLUR original image
              #test_grey_image = cv.GaussianBlur(test_grey_image,(5,5),0)
              #if imagelevel > 0:#smooth all images
              #    imgToSmooth = cv.LoadImage("grey_imageToBeScaled_level_%d.jpg"%(2**imagelevel))
              #    cv.Smooth(imgToSmooth, test_grey_image, cv.CV_GAUSSIAN, 5, 5)




              #grey_imageToBeScaled_level_ = cv2.imread("pyramidLevelImages/grey_imageToBeScaled_level_%d.jpg"%(2**imagelevel))#the larger the level number, the lowest the resolution the the image
	      #cv2.imshow("pyramidLevelImages/grey_imageToBeScaled_level_%d.jpg"%(2**imagelevel) , grey_imageToBeScaled_level_)
              ####cv.WaitKey()



              ##Save all the 3 subsampled training set images for each pyramid level. (if 3 levels then [0,1,2] level 0, [3,4,5] level 1, [6,7,8] level 2 )

              ##g_image[0],[1],[2] contain the grey_image unsampled
              ##g_image[3],[4],[5] contain the grey_image sampled at level 1 (all three [3],[4],[5] are identical)
              ##g_image[4],[5],[6] contain the grey_image sampled at level 2 (all three [4],[5],[6] are identical)
              ##..and so on.. if we have more than 3 pyramid levels

              if imagelevel == 0:

                  #convert image to numpy array
                  gradim = np.asarray(test_grey_image[:,:])#cv2.imread(cv2TocvConvertedImage)
                  gradim=sobel(gradim)#gradient(gradim, 1, 1 ,5)

                  source = gradim # source is numpy array
                  bitmap = cv.CreateImageHeader((source.shape[1], source.shape[0]), cv.IPL_DEPTH_8U, 3)
                  cv.SetData(bitmap, source.tostring(),
                             source.dtype.itemsize * 3 * source.shape[1])

                  self.g_image.append(self.__produce_gradient_image(bitmap, 2**imagelevel))

                  #NEED TO THINK WHAT WILL WE HAVE AS OUR 1ST level of the pyramid

              elif imagelevel > 0:#scale image to pyramidlevel blur & downsample

                  #if we are creating the 1 pyramid level sampling
                  if imageNameDownSampled==0:
                      img = cv2.imread(currentImageName[0],cv2.IMREAD_GRAYSCALE)

                      gradim=sobel(img)#gradient(img, 1, 1 ,5)
                      img=gradim
		      print "imageNameDownSampled %d"%(imageNameDownSampled)
		      print "1...imagelevel = %s "%(imagelevel)
                      #cv.WaitKey()

                  #else:
                  #    #use the previous level saved out downsampled&upsampled image
                  #    img = cv2.imread("grey_imageDownSampled_level_%d.jpg"%(testImageNameCounter-1))

                  #sub-sample iteratively the original target image 'imagelevel number of times'
                  for iterativeSamplingIndex in range(imagelevel):

                      h,w = img.shape[:2]

                      #gradim=sobel(img,1,1,5)
		      #img=gradim

                      #smoothed&downsampled & upsampled back
                      img = cv2.pyrDown( img,dstsize = (w/2, h/2) )
                      h,w = img.shape[:2]
                      img = cv2.pyrUp(img,dstsize = (2*w,2*h))
		      cv2.imshow("grey_imageDownSampled_%d_level_%d.jpg"%(testImageNameCounter,imagelevel),img)
                      cv2.imwrite("grey_imageDownSampled_%d_level_%d.jpg"%(testImageNameCounter,imagelevel),img)

                      test_grey_image = cv.LoadImage("grey_imageDownSampled_%d_level_%d.jpg"%(testImageNameCounter,imagelevel))
                      #cv.WaitKey()

		      print "iterativeSamplingIndex=%d , imagelevel=%d "%(iterativeSamplingIndex,imagelevel)
                      #cv.WaitKey()


		      #print 'scaled Down & up by..%d'%(2**imagelevel)
                      ####cv.WaitKey()


                  #Hence at level 1, there are 3 SAME subsampled images, each one corresponding to each shape
                  #Later, we are going to use each of these images for searching/finding normalized derivative profiles for each of the shapes.
                  #So, to sum up: the down sampled image should be the same for each of the shapes being tested,
                  #ex. Level 0:shape 1 tested for image1,                        shape 2 tested for image2,                        shape 3 tested for image3
                  #then
                  #ex. Level 1:shape 1 tested for image1 downsampled at level 1, shape 2 tested for image2 downsampled at level 1, shape 3 tested for image3 downsampled at level 1
                  #then
                  #ex. Level 2:shape 1 tested for image1 downsampled at level 2, shape 2 tested for image2 downsampled at level 2, shape 3 tested for image3 downsampled at level 2

                  self.g_image.append(self.__produce_gradient_image(test_grey_image, 2**imagelevel))



	      print 'length of g_image is %d at level %d'%(len(self.g_image),imagelevel)

              '''greyImage will have the greyscale image marked with landmarks'''
              greyImage.append(self.__produce_gradient_image(test_grey_image, 2**imagelevel))
	      #print 'gradient produced'
              ####cv.WaitKey()
	      #print 'now load the image of this level'


              '''create a copy of the greyscaled image to put the new landmarks onto'''
              trainingSetImage = cv.CreateImage(cv.GetSize(greyImage[0]), greyImage[0].depth, 1)

              #NEXT LINE PROBABLY NOT NEEDED!! - AS THE FOLLOWING LINE WILL BE OVERRIDEN BY THE NEXT STATEMENT (HENCE TRAININGSETIMAGE WILL BE OVERWRITTEN) -
              cv.Copy(greyImage[0], trainingSetImage)


              cv.Copy(self.g_image[ (testImageNameCounter-1) +(imagelevel*len(self.originalShapes))], trainingSetImage)#trainingSetImage contains the image corresponding
                                                                                                                       #to each shape image1 for shape1 at X level

              '''
              0 1  2  3
              4 5  6  7
              8 9  10 11

              4=0+4
              5=1+4
              6=2+4
              7=3+4

              8=4+4
              9=5+4
              10=6+4
              11=7+4

              4 is (numberofshapes)..so 8 coulde be expressed as 8=(testImageNameCounter-1)*len(self.originalShapes)+
                                                                 8=pyramidLevelnumber*numberofshapes+(testImageNameCounter-1)
                                                                hence..8=2*4+0 (2 second pyramid level level , 0th indexed image)
              '''


	      #print "g_image..%d"%( (testImageNameCounter-1) +(imagelevel*len(self.originalShapes)) )
              ###cv.WaitKey()
	      ###print "len greyImage=%s"%(len(greyImage[0]))

              #s=[]
              #asmshape.add_point(Point(100,100))
              #asmshape.add_point(Point(200,200))
              #asmshape.add_point(Point(300,300))

	      ###print "asm shape x&y length flattend (..so divide by 2 to get the total coordinate points) : %s"%( len(asmshape.get_vector()) )
	      ###print "asm shape : %s"%( asmshape.get_vector() )
              ####cv.WaitKey()

              '''get a vector of all landmark points of this shape'''
              asmshapePointVector=Shape.from_vector(asmshape.get_vector())#[320,185,309,196,292,203,276,209,264,217,256,237,261,252,275,255,294,243,308,238,327,240,345,241,342,224,336,217,335,209,348,203,359,210,369,214,376,206,392,211,400,215,404,214,402,198,399,189,390,181,389,172,383,160,373,147,354,145,338,161]
              totalLandmarksLeftToBeTested=len(asmshapePointVector.pts)



              '''CRUCIAL POINT MAKE SURE WE PASS ON THE 1ST ORDER SOBEL NORMALIZED DERIVATIVES OF THE TRAINING SET IMAGES'''

              #http://stackoverflow.com/questions/13104161/fast-conversion-of-iplimage-to-numpy-array
              gradimg = np.asarray(trainingSetImage[:,:])#cv2.imread(cv2TocvConvertedImage)
              #gradimg=sobel(gradimg,1,1,5)


              source = gradimg # source is numpy array
              bitmap = cv.CreateImageHeader((source.shape[1], source.shape[0]), cv.IPL_DEPTH_8U, 1)
              cv.SetData(bitmap, source.tostring(),
                         source.dtype.itemsize * 1 * source.shape[1])

              trainingSetImage=bitmap
              ##cv.WaitKey()






              #find g mean of each of the 30 landmarks of each asmshape and save it in IthImage_NormalizedtmpLandmarkDerivativeIntensityVector
              IthImage_NormalizedtmpLandmarkDerivativeIntensityVector=self.findLandmarkNormIIntesityVec (gMeanFlag, asmshapePointVector, asmshape, trainingSetImage, IthImage_NormalizedtmpLandmarkDerivativeIntensityVector,gaussianMatrix, trainingSetImage)

              #c=###cv.WaitKey()
              #if c==1048603 :#whichever integer key code makes the app exit
              #   exit()

              #save each image's  normalized derivative profile which contains all  normalized derivative profiles for each of its landmarks
              ## 0,1,2 are the normalized derivative profiles for each of the landmarks of image 1,2,3 for the 0th level
              ## 3,4,5 are the normalized derivative profiles for each of the landmarks of image 1,2,3 for the 1st level
              ## 6,7,8 are the normalized derivative profiles for each of the landmarks of image 1,2,3 for the 2nd level
              AllImages_NormalizedtmpLandmarkDerivativeIntensityVector.append(IthImage_NormalizedtmpLandmarkDerivativeIntensityVector)
	      ##print"\n\length of IthImage_NormalizedtmpLandmarkDerivativeIntensityVector vector=%s\n"%(len(IthImage_NormalizedtmpLandmarkDerivativeIntensityVector))
	      #print"\n\length of AllImages_NormalizedtmpLandmarkDerivativeIntensityVector vector=%s\n"%(len(AllImages_NormalizedtmpLandmarkDerivativeIntensityVector))
              ##cv.WaitKey()
	      ###print"\n\IthImage_NormalizedtmpLandmarkDerivativeIntensityVector vector=%s\n"%((IthImage_NormalizedtmpLandmarkDerivativeIntensityVector))



          #every AllImages_NormalizedtmpLandmarkDerivativeIntensityVector is a vector of 3 vectors
          #..each of which contains 30 elements each of which contains 6 profiles each of which contains 13 elements in the tangential direction

	  ###print"\n\length of IthImage_NormalizedtmpLandmarkDerivativeIntensityVector vector=%s\n"%(len(AllImages_NormalizedtmpLandmarkDerivativeIntensityVector[0][0]))
          ####cv.WaitKey()


          #Calculate yj_mean in the training set
          y_j_mean=[]
          meancounter=0

          correctionVectorForAllShapes=[]

          ####cv.WaitKey()

          #30 points
	  #print"\n\length of AllImages_NormalizedtmpLandmarkDerivativeIntensityVector vector=%s\n"%(len(AllImages_NormalizedtmpLandmarkDerivativeIntensityVector[0]))
          #cv.WaitKey()

          '''   AllImages_NormalizedtmpLandmarkDerivativeIntensityVector holds 3 shapes x 9 points x 6 profile x 13 elements each  '''

	  print"\n\length of AllImages_NormalizedtmpLandmarkDerivativeIntensityVector vector=%d\n"%(len(AllImages_NormalizedtmpLandmarkDerivativeIntensityVector))#3 levels
          #cv.WaitKey()
          #3 shapes
	  print"\n \ len AllImages_NormalizedtmpLandmarkDerivativeIntensityVector[0] vector=%s\n"%(len(AllImages_NormalizedtmpLandmarkDerivativeIntensityVector[0]))#at the 1st level:9 points
          #cv.WaitKey()

	  print"\n \ len AllImages_NormalizedtmpLandmarkDerivativeIntensityVector[0] vector=%s\n"%(len(AllImages_NormalizedtmpLandmarkDerivativeIntensityVector[0][0]))#at the  1st point :1 so [0] is needed as a helper
          #cv.WaitKey()

	  print"\n \ len AllImages_NormalizedtmpLandmarkDerivativeIntensityVector[0] vector=%s\n"%(len(AllImages_NormalizedtmpLandmarkDerivativeIntensityVector[0][0][0]))#at the 1st point: 7 profiles
          #cv.WaitKey()


	  #print"\n\ AllImages_NormalizedtmpLandmarkDerivativeIntensityVector vector=%s\n"%((AllImages_NormalizedtmpLandmarkDerivativeIntensityVector[1]))
          ##cv.WaitKey()
	  #print"\n\ AllImages_NormalizedtmpLandmarkDerivativeIntensityVector vector=%s\n"%((AllImages_NormalizedtmpLandmarkDerivativeIntensityVector[2]))
          ##cv.WaitKey()
	  #print"\n\ AllImages_NormalizedtmpLandmarkDerivativeIntensityVector vector=%s\n"%((AllImages_NormalizedtmpLandmarkDerivativeIntensityVector[3]))
          ##cv.WaitKey()



          #for each point compute its mean normalized derivative profile accross the training set
          for i in range(len(AllImages_NormalizedtmpLandmarkDerivativeIntensityVector[0])):#for each landmark in the set (30 for now)
            #Calculate yj_mean of this and every other landmark's, across training set of images & save it
            y_j_mean=[]
            meancounter=0


	    ##print '\n'
            #As we need to compute the y_j_mean for each level. Consequently, 30 landmarks x 3 images
            for index in range(len(self.originalShapes)):#for each image in the set (3 for now)

		##print "at shape %d"%(index)
		##print "landmark %d\n"%(i)


		##print "index+(pyramidLevels*imagelevel)=%d+(%d*%d) = %d"%( index,pyramidLevels, imagelevel  ,index+(pyramidLevels*imagelevel))
                ###cv.WaitKey()


                meancounter+=1
                if len(y_j_mean)==0:
                    #get the ith landmark derivative profiles, for all shapes.
                    #Add them and then Average them to derive the mean normalized derivative profile
                    #if 3 shapes, 30 landmarks-->then--> add the 1st landmark of the 1st shape, add the 1st landmark of the 2nd shape, add the 1st landmark of the 3rd shape
                    #then divide them by 3 to get the mean normalized derivative profile for the 1st landmark in the training set of images/shapes.

                    y_j_mean.append( np.array(AllImages_NormalizedtmpLandmarkDerivativeIntensityVector[index+ (imagelevel*self.pyramidLevels)][i]) )

                else:
                    y_j_mean[0]=y_j_mean[0]+ np.array(AllImages_NormalizedtmpLandmarkDerivativeIntensityVector[index+ (imagelevel*self.pyramidLevels)][i])

		#####print"np.array(profile)=%s"%(np.array(profile))
		####print"\y_j_mean=%s\n"%(y_j_mean[0])
                #####cv.WaitKey()

	    ###print "len(AllImages_NormalizedtmpLandmarkDerivativeIntensityVector)=%s"%(len(AllImages_NormalizedtmpLandmarkDerivativeIntensityVector))
	    ###print "len(AllImages_NormalizedtmpLandmarkDerivativeIntensityVector[index])=%s"%(len(AllImages_NormalizedtmpLandmarkDerivativeIntensityVector[index]))
            ####cv.WaitKey()

	    ###print "meancounter=%d"%(meancounter)
            y_j_mean = np.array(y_j_mean) / meancounter#len(self.asm.shapes)   #equation 14 of 'Subspace Methods for Pattern Recognition in Intelligent Environment' book

	    ###print "y_j_mean=%s"%(y_j_mean)
            ####cv.WaitKey()

            correctionVectorForAllShapes.append(y_j_mean)#THIS IS storing THE MEAN OF THE NORMALIZED DERIVATIVE PROFILE FOR EACH LANDMARK accross the training set (FOR EACH PYRAMID LEVEL)

	    ##print "len(correctionVectorForAllShapes)=%s"%(len(correctionVectorForAllShapes))
            ###cv.WaitKey()

          #return correctionVectorForAllShapes
          ##Save correction vector for this pyramid level
          PyramidLevel_AllImages_NormalizedtmpLandmarkDerivativeIntensityVector.append(correctionVectorForAllShapes)

	  #print "len(PyramidLevel_AllImages_NormalizedtmpLandmarkDerivativeIntensityVector)=%s"%(len(PyramidLevel_AllImages_NormalizedtmpLandmarkDerivativeIntensityVector))
          ###cv.WaitKey()

      #print "PyramidLevel_AllImages_NormalizedtmpLandmarkDerivativeIntensityVector=\n%s"%(PyramidLevel_AllImages_NormalizedtmpLandmarkDerivativeIntensityVector)




      ##COMPARE TWO IMAGES PIXEL BY PIXEL
      '''
      img1 = cv2.imread('grey_imageDownSampled_1_level_1.jpg')
      img2 = cv2.imread('grey_imageDownSampled_1_level_2.jpg')


      width, height, chanels = np.shape(img1)


      for py in range(0,h):
          for px in range(0,w):


              for x, y in zip(img1[py][px], img2[py][px]):
                      if x != y:
			  print 'Pixels Differ'
                          cv.WaitKey()
			  print "img1[%d][%d]=%s"%(py,px,img1[py][px])
			  print "img2[%d][%d]=%s"%(py,px,img2[py][px])
                          cv.WaitKey()
                      else:
			  print "img1[%d][%d]=%s"%(py,px,img1[py][px])
			  print "img2[%d][%d]=%s"%(py,px,img2[py][px])
      '''

      ###cv.WaitKey()

      return PyramidLevel_AllImages_NormalizedtmpLandmarkDerivativeIntensityVector


  def MahalanobisDist(self, x, y):#, inv_covariance_xy
      covariance_xy = np.cov(x,y, rowvar=1)
      inv_covariance_xy = np.linalg.pinv(covariance_xy)

      xy_mean = np.mean(x),np.mean(y)
      x_diff = np.array([x_i - xy_mean[0] for x_i in x])
      y_diff = np.array([y_i - xy_mean[1] for y_i in y])
      diff_xy = np.transpose([x_diff, y_diff])

      md = []
      totalsum=0
      for i in range(len(diff_xy)):
          calcVal=np.sqrt(np.dot(np.dot(np.transpose(diff_xy[i]),inv_covariance_xy),diff_xy[i]))
          md.append(calcVal)
          totalsum+=calcVal
      #return md
      return totalsum

  def findcurrentShapeGmean(self, currentshape, One_NormalizedtmpLandmarkDerivativeIntensityVector, gaussianMatrix, currentSearchPyramidLevel):


      gMeanFlag=0

      '''for each image/shape'''
      #for asmshape in self.asm.shapes:#self.originalShapes

      #add another list of derivative profile vectors for all landmarks which correspond to each image
      #curShape_NormalizedtmpLandmarkDerivativeIntensityVector.append([])
      curShape_NormalizedtmpLandmarkDerivativeIntensityVector=[]

      #1)for each image, convert to grayscale, iterate over the  corresponding points, and multiply profile pixel intensity with corresponding kernel gaussian distr. matrix element,
      #..then get the average of them for each tangential search profile and store it in the gith position of the g vector of elements along the whisker.


      #just test pixel value
      #####print'my greyImage_gradient:'
      #####printgreyImage[0][207,282]


      #######cv.WaitKey()
      ####print"testing image=%s"%("grey_image_"+str(testImageNameCounter)+".jpg")
      #######cv.WaitKey()

      #      testImageNameCounter=0
      #      #convert this test_grey_image to grayscale
      #      greyImage = []
      #      testImageNameCounter=testImageNameCounter+1
      #      '''so for shape 1..load  gray_image_1, for shape 2..load  gray_image_2 etc'''

      #      currentImageName="grey_image_"+str(testImageNameCounter)+".jpg"
      #      test_grey_image = cv.LoadImage(currentImageName)


      #      '''greyImage will have the greyscale image marked with landmarks'''
      #      greyImage.append(self.__produce_gradient_image(test_grey_image, 2**0))


      #      '''create a copy of the greyscaled image to put the new landmarks onto'''
      #      trainingSetImage = cv.CreateImage(cv.GetSize(greyImage[0]), greyImage[0].depth, 1)
      #      cv.Copy(greyImage[0], trainingSetImage)


      ###print "len greyImage=%s"%(len(greyImage[0]))

      #s=[]
      #currentshape.add_point(Point(100,100))
      #currentshape.add_point(Point(200,200))
      #currentshape.add_point(Point(300,300))

      ###print "asm shape x&y length flattend (..so divide by 2 to get the total coordinate points) : %s"%( len(currentshape.get_vector()) )
      ###print "asm shape : %s"%( currentshape.get_vector() )
      ####cv.WaitKey()

      '''get a vector of all landmark points of this shape'''
      currentshapePointVector=Shape.from_vector(currentshape.get_vector())#[320,185,309,196,292,203,276,209,264,217,256,237,261,252,275,255,294,243,308,238,327,240,345,241,342,224,336,217,335,209,348,203,359,210,369,214,376,206,392,211,400,215,404,214,402,198,399,189,390,181,389,172,383,160,373,147,354,145,338,161]
      totalLandmarksLeftToBeTested=len(currentshapePointVector.pts)

      #trainingSetImage = cv.CreateImage(cv.GetSize(self.g_image[currentSearchPyramidLevel]), self.g_image[currentSearchPyramidLevel].depth, 1)
      #cv.Copy(self.g_image[currentSearchPyramidLevel], trainingSetImage)

      ## mySELFIMAGE_TargetImage SHOULDN'T NEED TO REPLACE self.image AS IT'S OUR TARGET IMAGE
      '''CRUCIAL POINT MAKE SURE WE PASS ON THE 1ST ORDER SOBEL NORMALIZED DERIVATIVES OF THE TRAINING SET IMAGES'''
      '''
      #http://stackoverflow.com/questions/13104161/fast-conversion-of-iplimage-to-numpy-array
      gradimg = np.asarray(self.image[:,:])#cv2.imread(cv2TocvConvertedImage)
      gradimg=sobel(gradimg,1,1,5)


      source = gradimg # source is numpy array
      bitmap = cv.CreateImageHeader((source.shape[1], source.shape[0]), cv.IPL_DEPTH_8U, 1)
      cv.SetData(bitmap, source.tostring(),
                 source.dtype.itemsize * 1 * source.shape[1])

      mySELFIMAGE_TargetImage=bitmap
      '''




      #find g mean of 30 landmarks of each shape and save it in curShape_NormalizedtmpLandmarkDerivativeIntensityVector
      self.findLandmarkNormIIntesityVec (gMeanFlag, currentshapePointVector, currentshape, self.greyTargetImage[currentSearchPyramidLevel], curShape_NormalizedtmpLandmarkDerivativeIntensityVector,gaussianMatrix, self.greyTargetImage[currentSearchPyramidLevel])


      #save each image's normalized derivative profile which contains all normalized derivative profiles for each of its landmarks
      #Hence if 9 landmarks: One_NormalizedtmpLandmarkDerivativeIntensityVector has 9 elements
      One_NormalizedtmpLandmarkDerivativeIntensityVector.append(curShape_NormalizedtmpLandmarkDerivativeIntensityVector)
      print"\n\ WOW length of curShape_NormalizedtmpLandmarkDerivativeIntensityVector vector=%s\n"%(len(curShape_NormalizedtmpLandmarkDerivativeIntensityVector))
      #cv.WaitKey()
      #Each element has 7 derivative profiles
      print"\n\curShape_NormalizedtmpLandmarkDerivativeIntensityVector vector=%s\n"%((curShape_NormalizedtmpLandmarkDerivativeIntensityVector))
      #cv.WaitKey()


      #every One_NormalizedtmpLandmarkDerivativeIntensityVector is a vector of 3 vectors
      #..each of which contains 30 elements each of which contains 6 profiles each of which contains 13 elements in the tangential direction

      print"\n\length of curShape_NormalizedtmpLandmarkDerivativeIntensityVector vector=%s\n"%(len(One_NormalizedtmpLandmarkDerivativeIntensityVector[0][0]))
      #cv.WaitKey()

      print"\nlength of One_NormalizedtmpLandmarkDerivativeIntensityVector[0]=%s\n"%(len(One_NormalizedtmpLandmarkDerivativeIntensityVector[0]))
      #cv.WaitKey()

      print"\n\length of One_NormalizedtmpLandmarkDerivativeIntensityVector=%s\n"%(len(One_NormalizedtmpLandmarkDerivativeIntensityVector))
      #cv.WaitKey()


      #Calculate yj_mean of the training set
      currentShape_y_j_mean=[]
      meancounter=0

      correctionVectorForAllShapes=[]

      ####cv.WaitKey()
      #f = open('landmarksTestGChosen/landmark_g_Chosen','w')
      for i in range(len(One_NormalizedtmpLandmarkDerivativeIntensityVector[0])):#for each landmark in the set (30 for now)
        #Calculate yj_mean
        #currentShape_y_j_mean=[]
        #meancounter=0

	print "len(One_NormalizedtmpLandmarkDerivativeIntensityVector[0]) = %d"%(len(One_NormalizedtmpLandmarkDerivativeIntensityVector[0]))
        #cv.WaitKey()

        for index in range(len(One_NormalizedtmpLandmarkDerivativeIntensityVector)):#for each image in the set (3 for now)

	    ##print "at shape %d"%(index)
	    ##print "landmark %d"%(i)
            ####cv.WaitKey()

	    print "len(One_NormalizedtmpLandmarkDerivativeIntensityVector) = %d"%(len(One_NormalizedtmpLandmarkDerivativeIntensityVector))
            #cv.WaitKey()


            meancounter+=1
            if len(currentShape_y_j_mean)==0:
                currentShape_y_j_mean.append( np.array(One_NormalizedtmpLandmarkDerivativeIntensityVector[index][i]))#append 6 profiles with 13 elements each one (of each image)
            else:
                currentShape_y_j_mean[0]=currentShape_y_j_mean[0]+ np.array(One_NormalizedtmpLandmarkDerivativeIntensityVector[index][i])#add the next 6 profiles of the same landmarks in training set of images
	    #####print"np.array(profile)=%s"%(np.array(profile))
	    ##print"\currentShape_y_j_mean=%s\n"%(currentShape_y_j_mean[0])
            ####cv.WaitKey()

	###print "len(One_NormalizedtmpLandmarkDerivativeIntensityVector)=%s"%(len(One_NormalizedtmpLandmarkDerivativeIntensityVector))
	###print "len(One_NormalizedtmpLandmarkDerivativeIntensityVector[index])=%s"%(len(One_NormalizedtmpLandmarkDerivativeIntensityVector[index]))
        ####cv.WaitKey()

	print "meancounter=%d"%(meancounter)
        #cv.WaitKey()
        currentShape_y_j_mean = np.array(currentShape_y_j_mean) / meancounter#len(self.asm.shapes)
        correctionVectorForAllShapes.append(currentShape_y_j_mean)

      #WE ARE NOT USING THE VECTOR/ CURRENTLY IT ALSO MAKES NO SENSE AS One_NormalizedtmpLandmarkDerivativeIntensityVector is a length 1 vector
      return correctionVectorForAllShapes


  #General description of how it fitting works,
  '''
  1)calc training set g mean once

  2)calc self.CovarianceMatricesVec once based on findtrainingGmean

  3)then at each iteration:

        i)calc current  live shape (for all its landmarks) the norm derivative profile (One_NormalizedtmpLandmarkDerivativeIntensityVector)
        ii)retrieve the training set g mean precalculated for this landmark
        iii)retrieve the self.CovarianceMatricesVec once based on findtrainingGmean

              for each element in this landmark (..each of the 7 along the whisker)
                      calc mahalanobis to find minimum distance between [ this live shape's landmark profile v0,v1,v2..,v6 &
                      the training set's shape for this landmark profile v0_mean,v1_mean,v2_mean..,v6_mean],
                      (which means max resembalance between the two)

        Note (the covariance matrix used is only calculated once during traing for each landmakr)

  '''

  def getCorrectedLandmarkPoint(self,trainingGmean, iterationIndex, currentSearchPyramidLevel):

      #####print"asm.mean",asm.mean
      #scale=0

      totalLandmarkNumber=-1

      #################################################
      #################################################
      #MAHALANOBIS:COULD COMPUTE Gmean here.

      #if y=6 pixels+1 along the whisker [-3..to..3] && x=12 pixels+1 for each search profile,formed either side of each centered position y  [-6..to..6] along the whisker
      #the pseudo algorithm should:
      #sample and create a y element vector 'g', of which each element contains a "sampled" intensity value, as a result of sampling y intensities values along the whisker for this landmark point of this training image
      #calculate derivative profile vector 'dg' of this landmark by calculating differences between elements on g
      #normalize the derivative profile y[i,j] for this landmark by dividing the vector by the sum of all individual dg elements
      #find the mean of the normalized derivative profiles ymean[i,j] of this landmark by examining all images in the provided training set
      #calculate Covariance matrix 'CovMat' for this landmark by using: the following equation, examining all images for a landmark:
      '''
      C[i,j] = ( 1/totalnumberofImagesInTheTrainingSet ) * SUMOF[( y[i,j] - ymean[i,j] ) * ( y[i,j] - ymean[i,j] ).transposed]
      '''

      #important: profile sampling could mean: adding to g(ith) vec intensities of all 6+1 pixels of each search profile to from g[i,j,-6] or g[i,j,-5] ...or g[i,j,6] eq.11 of 'Subspace Methods for Pattern Recognition in Intelligent Environment' book
      #important: by acquiring all g(ith) vecs we can take the gmean of them (whatever that means) and plug it in the following mahalanobis equation

          # during searching at a landmark choose as best fit the pixel along the normal whisker whose g search profile vector has the lowest mahalanobis distance from the gmean
          #mahalanobisDist=(g-gmean).transposed * InverseCovMat * (g-gmean) : we should test against all g


      #So..without further ado
      #We first need to get graylevel info for the ith image.
      #Hence we load the ith image, we convert it to grayscale, and perform sampling along the normal to build a gray-level profile for it
      #We then do that for all of the training images in the set.
      #And so we end up having formed gray-level profiles for all of them..from which we can derive the coresponding covariance matrix for each of the images and plug it to the mahalanobis equation.

      #      http://dev.theomader.com/gaussian-kernel-calculator/ matrix mask source

      #Need to have-create images corresponding to shape1,2,3..... #.pts
#      gaussianMatrix=np.array([
#                                                          [0.000158,	0.000608,        0.00183,	0.004312,	0.007953,	0.011483,	0.012978,	0.011483,	0.007953,	0.004312,	0.00183,	0.000608,	0.000158],
#                                                          [0.000291,	0.001121,	0.003375,	0.007953,	0.014669,	0.021179,	0.023938,	0.021179,	0.014669,	0.007953,	0.003375,	0.001121,	0.000291],
#                                                          [0.000421,	0.001618,	0.004873,	0.011483,	0.021179,	0.030579,	0.034561,	0.030579,	0.021179,	0.011483,	0.004873,	0.001618,	0.000421],

#                                                          [0.000476,	0.001829,	0.005508,	0.012978,	0.023938,	0.034561,	0.039062,	0.034561,	0.023938,	0.012978,	0.005508,	0.001829,	0.000476],

#                                                          [0.000421,	0.001618,	0.004873,	0.011483,	0.021179,	0.030579,	0.034561,	0.030579,	0.021179,	0.011483,	0.004873,	0.001618,	0.000421],
#                                                          [0.000291,	0.001121,	0.003375,	0.007953,	0.014669,	0.021179,	0.023938,	0.021179,	0.014669,	0.007953,	0.003375,	0.001121,	0.000291],
#                                                          [0.000158,	0.000608,	0.00183,	0.004312,	0.007953,	0.011483,	0.012978,	0.011483,	0.007953,	0.004312,	0.00183,	0.000608,	0.000158]
#                                                          ])

#      gaussianMatrix=np.array([
#                                                          [0.005084,	0.009377,	0.013539,	0.015302,	0.013539,	0.009377,	0.005084],
#                                                          [0.009377,	0.017296,	0.024972,	0.028224,	0.024972,	0.017296,	0.009377],
#                                                          [0.013539,	0.024972,	0.036054,	0.040749,	0.036054,	0.024972,	0.013539],

#                                                          [0.015302,	0.028224,	0.040749,	0.046056,	0.040749,	0.028224,	0.015302],

#                                                          [0.013539,	0.024972,	0.036054,	0.040749,	0.036054,	0.024972,	0.013539],
#                                                          [0.009377,	0.017296,	0.024972,	0.028224,	0.024972,	0.017296,	0.009377],
#                                                          [0.005084,	0.009377,	0.013539,	0.015302,	0.013539,	0.009377,	0.005084]
#                                                          ])

#      #Transposed matrix to align with the iteration from bottom(tangent) left(whisker)..to..up(tangent right(whisker))
#      gaussianMatrix=np.transpose(gaussianMatrix)

#      ####print"gaussianMatrix=%s"%(gaussianMatrix)

#      #iterate the gaussian distribution matrix
#      for (x,y), value in np.ndenumerate(gaussianMatrix):
#                  ##print"%d, %d =%s"%(x,y,gaussianMatrix[x][y])
#                  #######cv.WaitKey()


      #http://dev.theomader.com/gaussian-kernel-calculator/  Sigma=2, Kernel Size=13
      '''
          0.000006	0.000022	0.000067	0.000158	0.000291	0.000421	0.000476	0.000421	0.000291	0.000158	0.000067	0.000022	0.000006
          0.000022	0.000086	0.000258	0.000608	0.001121	0.001618	0.001829	0.001618	0.001121	0.000608	0.000258	0.000086	0.000022
          0.000067	0.000258	0.000777	0.00183		0.003375	0.004873	0.005508	0.004873	0.003375	0.00183		0.000777	0.000258	0.000067

          #core kernel for (3+1)x(6+1) elements
          0.000158	0.000608	0.00183		0.004312	0.007953	0.011483	0.012978	0.011483	0.007953	0.004312	0.00183		0.000608	0.000158
          0.000291	0.001121	0.003375	0.007953	0.014669	0.021179	0.023938	0.021179	0.014669	0.007953	0.003375	0.001121	0.000291
          0.000421	0.001618	0.004873	0.011483	0.021179	0.030579	0.034561	0.030579	0.021179	0.011483	0.004873	0.001618	0.000421
          0.000476	0.001829	0.005508	0.012978	0.023938	0.034561	0.039062	0.034561	0.023938	0.012978	0.005508	0.001829	0.000476
          0.000421	0.001618	0.004873	0.011483	0.021179	0.030579	0.034561	0.030579	0.021179	0.011483	0.004873	0.001618	0.000421
          0.000291	0.001121	0.003375	0.007953	0.014669	0.021179	0.023938	0.021179	0.014669	0.007953	0.003375	0.001121	0.000291
          0.000158	0.000608	0.00183		0.004312	0.007953	0.011483	0.012978	0.011483	0.007953	0.004312	0.00183		0.000608	0.000158
          #core kernel for (3+1)x(6+1) elements

          0.000067	0.000258	0.000777	0.00183		0.003375	0.004873	0.005508	0.004873	0.003375	0.00183		0.000777	0.000258	0.000067
          0.000022	0.000086	0.000258	0.000608	0.001121	0.001618	0.001829	0.001618	0.001121	0.000608	0.000258	0.000086	0.000022
          0.000006	0.000022	0.000067	0.000158	0.000291	0.000421	0.000476	0.000421	0.000291	0.000158	0.000067	0.000022	0.000006
      '''


      #this encapsulates all images' normalized derivative profiles for all their landmarks
      #AllImages_NormalizedtmpLandmarkDerivativeIntensityVector=[]


      #for all 3 images' vectors of landmark points in the training set
      #testImageNameCounter=0


      '''
      #convert this test_grey_image to grayscale
      greyImage = []
      testImageNameCounter=testImageNameCounter+1
      #so for shape 1..load  gray_image_1, for shape 2..load  gray_image_2 etc

      test_grey_image = cv.LoadImage("grey_image_"+str(testImageNameCounter)+".jpg")
      currentImageName="grey_image_"+str(testImageNameCounter)+".jpg"

      #greyImage will have the greyscale image marked with landmarks
      greyImage.append(self.__produce_gradient_image(test_grey_image, 2**0))


      #create a copy of the greyscaled image to put the new landmarks onto
      oneTrainingImageVectorimage = cv.CreateImage(cv.GetSize(greyImage[0]), greyImage[0].depth, 1)
      cv.Copy(greyImage[0], oneTrainingImageVectorimage)
      '''

      #trainingGmean = self.findtrainingGmean(testImageNameCounter, AllImages_NormalizedtmpLandmarkDerivativeIntensityVector, gaussianMatrix)
      ###print "len(trainingGmean)=%s"%(len(trainingGmean))
      ####cv.WaitKey()


      #Now find y_j_mean for the current shape and then perform mahalanobis check between the mean_obtained_from_training and the current mean for each landmark
      ##print 'Now find y_j_mean for the current shape and then perform mahalanobis check between the mean_obtained_at_training and the each of the current g profiles of each landmark to find match mathc than minimizes..'
      ####cv.WaitKey()

      ##print"\n trainingGmean vector=\n%s\n"%(trainingGmean)
      ####cv.WaitKey()

      '''
      for i in range(9):

          testGimage = cv.CreateImage(cv.GetSize(self.g_image[i]), self.g_image[i].depth, 1)
          cv.Copy(self.g_image[i], testGimage)
          cv.NamedWindow("testGimage", cv.CV_WINDOW_NORMAL)
	  cv.ShowImage("testGimage",testGimage)
          ###cv.WaitKey()
      '''




      #target image
      self.image
      #self.g_image[currentSearchPyramidLevel*3]##as we have 3 different images in the training set and 3 levels in the pyramid, scale
      #get a vector of all current landmark points
      currentShape=self.shape#Shape.from_vector(self.asm.shapes[index].get_vector())#self.originalShapes
      #find y_j_mean for the current shape
      One_NormalizedtmpLandmarkDerivativeIntensityVector = []
      #CURRENTLY currentShapeGmean IS NOT USED, HENCE WHATEVER CORRECTIONVECTOR shapes returnt are worthless, however One_NormalizedtmpLandmarkDerivativeIntensityVector gets filled
      currentShapeGmean = self.findcurrentShapeGmean(currentShape, One_NormalizedtmpLandmarkDerivativeIntensityVector, self.gaussianMatrix, currentSearchPyramidLevel)


      print "len(One_NormalizedtmpLandmarkDerivativeIntensityVector)=%s"%(len(One_NormalizedtmpLandmarkDerivativeIntensityVector[0]))
      #cv.WaitKey()
      print"\n One_NormalizedtmpLandmarkDerivativeIntensityVector=\n%s\n"%(One_NormalizedtmpLandmarkDerivativeIntensityVector)
      #cv.WaitKey()


      '''CALCULATE ALL COVARIANCE MATRICES ONLY ONCE AT THE START SO AS TO FORM THE GREY LEVEL PROFILE & ...
       and now THAT we have for each landmark a  certain GREY profile yj ,  we can use for the image search,
       and particularly finding the  desired movements of the landmarks that take xi to xi + dxi .'''



      #if list empty
      '''self.COVARIANCE_CALCULATION_ONCE != True'''

      ##print "CHECK EMTPY COVARIANCE_CALCULATION_ONCE"
      ##print "level %d"%(currentSearchPyramidLevel)
      ##print self.CovarianceMatricesVec[currentSearchPyramidLevel]
      ##cv.WaitKey()


      #repr is for formating numpy arrays with commas
      np.set_printoptions(threshold=np.nan)

      ##If pyramid level in list CovarianceMatricesVec is empty, then append Covariance Matrices to it (same for next level, and same for level up above in in the CovarianceMatricesVec list)
      if not self.COVARIANCE_CALCULATION_ONCE:#self.CovarianceMatricesVec[currentSearchPyramidLevel]:

          ##NEXT LINE WILL PREVENT FROM RE-CALCULATING CovarianceMatricesVec, HOWEVER FOR NOW LET'S RECALCULATE THEM..
          self.COVARIANCE_CALCULATION_ONCE=True
	  #print "\nCOVARIANCE_CALCULATION_ONCE"
          #cv.WaitKey()

          #x=np.array(One_NormalizedtmpLandmarkDerivativeIntensityVector[0][0])
          #self.tmpCovar = np.cov(x, rowvar=0)#SAME

          #for all pyramidlevels, calulcate covariance matrices
          for imagelevel in range(self.pyramidLevels):#3 for now

              #for p_num in range(len(One_NormalizedtmpLandmarkDerivativeIntensityVector[0])):
               for p_num in range(len(trainingGmean[imagelevel])):#for each of the Gmean points saved
                  #tmpCovar= cv2.calcCovarMatrix(One_NormalizedtmpLandmarkDerivativeIntensityVector[0][p_num], cv.CV_COVAR_COLS,cv.CV_64F | cv.CV_COVAR_NORMAL | cv.CV_COVAR_USE_AVG, trainingGmean[p_num][0])

                  #Calculate the covariance matrix of each of the landmark points at each pyramid level, by examining the..
                  #each landmark's Normalized Derivative Intensity Vector

                  x=0

                  #As we need to computer the y_j_mean for each level. Consequently, 30 landmarks x 3 images
                  for index in range(len(self.originalShapes)):#for each image in the set (3 for now):

                      '''   self.AllImages_NormalizedtmpLandmarkDerivativeIntensityVector holds 4 shapes x 20 points x 6 profile x 13 elements each  '''

                      #FOR THIS LEVEL and FOR THIS LANDMARK NUMBER: add in "this landmark point, of each shape from the training set out of all shapes/images used for training"
                      x+= np.array(self.AllImages_NormalizedtmpLandmarkDerivativeIntensityVector[index+ (imagelevel*self.pyramidLevels)][p_num])
		      print "Covar to calulate upon: x=%s"%(x)
                      #cv.WaitKey()

                  #Get the average of landmark X (for this level)
                  for i in x:
                       i /= len(self.originalShapes) #6 is the number of rows
		  #print "DIVIDED x="
		  #print x
                  ##cv.WaitKey()

                  '''WAS WORKING with this'''
                  y=np.array( trainingGmean[imagelevel][p_num][0] )
		  ##print "x="
		  ##print repr(x)
                  ##cv.WaitKey()

		  ##print "y="
		  ##print repr(y)
                  ##cv.WaitKey()

                  #x=np.array(One_NormalizedtmpLandmarkDerivativeIntensityVector[0][p_num])# NOT CORRECT
                  tmpCovar=np.cov(x,rowvar=0) #WORKS CONVERGES QUICKLY
                  #tmpCovar=np.cov(x,y) #SHOULD SAVE TIME BUT NOOOOOOO. AS IT PRODUCES A 12 ELEMENT INSTEAD OF A 13 ELEMENT ARRAY THAT WE WANT (HOWEVER BOTH THE previous line and the cv2.calcCovarianceMatrix produce 13 ELEMENT ARRAYS)

                  '''WAS WORKING with this'''
                  #tmpCovarTEST=0
                  #tmpCovarTEST= cv2.calcCovarMatrix(x, cv.CV_COVAR_COLS,cv.CV_64F | cv.CV_COVAR_NORMAL | cv.CV_COVAR_USE_AVG, y[0]) #LOOKS LIKE IT WORKS..CONVERGES SLOWLY
                  #for i in tmpCovarTEST[0]:
                  #    i /= 6 #6 is the number of rows


		  #print "tmpCovar"
		  #print tmpCovar
                  ##cv.WaitKey()

		  ##print "tmpCovarTEST"
		  ##print tmpCovarTEST
                  ##cv.WaitKey()

                  self.CovarianceMatricesVec[imagelevel].append(tmpCovar)#tmpCovarTEST
                  #use the already calculated corresponding covariance matrix

		  ##print "\nCOVARIANCE_CALCULATION_ONCE"

	       #print "len of cov at pyramid level %d is %d \n = %s"%(imagelevel, len(self.CovarianceMatricesVec[imagelevel]), self.CovarianceMatricesVec[imagelevel])
	       ##print "len of cov %d is %d \n = %s"%(1, len(self.CovarianceMatricesVec[1]), self.CovarianceMatricesVec[1])
	       ##print "len of cov %d is %d \n = %s"%(0, len(self.CovarianceMatricesVec[0]), self.CovarianceMatricesVec[0])
               #cv.WaitKey()




      #Calculate Covariance matrix, for this ith image for the jth landmark from (y_j_mean &  IthImage_NormalizedtmpLandmarkDerivativeIntensityVector[i])
      '''
      We want to find the covariance matrix with minimum volume encompassing some % of the data set.
      Let us assume we have a set of observations X = {x1, x2, ... xN} of size N.  Further let us assume 10% of samples are outliers.
      One could construct a brute-force algorithm as follows:

      determine all unique sample subsets of size h = 0.9N
      for each subset S
      compute the covariance matrix of Cs = S
      compute Vs = det(Cs)
      choose the subset where Vs is a minimal
      Geometrically, the determinant is the volume of the N dimensional vector space implied by the covariance matrix.
      Minimizing the ellipsoid is equivalent to minimizing the volume.
      '''

      if totalLandmarkNumber==-1:
          totalLandmarkNumber=len(self.AllImages_NormalizedtmpLandmarkDerivativeIntensityVector[0])#for 30 totalLandmarkNumber

      #for s in self.asm.shapes:

      correctionVectorForAllShapes=[]
      #for p_num in range(totalLandmarkNumber):#0..29 iterate only for 30 elements, but for a different asmshapePointVector
      #for index in range(len(AllImages_NormalizedtmpLandmarkDerivativeIntensityVector)):


      #Now we need to find the g profile from each landmark of the current shape that minimizes the mahalanobis distance between this and the one found as the Gmean of training set (for this landmark)
      ##print 'Now we need to find the g profile from each landmark of the current shape that minimizes the mahalanobis distance between this and the one found as the Gmean of training set (for this landmark)'
      ####cv.WaitKey()

      ###print 'asmshapePointVector=%s'%(asmshapePointVector.pts[0])
      ##print 'len(One_NormalizedtmpLandmarkDerivativeIntensityVector)=%s'%(len(One_NormalizedtmpLandmarkDerivativeIntensityVector[0][0]))
      ####cv.WaitKey()


      #IF WE HAVE 3 PYRAMID LEVELS THEN WE trainingGmean IS OF LENGTH 3
      #Each one contains the mean normalized derivative profiles of all landmarks across training set at a specified pyramid level
      pointnumber=0
      #print "trainingGmean[currentSearchPyramidLevel][%d]=%s"%(pointnumber,trainingGmean[currentSearchPyramidLevel][pointnumber][0])

      #print repr(trainingGmean[currentSearchPyramidLevel][pointnumber][0])

      ##cv.WaitKey()

      #print 'length of trainingGmean=%s'%(len(trainingGmean))

      ##Count the number of landmarks that moved to 50% of the total length of profile around landmark (measure 'move to next pyramid level')
      bestIndexWithinHalfProfileLengthCounter=0

      #set it back to Fals for next pyramidlevel image search
      breaktonextpyramidlevel=False

      preSpecifiedNumberOfLevelIterations=5


      targetImageTocheckAgainst=self.greyTargetImage[currentSearchPyramidLevel]#g_image[currentSearchPyramidLevel] #self.image
      CHOSEN_BEST = cv.CreateImage(cv.GetSize(targetImageTocheckAgainst), targetImageTocheckAgainst.depth, 1)
      cv.Copy(targetImageTocheckAgainst, CHOSEN_BEST)

      #convert the grayscaled targetimage to color..so as to see the possible choices along the wisker
      size = cv.GetSize(CHOSEN_BEST)
      print "len of greyTargetImage as many as the pyramid level"
      #convert CHOSEN_BEST to RGB  and saved it to grey_image
      grey_image = cv.CreateImage(size, 8, 3)
      cv.CvtColor(CHOSEN_BEST, grey_image, cv.CV_GRAY2RGB)
      #CHOSEN_BEST=grey_image



      cv.SaveImage("current_targetImageTocheckAgainst.jpg", grey_image)

      #read back so as to CONVERT for convenience (to play with cv2 instead of cv / actually need to convert all cv stuff to cv2 )
      CHOSEN_BEST = cv2.imread("current_targetImageTocheckAgainst.jpg",cv2.IMREAD_GRAYSCALE)

      f = open('landmarksTestGChosen/landmark_g_Chosen','w')


      #on the very 1st iteration, we initialize the list to (landmarkpointnumber, TopLeftPoint(-1,-1), BottomRightPoint(-1,-1))
      if not self.prevBestHistShapePointsList:
	  print "self.prevBestHistShapePointsList is Empty (we are currently..at the first iteration)"
          for p_num in range(len(One_NormalizedtmpLandmarkDerivativeIntensityVector[0])):
                TopLeftPoint=(-1,-1)
                BottomRightPoint=(-1,-1)
                prevLandmarkElement=(p_num, TopLeftPoint, BottomRightPoint)
                self.prevBestHistShapePointsList.append(prevLandmarkElement)


      #for each current/live shape (which contains ALL landmarks norm derivative profiles) , compare ONE specific landmark with
      #the Gmean norm derivative profile precalculated at training phase, using the covariance matrix based also on the Gmean precalculated at training phase
      for p_num in range(len(One_NormalizedtmpLandmarkDerivativeIntensityVector[0])):
      #####print"\n np.array(IthImage_NormalizedtmpLandmarkDerivativeIntensityVector[p_num])=%s"%(np.array(IthImage_NormalizedtmpLandmarkDerivativeIntensityVector[p_num]).flatten())
      #####print"\n np.array(y_j_mean)=%s"%(np.array(y_j_mean[0]).flatten())

	  print "One_NormalizedtmpLandmarkDerivativeIntensityVector =%s"%(One_NormalizedtmpLandmarkDerivativeIntensityVector)
          #cv.WaitKey()

	  print "One_NormalizedtmpLandmarkDerivativeIntensityVector[0][p_num] =%s"%(One_NormalizedtmpLandmarkDerivativeIntensityVector[0][p_num])
          #cv.WaitKey()


          #x is the intensity vector for this profile , g , for each landmark (contains 6 search profiles x 13 subprofiles)
          x=np.array(One_NormalizedtmpLandmarkDerivativeIntensityVector[0][p_num])
          #y is the mean intensity vector for the 6 profiles (shouldn't this be 7 instead of 6 profiles ??? )..--> no, cause we are taking the derivative! so there will be n-1. Hence ..6
          y=np.array(trainingGmean[currentSearchPyramidLevel][p_num][0]) #What am i supposed to get here?? The ith landmark's training G mean (containing 2d intensity info 6 (7-1 dg) whisker points and 13 tangent to normal points)
          #x=x.tolist()
          #y=y.tolist()

	  #print "x=%s"%(repr(x))
          ##cv.WaitKey()

	  #print "y=%s"%(repr(y))
          ##cv.WaitKey()


	  ###print "len trainingGmean=%s"%(len(trainingGmean))
          ####cv.WaitKey()
	  ###print "len trainingGmean[0]=%s"%(len(trainingGmean[0]))
          ####cv.WaitKey()
	  ###print "len trainingGmean[0][0]=%s"%(len(trainingGmean[0][0]))
          ####cv.WaitKey()


          #Now for each landmark one MYCOVAR ..does it relate to mahalanobis calculation correctly


          #################################
          #not currently used bu maybe should constider using it
          #mycovar = np.cov(x, y, rowvar=0)
	  ###print "mycovar=%s"%(mycovar)
	  ###print "mycovar=%s"%(len(mycovar))
          #c=###cv.WaitKey()
          #if c==1048603 :#whichever integer key code makes the app exit
          #    exit()

          #mycovar=0
          #mycovar=cv2.calcCovarMatrix(x, mycovar, y, cv.CV_COVAR_NORMAL )#| cv.CV_COVAR_COLS



	  ###print "mycovar=%s"%(mycovar)
          ####cv.WaitKey()
          #################################

	  ####print"AllImages_NormalizedtmpLandmarkDerivativeIntensityVector[%d][%d]=%s\n"%(index,p_num,x)
	  ####print"y_j_mean[0]=%s\n"%(y)


          #calculate Mahalanobis best smaller distance
          #covar = np.cov(x, rowvar=0)
	  ###print "covar=%s"%(covar)
	  ###print "covar=%s"%(len(covar))
          #c=###cv.WaitKey()
          #if c==1048603 :#whichever integer key code makes the app exit
          #    exit()
          ####cv.WaitKey()

          #invcovar = np.linalg.inv(covar.reshape((13,13)))

          #invcovar=0
          #invcovar=cv2.invert(mycovar, invcovar, cv2.DECOMP_SVD)#'''covar'''
          #invcovar=invcovar[1]

	  ###print "invcovar=%s"%(invcovar)
	  ###print "invcovar=%s"%(len(invcovar))
          #c=###cv.WaitKey()
          #if c==1048603 :#whichever integer key code makes the app exit
          #    exit()

	  ####print"inverse covariance matrix shape=%s"%(str(invcovar.shape))

          bestProfileMatchWithSmallerMahalanobisDistanceFrom_gProf_to_meanProf=-1;#an intentionally negative init number
          bestIndexProfile=-1



          #SORT OF WORKING LOGIG OF INVERSE COVARIANCE MATRIX CALCULATION WORKING
          #mycovar = np.cov(x[j], y[j], rowvar=0)

          #diff=x-y
          #mycovar=0
          #mycovar = np.cov(diff, rowvar=0)

          #turn numpy arrays to CvMats
          #xRows = x.shape[0]#rows
          #xColumns = x.shape[1]#columns
          #yRows = y.shape[0]#rows
          #yColumns = y.shape[1]#columns

          #a = np.zeros((xRows, xColumns))
          #x = cv.fromarray(x)
	  ###print x
          #y = cv.fromarray(y)
	  ###print y

          ##MAJOR CHANGE...........
          #cv.CalcCovarMatrix(x, mycovar, y, cv.CV_COVAR_COLS | cv.CV_COVAR_NORMAL | cv.CV_COVAR_USE_AVG)
          #mycovar= cv2.calcCovarMatrix(x, cv.CV_COVAR_COLS,cv.CV_64F | cv.CV_COVAR_NORMAL | cv.CV_COVAR_USE_AVG, y[0])#http://www.programmershare.com/3791731/

          mycovar=self.CovarianceMatricesVec[currentSearchPyramidLevel][p_num]#self.tmpCovar
	  #print "mycovar=%s"%(repr(mycovar))
	  #print "mycovar=%s"%(len(mycovar))
          #c=#cv.WaitKey()
          #if c==1048603 :#whichever integer key code makes the app exit
          #    exit()

          ##MAJOR CHANGE...........

          #divide covariance by row number (NEEDED IF COVARIANCE MATRIX HAS BEEN CALCULCATED WITH CV.CALCCOVARIANCEMATRIX() METHOD, as it's not normalized)
	  ##print "mycovar before element division\n"
	  ##print (mycovar[0])

          #for i in mycovar[0]:
          #    i /= 6 #6 is the number of rows


	  ##print "mycovar\n"
	  ##print (mycovar[0])
	  ###print "mycovar=%s"%(mycovar)
	  ###print "mycovar=%s"%(len(mycovar))
          #c=###cv.WaitKey()
          #if c==1048603 :#whichever integer key code makes the app exit
          #    exit()

          invcovar=0

          #WAS
          invcovar=cv2.invert(mycovar, invcovar, cv2.DECOMP_SVD)#mycovar[0] when used with cv2.calcCovarMatrix #'''OR covar'''
          invcovar=invcovar[1]

          #does not converge properly
          #invcovar=scipy.linalg.pinv(mycovar)

          #invcovar = np.linalg.solve(invcovar.T.dot(invcovar), invcovar.T)


          #NOW IS
          #invcovar = np.linalg.inv(mycovar)



	  ##print "invcovaself.currentSearchPyramidLevelr=%s"%(invcovar)
          #c=###cv.WaitKey()
          #if c==1048603 :#whichever integer key code makes the app exit
          #  exit()


	  #print "invcovar=%s"%(repr(invcovar))
	  #print "invcovar=%s"%(len(invcovar))
          #c=#cv.WaitKey()
          #if c==1048603 :#whichever integer key code makes the app exit
          #    exit()


	  ###print"total_of %d g search profile elements\n"%(x.shape[0])
          ######cv.WaitKey()

          #mahalfile = open('mahalanobisChoiceCheeck/normDerIntProf.txt'+str(p_num),'w')


          #Save all sums of sub tangential subprofiles, of the whisker points --->(all 13 tangential elements' sum)
          SumSubprofElemVecX=[]
          SumSubprofElemVecY=[]

          #for this landmark the are 6 profiles (13 elements each) of which we find the best mahalanobis for each search profile to select the bestIndexProfile accordingly
          for j in range(x.shape[0]):
          #for j in range(len(x)):


              #s = np.array([[20,55,119],[123,333,11],[113,321,11],[103,313,191],[123,3433,1100]])
	      ###print "x[j]=%s"%(x[j])
              ####cv.WaitKey()
              #co = np.cov(s[0],s[1], rowvar=0)
	      ###print "co=%s"%(co)
              ####cv.WaitKey()


	      #print "x[j]=%s"%(x[j])
              #cv.WaitKey()
	      ##print "y=%s"%(y)
              ####cv.WaitKey()

	      ###print "x[j]-y[j]=%s"%(x[j]-y[j])
              ####cv.WaitKey()


              #mycovar = np.cov(x[j], y[j], rowvar=0)

              #diff=x-y
              #mycovar = np.cov(diff, rowvar=0)

	      ###print "mycovar=%s"%(mycovar)
	      ###print "mycovar=%s"%(len(mycovar))
              #c=###cv.WaitKey()
              #if c==1048603 :#whichever integer key code makes the app exit
              #    exit()

              #invcovar=0
              #invcovar=cv2.invert(mycovar, invcovar, cv2.DECOMP_SVD)#'''covar'''
              #invcovar=invcovar[1]

	      #print "invcovar=%s"%(invcovar)
	      ###print "invcovar=%s"%(len(invcovar))
              #c=cv.WaitKey()
              #if c==1048603 :#whichever integer key code makes the app exit
              #    exit()

              #repr is for formating numpy arrays with commas
	      #print "x=%s"%(repr(x))
              #cv.WaitKey()
	      #print "y=%s"%(repr(y))
              #cv.WaitKey()

	      ####print"test g%d %s...with...gmean%d %s"%(j, x[j],j, y[j])

              #PRACTICALLY COMPARING 'HOW DIFFERENT THE NORMALIZED DERIVATIVES OF THE CURRENT/LIVE SHAPE ON THE TARGET IMAGE IS with regards to..
              #THE NORMALIZED DERIVATIVES OF THE GMEAN SHAPE (repeat for each level)
              tmpMahalDist=scipy.spatial.distance.mahalanobis( x[j], y[j], invcovar)# currently works with fabs applied to inv covar matrix..however this should not be the case
              #tmpMahalDist=scipy.spatial.distance.mahalanobis(x,y,invcovar);
	      ####print"g%d MahalanobisDistance=%s\n"%(j,tmpMahalDist)

              #NEW WAY OF CALCULATION OF THE MAHALANOBIS
              #tmpMahalDist=self.MahalanobisDist(x[j], y[j])#invcovar


              #mahalfile.write(str(x[j])+"\n")
	      #print "J=%s"%(j)
              #exitONKeyPress()

              #first time it runs for each landmark
              if bestProfileMatchWithSmallerMahalanobisDistanceFrom_gProf_to_meanProf==-1:
                  bestProfileMatchWithSmallerMahalanobisDistanceFrom_gProf_to_meanProf= tmpMahalDist
                  bestIndexProfile=j


              else:
                  if tmpMahalDist < bestProfileMatchWithSmallerMahalanobisDistanceFrom_gProf_to_meanProf:
                      bestProfileMatchWithSmallerMahalanobisDistanceFrom_gProf_to_meanProf= tmpMahalDist
                      bestIndexProfile=j
		      ###print "bestProfileMatchWithSmallerMahalanobisDistanceFrom_gProf_to_meanProf for this landmark=%s"%(bestProfileMatchWithSmallerMahalanobisDistanceFrom_gProf_to_meanProf)







	      print "j=%d,   tmpMahalDist FOR THIS landmark=%s"%(j,tmpMahalDist)
              #cv.WaitKey()



              #Add up all tangential subprofile elements
              #Check x[i] & y[i] sums with regards to the intensity of the final pixel chosen
              #x[i]
              #              SumSubprofElem=0
              #              for subprofElem in x[j]:
              #                  SumSubprofElem+=subprofElem
	      #                  ##print "subprofElemX=%f"%(subprofElem)
              #                  ####cv.WaitKey()
	      #              ##print "\nSumSubprofElemX=%f"%(SumSubprofElem)
              #              SumSubprofElemVecX.append(SumSubprofElem)
              #              #y[i]
              #              SumSubprofElem=0
              #              for subprofElem in y[j]:
              #                  SumSubprofElem+=subprofElem
	      #                  ##print "subprofElemY=%f"%(subprofElem)
              #                  ####cv.WaitKey()
	      #              ##print "\nSumSubprofElemY=%f"%(SumSubprofElem)
              #              SumSubprofElemVecY.append(SumSubprofElem)



	  ##print"profile g%d has the bestProfileMatchWithSmallerMahalanobisDistanceFrom_gProf_to_meanProf=%s"%(bestIndexProfile,bestProfileMatchWithSmallerMahalanobisDistanceFrom_gProf_to_meanProf)
          ####cv.WaitKey()

	  ####print"g%d model profile chosen:\n%s"%(bestIndexProfile,x[bestIndexProfile])
	  ####print"landmark%d to be corrected based on g%d best profile model"%(p_num,bestIndexProfile)
          ######cv.WaitKey()



          ##########################################################################################
          ##########################################################################################
          #calculate the normal corresponding to point p_num of the mean shape

	  ####print"totalLandmarkNumber=%d"%(totalLandmarkNumber)
	  ####print"currentShape=%d"%(len(currentShape.pts))
	  ####print"\n\npoint number=%d\n\n"%(p_num)
          norm = currentShape.get_normal_to_point(p_num)
	  ####print"norm\n",norm

          #if p_num==29 :
          #    #####cv.WaitKey()

          #get p, each of the 2d points of the mean shape shape in the training model
          p = currentShape.pts[p_num]#1st point ..2nd point.. and so on

	  ####print"p_num%d"%(p_num)
          ######cv.WaitKey()
	  ####print"p=%s"%(p)
          ######cv.WaitKey()

	  ####print"Points of the mean shape of the Trained Set",currentShape.pts

          # Scan/Sample over the whisker
          max_pt = p
          max_edge = 0


          #'''
          #New Way (Mahalanobis based chosen point)
          #based on: 2(m - k) + 1  , from http://www.face-rec.org/algorithms/AAM/app_models.pdf
          #m from 0<=m<=5

          #normalize the 0..6 to -3..3 range
          eithersideOfNormalPixels=3

          #genericlPxlOffset scale sampler scales  on each side of the normal, meaning that:
          #(-3 -->becomes -9), (-2 -->becomes -6), (-1 -->becomes -3), (0 -->becomes 0), (1 -->becomes 3), (2 -->becomes 6), (3 -->becomes 9)
          genericlPxlOffset=1

          #Pixel selected based on mahalanobis minimization
          side_center= genericlPxlOffset*(bestIndexProfile-eithersideOfNormalPixels) # (-5..-3..-1...1...3..5..)  , #bestIndexProfile-3
          #
          side_center_plus= genericlPxlOffset*((bestIndexProfile+genericlPxlOffset)-eithersideOfNormalPixels)
          #
          side_center_minus= genericlPxlOffset*((bestIndexProfile-genericlPxlOffset)-eithersideOfNormalPixels)

	  print "shapes length = %d"%(x.shape[0])
	  #print "bestIndexProfile is %d..so final chosen side is: %d"%(bestIndexProfile,side_center)
	  print "bestIndexProfile is %d..so final chosen side is: %d"%(bestIndexProfile,side_center)

          #compare with previous iteration saved, and if <=10% is different, means that they don't tend to vary that much, so STOP search!
          #(Benjelloun et. al 2011), 'A Framework of Vertebra Segmentation Using the Active Shape Model-Based Approach'
          self.savedBestIndicesCurrentIteration[p_num] = bestIndexProfile
          if iterationIndex==0:
              self.savedBestIndicesPreviousIteration.append(-1)#to make sure zip(self.savedBestIndicesCurrentIteration, self.savedBestIndicesPreviousIteration) further down when comparing will NOT return an empty list


	  print "iterationIndex=%d, savedBestIndicesCurrentIteration=%s"%(iterationIndex,self.savedBestIndicesCurrentIteration)


          #cv.WaitKey()


	  ##print "x.shape[0]/2=%d"%(x.shape[0]/2)
          ####cv.WaitKey()
	  ##print "len(One_NormalizedtmpLandmarkDerivativeIntensityVector[0])/2=%d"%(len(One_NormalizedtmpLandmarkDerivativeIntensityVector[0])/2)
          ####cv.WaitKey()

          #We record the number of times that the best found pixel along a search profile is within the central 50% of the profile

          ninentyPerCent=1.0/10.0
          halfShapeProfile = (x.shape[0]/2) #3
          stepUp = halfShapeProfile/2  #1
          stepDown = - halfShapeProfile/2 #-1



	  #print "x.shape=%s < x.shape[0]=%s"%(x.shape, x.shape[0])
	  print "( bestIndexProfile=%d > halfShapeProfile(%d)+stepDown(%d) = %d )"%(bestIndexProfile, halfShapeProfile, stepDown, halfShapeProfile+stepDown)
          #cv.WaitKey()
	  print "( bestIndexProfile=%d < halfShapeProfile(%d)+stepUp(%d) = %d )"%(bestIndexProfile, halfShapeProfile,stepUp , halfShapeProfile+stepUp)
          #cv.WaitKey()

          if ( bestIndexProfile > halfShapeProfile+stepDown ) and  ( bestIndexProfile < halfShapeProfile+stepUp ) and self.currentSearchPyramidLevel >=1:#as it can't go further down than 0 level in res pyramid

	      #print "bestIndexProfile=%d < halfShapeProfile=%d"%(bestIndexProfile,x.shape[0]/2)
              #cv.WaitKey()

              bestIndexWithinHalfProfileLengthCounter+=1

	      #print "COMPARE: %d==%d"%(bestIndexWithinHalfProfileLengthCounter,float(len(One_NormalizedtmpLandmarkDerivativeIntensityVector[0])*ninentyPerCent))

	      #print "CURRENT PYRAMIDLEVEL : %d"%(self.currentSearchPyramidLevel)
              ###cv.WaitKey()

	      print "INCREASE!! bestIndexWithinHalfProfileLengthCounter=%d  == float(len(One_NormalizedtmpLandmarkDerivativeIntensityVector[0])*ninentyPerCent)=%d"%(bestIndexWithinHalfProfileLengthCounter, float(len(One_NormalizedtmpLandmarkDerivativeIntensityVector[0])*ninentyPerCent))
              #cv.WaitKey()

              #A convergence criterion is that a sufficient number of landmarks are reached (in this case we choose 90% of the total number of landmarks)
              if bestIndexWithinHalfProfileLengthCounter == float(len(One_NormalizedtmpLandmarkDerivativeIntensityVector[0])*ninentyPerCent):


		  print "YEAH bestIndexWithinHalfProfileLengthCounter == (len(One_NormalizedtmpLandmarkDerivativeIntensityVector[0])*ninentyPerCent): %d=%d"%(bestIndexWithinHalfProfileLengthCounter , (len(One_NormalizedtmpLandmarkDerivativeIntensityVector[0])*ninentyPerCent))
                  cv.WaitKey()

                  self.currentSearchPyramidLevel-=1
                  breaktonextpyramidlevel=True

		  print "MOVE TO NEXT PYRAMID LEVEL AS a sufficient number MAX ITERATIONS SPECIFIED has been reached (CURENTLY is SET to %d)"%(preSpecifiedNumberOfLevelIterations)
                  #cv.WaitKey()

		  #print "now fill the rest from %d to %d !! "%( len(correctionVectorForAllShapes), len(One_NormalizedtmpLandmarkDerivativeIntensityVector[0]))
                  ###cv.WaitKey()



                  '''BUT BEFORE that we need to fill the rest of correctionVectorForAllShapes with the same non-modified elements already there from One_NormalizedtmpLandmarkDerivativeIntensityVector[0][p_num]'''
                  for restUnmodifiedindex in range ( len(correctionVectorForAllShapes) , len(One_NormalizedtmpLandmarkDerivativeIntensityVector[0]), 1 ):
                      correctionVectorForAllShapes.append( currentShape.pts[restUnmodifiedindex] )

		      #print "len correctionVectorForAllShapes %d"%( len(correctionVectorForAllShapes) )
                      ###cv.WaitKey()


		  #print "break to next pyramidlevel: [%d]"%(self.currentSearchPyramidLevel)
                  f.close()
                  return correctionVectorForAllShapes
                  ###cv.WaitKey()
                  break

          if ((iterationIndex+1)%(preSpecifiedNumberOfLevelIterations+1))==0  and self.currentSearchPyramidLevel >=1 :

	      #print "(iterationIndex+1) mod (preSpecifiedNumberOfLevelIterations+1)==0): %d=0"%(((iterationIndex+1)%(preSpecifiedNumberOfLevelIterations+1)))
              ###cv.WaitKey()

              self.currentSearchPyramidLevel-=1
              breaktonextpyramidlevel=True

	      print "MOVE TO NEXT PYRAMID LEVEL AS a sufficient number MAX ITERATIONS SPECIFIED has been reached (CURENTLY is SET to %d)%",(preSpecifiedNumberOfLevelIterations)
              #cv.WaitKey()

              '''BUT BEFORE that we need to fill the rest of correctionVectorForAllShapes with the same non-modified elements already there from One_NormalizedtmpLandmarkDerivativeIntensityVector[0][p_num]'''
              for restUnmodifiedindex in range ( len(correctionVectorForAllShapes) , len(One_NormalizedtmpLandmarkDerivativeIntensityVector[0]), 1 ):
                  correctionVectorForAllShapes.append( currentShape.pts[restUnmodifiedindex] )

		  #print "len correctionVectorForAllShapes %d"%( len(correctionVectorForAllShapes) )
                  ###cv.WaitKey()


	      #print "break to next pyramidlevel: [%d]"%(self.currentSearchPyramidLevel)
              f.close()
              return correctionVectorForAllShapes
              ###cv.WaitKey()

              break


          #Hack the multi-resolution model by specifying a percentage based on 100 iterations

          #if iterationIndex<=10:
          #    side_center = side_center*4

          #elif iterationIndex>33 and iterationIndex<66:
          #    side_center = side_center*2
          #elif iterationIndex>=66 and iterationIndex:
          #    side_center = side_center*1



	  ##print"side_center=%d"%(side_center)
          ######cv.WaitKey()

	  #print"length of g_image=%d"%(len(self.g_image))
          ####cv.WaitKey()

          f.write('for landmark '+str(p_num)+') '+str(side_center)+'     is chosen')
          f.write("\n")

          imgtmp = cv.CreateImage(cv.GetSize(self.greyTargetImage[currentSearchPyramidLevel]), self.greyTargetImage[currentSearchPyramidLevel].depth, 1)
          cv.Copy(self.greyTargetImage[currentSearchPyramidLevel], imgtmp)

          #along the whisker
          #for side_center in range(-3,4):#along normal profile
          # Normal to normal...

          #new_p = Point(p.x + side_center*-norm[1], p.y + side_center*norm[0])#why ..the other way around?

          #calculate new point on the normal
          new_p = Point(p.x + (side_center)*norm[0], p.y + (side_center)*norm[1])


          #SHOW ALL POSSIBLE CHOICE BASED ON MAHALANOBIS WITH BLUE COLOR plus THE ACTUAL CHOICE WITH GREEN COLOR
          #TESTCHOICE = cv.CreateImage(cv.GetSize(self.image), self.image.depth, 3)
          #cv.Copy(self.image, TESTCHOICE)

          whiskerKeyCheck=0#cv.WaitKey(1)

          #check for 'w' (to step into individual points along the whisker)
          if whiskerKeyCheck == 1048695:
               WHISKER_CHOICE=1
          else:
               WHISKER_CHOICE=0

          #stop and show possible whisker points
          if WHISKER_CHOICE==1:
              cv.WaitKey()


          for j in range(x.shape[0]):#for all the profile elements

	      print j

              mypoint = Point(p.x + (genericlPxlOffset*(j-eithersideOfNormalPixels))*norm[0], p.y + (genericlPxlOffset*(j-eithersideOfNormalPixels))*norm[1])


              #if ( int(new_p.x)==int(mypoint.x) and int(new_p.y)==int(mypoint.y) ):
              #    cv.Circle(CHOSEN_BEST, ( int(mypoint.x), int(mypoint.y) ), 1, (0,255,0))
              #else:



              #cv.Circle(CHOSEN_BEST, ( int(mypoint.x), int(mypoint.y) ), 1, (255,0,0))
              #cv.NamedWindow("CHOSEN_BEST", cv.CV_WINDOW_NORMAL)
	      #cv.ShowImage("CHOSEN_BEST",CHOSEN_BEST)


	      print "double check whisker intensity values [%d, %d]=%s "%(int(mypoint.y),  int(mypoint.x), targetImageTocheckAgainst[int(mypoint.y),  int(mypoint.x)]  )
              #cv.WaitKey()

              cv2.circle(CHOSEN_BEST, ( int(mypoint.x), int(mypoint.y) ), 1, (255,0,0))
              cv2.namedWindow("CHOSEN_BEST", cv.CV_WINDOW_NORMAL)
	      cv2.imshow('CHOSEN_BEST',CHOSEN_BEST)

              if WHISKER_CHOICE==1:
                  cv.WaitKey()


              #cv.NamedWindow("TESTCHOICE", cv.CV_WINDOW_NORMAL)
	      #cv.ShowImage("TESTCHOICE",TESTCHOICE)
              #cv.NamedWindow("CHOSEN_BEST", cv.CV_WINDOW_NORMAL)
	      #cv.ShowImage("CHOSEN_BEST",CHOSEN_BEST)
	      print "currentSearchPyramidLevel=%d"%(currentSearchPyramidLevel)


          #cv.Circle(TESTCHOICE, ( int(new_p.x), int(new_p.y) ), 1, (0,255,0))
          #cv.NamedWindow("TESTCHOICE", cv.CV_WINDOW_NORMAL)
          #c=cv.WaitKey()


          #if ((x >= 0 and x <= self.image.width) and (y >= 0 and y <= self.image.height)):

          #cv.WaitKey()

	  #print "targetImageTocheckAgainst[int(%f-1), int(%f-1)] %s"%(new_p.y, new_p.x,  targetImageTocheckAgainst[int(new_p.y-1), int(new_p.x-1)])
          #cv.WaitKey()
	  #print np.asarray(self.greyTargetImage[currentSearchPyramidLevel])
          #cv.WaitKey()

          ## The new_p is calculated based on the side chosen which is based on the minimization of the mahalanobis distance amongst normalized derivative profile vectors for this landmark
          npx=int(new_p.x)
          npy=int(new_p.y)

          #cv.Circle(CHOSEN_BEST, ( npx, npy ), 1, (0,0,255))
	  #print "targetImageTocheckAgainst[%d , %d] = %d"%(npy, npx,targetImageTocheckAgainst[npy, npx])
          #cv.NamedWindow("CHOSEN_BEST", cv.CV_WINDOW_NORMAL)
	  #cv.ShowImage("CHOSEN_BEST",CHOSEN_BEST)

          cv2.circle(CHOSEN_BEST,( npx, npy ), 1, (0,0,255))
	  #print "targetImageTocheckAgainst[%d , %d] = %d"%(npy, npx,targetImageTocheckAgainst[npy, npx])
          cv2.namedWindow("CHOSEN_BEST", cv.CV_WINDOW_NORMAL)
	  cv2.imshow('CHOSEN_BEST',CHOSEN_BEST)


          if WHISKER_CHOICE==1:
              cv.WaitKey()

          #Draw rotateRectangle
          #http://stackoverflow.com/questions/3365171/calculating-the-angle-between-two-lines-without-having-to-calculate-the-slope
          #http://opencvpython.blogspot.co.uk/2012/06/contours-2-brotherhood.html
          numpyImage = np.asarray(targetImageTocheckAgainst[:,:])
          im_height, im_width = numpyImage.shape
          tmp = cv2.cvtColor(numpyImage,cv2.COLOR_GRAY2BGR)

          ##OUTSIDE OBJECT TEST BOX AREA / GRAB HISTOGRAM surrounding the area starting from the pixel chosen based on mahalanobis distance extende for
          ##and then put it in the dictionary of previous iterations
          ## (extent the current prevBestHistShapePointsList to include 2 instead of one rectangles)

          #
          offsetByPixels=4
          whisker_center_point = Point(p.x + (side_center+offsetByPixels)*norm[0], p.y + (side_center+offsetByPixels)*norm[1])
          whisker_center_plus_point = Point(p.x + (side_center_plus+2)*norm[0], p.y + (side_center_plus+2)*norm[1])
          #whisker_center_point_plusOnXAxisOnly =  Point(p.x + (side_center_plus)*norm[0], p.y)


          centerX = whisker_center_point.x
          centerY = whisker_center_point.y
          ##the following 2 lines are used just to calculate the angle of rotation of the rectangle area of the histogram
          plused_npx     = whisker_center_plus_point.x
          plused_npy     = whisker_center_plus_point.y
          #Save them for later use it for later use
          #
          centerX_plus4_OfsetPxls= centerX
          centerY_plus4_OfsetPxls= centerY
          plused_npx_plus4_OfsetPxls= plused_npx
          plused_npy_plus4_OfsetPxls= plused_npy
          #
          #Save the above for later use it for later use

          width=13
          height=7

          #find angle between a line horizontal to the x axis starting from the landmark origin &
          # a line starting from the origin and ending at the further point along the whisker
          angle=getAngleBetween2Lines(centerX,centerY,centerX+10,centerY, centerX,centerY,plused_npx,plused_npy) * (180.0/math.pi)

	  print angle
          rect = ((centerX,centerY),(width,height), -angle-90)#rotate and place horizontaly
          box = cv2.cv.BoxPoints(rect)
          box = np.int0(box)

          ## make sure we grab and compare only against the original TARGET image histograms, NOT derivatives of it
          #originalImageTargetConverted = np.asarray(self.image)
          originalImageTargetConverted = cv2.imread(sys.argv[2],cv2.IMREAD_GRAYSCALE)
          #needed casue we cannot draw on the originalImageTargetConverted, otherwise the coloured rectangles drawned will modify the image itself
          drawDebugoriginalImageTargetConverted = cv2.imread(sys.argv[2],cv2.IMREAD_GRAYSCALE)

          cv2.drawContours(drawDebugoriginalImageTargetConverted,[box],0,(255,0,255))
          cv2.namedWindow("drawDebugoriginalImageTargetConverted", cv.CV_WINDOW_NORMAL)
	  cv2.imshow('drawDebugoriginalImageTargetConverted',drawDebugoriginalImageTargetConverted)

          cv2.namedWindow("originalImageTargetConverted", cv.CV_WINDOW_NORMAL)
	  cv2.imshow('originalImageTargetConverted',originalImageTargetConverted)

          '''
          cv2.drawContours(CHOSEN_BEST,[box],0,(255,0,255))
          cv2.namedWindow("CHOSEN_BEST", cv.CV_WINDOW_NORMAL)
	  cv2.imshow('CHOSEN_BEST',CHOSEN_BEST)
          cv.WaitKey()
          '''


          ## now extract histogram of the outside box,by extrating the skewed outside box area only
          outsideBoxImage = rotatedRectExtractionArea(originalImageTargetConverted,centerX,centerY,plused_npx,plused_npy)
          # Convert BGR to Gray if needed

          outsideBoxImageGrayed=outsideBoxImage
          if len(outsideBoxImage.shape)==3:
	      print'colored image needs to be converted to grayscale'
              outsideBoxImageGrayed = cv2.cvtColor(outsideBoxImage, cv2.COLOR_BGR2GRAY)

          outsideBoxImage=outsideBoxImageGrayed
          outsideBoxImageHist = cv2.calcHist([outsideBoxImage],[0],None,[256],[0,256])

          #show the hist of outsideBoxImageHist
          ##showHistogram(outsideBoxImageHist,'outsideBoxImageHist')
          #cv.WaitKey()



          ##INSIDE OBJECT TEST BOX AREA / GRAB HISTOGRAM and then put it in the dictionary of previous iterations
          ## (extent the current prevBestHistShapePointsList to include 2 instead of one rectangles)
          offsetByPixels=-4
          whisker_center_point = Point(p.x + (side_center+offsetByPixels)*norm[0], p.y + (side_center+offsetByPixels)*norm[1])#NOTE offsetByPixels changes to -4 to identify 'inside object box window'
          whisker_center_plus_point = Point(p.x + (side_center_plus+2)*norm[0], p.y + (side_center_plus+2)*norm[1])
          #whisker_center_point_plusOnXAxisOnly =  Point(p.x + (side_center_plus)*norm[0], p.y)

          centerX = whisker_center_point.x
          centerY = whisker_center_point.y
          plused_npx     = whisker_center_plus_point.x
          plused_npy     = whisker_center_plus_point.y
          #Save them for later use it for later use
          #
          centerX_minus4_OfsetPxls= centerX
          centerY_minus4_OfsetPxls= centerY
          plused_npx_minus4_OfsetPxls= plused_npx
          plused_npy_minus4_OfsetPxls= plused_npy
          #
          #Save the above for later use it for later use
          width=13
          height=7

          #find angle between a line horizontal to the x axis starting from the landmark origin &
          # a line starting from the origin and ending at the further point along the whisker
          angle=getAngleBetween2Lines(centerX,centerY,centerX+10,centerY, centerX,centerY,plused_npx,plused_npy) * (180.0/math.pi)

	  print angle
          rect = ((centerX,centerY),(width,height), -angle-90)#rotate and place horizontaly
          box = cv2.cv.BoxPoints(rect)
          box = np.int0(box)          

          cv2.drawContours(drawDebugoriginalImageTargetConverted,[box],0,(0,255,255))
          cv2.namedWindow("drawDebugoriginalImageTargetConverted", cv.CV_WINDOW_NORMAL)
	  cv2.imshow('drawDebugoriginalImageTargetConverted',drawDebugoriginalImageTargetConverted)

          cv2.namedWindow("originalImageTargetConverted", cv.CV_WINDOW_NORMAL)
	  cv2.imshow('originalImageTargetConverted',originalImageTargetConverted)

          '''
          cv2.drawContours(CHOSEN_BEST,[box],0,(0,255,255))
          cv2.namedWindow("CHOSEN_BEST", cv.CV_WINDOW_NORMAL)
	  cv2.imshow('CHOSEN_BEST',CHOSEN_BEST)
          cv.WaitKey()
          '''



          ##now extract histogram of the inside yellow box,by extrating the skewed inside box area only
          insideBoxImage = rotatedRectExtractionArea(originalImageTargetConverted,centerX,centerY,plused_npx,plused_npy)
          # Convert BGR to Gray if needed
          insideBoxImageGrayed=insideBoxImage
          if len(insideBoxImage.shape)==3:
	      print'colored image needs to be converted to grayscale'
              insideBoxImageGrayed = cv2.cvtColor(insideBoxImage, cv2.COLOR_BGR2GRAY)
          insideBoxImage=insideBoxImageGrayed
          insideBoxImageHist = cv2.calcHist([insideBoxImage],[0],None,[256],[0,256])

          #show the hist of insideBoxImage
          #showHistogram(insideBoxImageHist,'insideBoxImage')
          #cv.WaitKey()

          ## Now, compare the two areas (outside/inside) based on their histograms, AGAINST THE  H A R D C O D E D
          ## RECTANGLE ON THE IMAGE, that belongs to a part of the vertebra
          target_x1=235#175
          target_x2=250#200
          target_y1=125#100
          target_y2=140#125
          targetImage = cv2.imread( sys.argv[2],cv2.IMREAD_GRAYSCALE)
          targetRoi = hist_lines(targetImage,target_x1,target_x2,target_y1,target_y2)#x1x2 y1y2

          outsideBoxMatch = cv2.compareHist(outsideBoxImageHist,targetRoi,cv2.cv.CV_COMP_CORREL)
          insideBoxMatch = cv2.compareHist(insideBoxImageHist,targetRoi,cv2.cv.CV_COMP_CORREL)

          #if method 0/CV_COMP_CORREL has been used in compareHist then->: the highest the value, the more accurate the similarity
          if outsideBoxMatch > insideBoxMatch:
		  print "outsideBox (purple) more_similar is more similar to targetRoi"
                  outsideBoxBetterMatch=True
                  insideBoxBetterMatch=False
          else:
		  print "insideBox (yellow) is more similar to targetRoi"
                  insideBoxBetterMatch=True
                  outsideBoxBetterMatch=False


          #cv2.waitKey(0)


          normX_TopLeft=0
          normY_TopLeft=0
          normX_BottomRight=0
          normY_BottomRight=0

          movePoint=0

          #use the plus4_Offset Pixels coordinates saved (belonging to the outter histogram window)
          if outsideBoxBetterMatch == True:
              normX_TopLeft = centerX_plus4_OfsetPxls
              normY_TopLeft = centerY_plus4_OfsetPxls
              normX_BottomRight = plused_npx_plus4_OfsetPxls
              normY_BottomRight = plused_npy_plus4_OfsetPxls

              #UPDATE new point npx, npy to ----> increase offset to the plus side along the normal..so..move towards the outside of the target object
	      offsetByPixels= +1
              movePoint = Point(p.x + (side_center+offsetByPixels)*norm[0], p.y + (side_center+offsetByPixels)*norm[1])

          elif insideBoxBetterMatch == True:
              normX_TopLeft = centerX_minus4_OfsetPxls
              normY_TopLeft = centerY_minus4_OfsetPxls
              normX_BottomRight = plused_npx_minus4_OfsetPxls
              normY_BottomRight = plused_npy_minus4_OfsetPxls

              #UPDATE new point npx, npy to ----> decrease offset to the plus side along the normal..so..move towards the inside of the target object
	      offsetByPixels= -1
              movePoint = Point(p.x + (side_center+offsetByPixels)*norm[0], p.y + (side_center+offsetByPixels)*norm[1])



          #new_npx = Point((normX_TopLeft)-6, (normY_TopLeft)-3)
	  #print "new_p_TopLeft=%s"%(new_npx)
          #new_npy = Point((normX_BottomRight)-6, (normY_BottomRight)-3)
	  #print "new_p_BottomRight=%s"%(new_npy)

          #npX_updated = (normX_TopLeft + normX_BottomRight)/2
          #npY_updated = (normY_TopLeft + normY_BottomRight)/2


          #This point "npx, npy" is likely to represent an edge (as targetImageTocheckAgainst contains the 1st derivative of the actual image,
          #smoothed for thislevel)
          #So, check the intensity resemblance first..then further down check the texture resemblance too
          #(more specifically check which of the 2 histograms holds a value closer to the target image,
          #and update accordingly towards the inned or the outter histwindow point.
          #The original aim is to identify and differ between the inner object texture and the outside object textrue according to Van Ginneken et al. point along the normal 'inside' or 'outside' of the target object)

	  print "targetImageTocheckAgainst[%d, %d]=%s >\t max_edge=%s"%(npy, npx, targetImageTocheckAgainst[npy, npx] , max_edge )
          #cv.WaitKey()

          #if targetImageTocheckAgainst[npy, npx] > max_edge:
          if True:
          #if targetImageTocheckAgainst[npy, npx] > 0:#since_max edge is always 0, there's no point in comparing against it

	    ## Print the highest pixel intensity value along the whisker that's not black
            ##(as targetImageTocheckAgainst is grayscale, and max_edge starts at 0 for every landmark whisker at every iteration)


            #WATCH HISTOGRAM CALCULATIONS/MATCHINGS ARE CURRENTLY NOT USED
	    npx=movePoint.x
	    npy=movePoint.y

            #ADD HIST2 comparison in here
	    ### HERE WE NEED TO PERFORM SOME SORT OF HISTOGRAM MATCHING (IN/OUT OF THE OBJECT)
            ### SO AS TO DECIDE UPON THIS AS WELL AND NOT ONLY USING THE WEIGHTED GRADIENT INTENSITY METRIC
            ### compare target image ROI x1,x2 y1,y2 with precalculated value for this target area of the image

            #new possible "to update to" point window (offset normal precalculated by -5,5 points)
            new_p_TopLeft = Point(p.x + (side_center)*norm[0]-(2*genericlPxlOffset), p.y + (side_center)*norm[1]-genericlPxlOffset)
	    #print "new_p_TopLeft=%s"%(new_p_TopLeft)
            new_p_BottomRight = Point(p.x + (side_center)*norm[0]+(2*genericlPxlOffset), p.y + (side_center)*norm[1]+genericlPxlOffset)
	    #print "new_p_BottomRight=%s"%(new_p_BottomRight)


            #new_p_TopLeft = Point((normX_TopLeft)-6, (normY_TopLeft)-3)
	    #print "new_p_TopLeft=%s"%(new_p_TopLeft)
            #new_p_BottomRight = Point((normX_BottomRight)-6, (normY_BottomRight)-3)
	    #print "new_p_BottomRight=%s"%(new_p_BottomRight)




            ###NOT USED .................

            numpyImage = np.asarray(targetImageTocheckAgainst[:,:])
            color=(0,0,255)
            showRectOnImage(numpyImage, new_p_TopLeft.x, new_p_BottomRight.x , new_p_TopLeft.y, new_p_BottomRight.y, color )
            #cv.WaitKey()

            #Only update if texture window is a better fit, for this point new too compare to the previous saved one
            UpdateToNpxNpy=False

            #check topLeft points if -1,-1 , which indicates that the self.prevBestHistShapePointsList is not filled with data yet
            #then we don't perform any histogram comparison whatsovever
            ###NOT USED .................
            if (self.prevBestHistShapePointsList[p_num][1]).__eq__(Point(-1,-1)):
		print 'self.prevBestHistShapePointsList[%d][%d] = %s'%(p_num,1,self.prevBestHistShapePointsList[p_num][1])
		print 'list is not filled with data yet'
		#cv.WaitKey()
            else:
                #if self.prevBestHistShapePointsList not empty then use the previously saved best window coordinates of the points saved in the prevous shape iteration,
                #to compare against the new ones

                prev_p_TopLeft = self.prevBestHistShapePointsList[p_num][1]#save topleft
                prev_p_BottomRight = self.prevBestHistShapePointsList[p_num][2]#save bottomright

		#print 'prev_p_BottomRight=%s'%(prev_p_BottomRight)
            ###NOT USED .................

                UpdateToNpxNpy = True#hist2(self.image, new_p_TopLeft.x,new_p_BottomRight.x,new_p_TopLeft.y,new_p_BottomRight.y, prev_p_TopLeft.x,prev_p_BottomRight.x,prev_p_TopLeft.y,prev_p_BottomRight.y)
		print "UpdateToNpxNpy=%s"%(UpdateToNpxNpy)
                if UpdateToNpxNpy == True:
                    #cv.WaitKey()
		    print "UpdateToNpxNpy=%s"%(UpdateToNpxNpy)


                '''
                #assign new possible update point ROI window
                ROI_1_x1=new_p_TopLeft.x
                ROI_1_x2=new_p_BottomRight.x
                ROI_1_y1=new_p_TopLeft.y
                ROI_1_y2=y2=new_p_BottomRight.y

                #assign previous-saved point ROI window, we were currently at
                ROI_2_x1=prev_p_TopLeft.x
                ROI_2_x2=prev_p_BottomRight.x
                ROI_2_y1=prev_p_TopLeft.y
                ROI_2_y2=prev_p_BottomRight.y


                #draw newer green window along with CHOSEN BEST WHISKER
                topLeftCorner=( int(ROI_1_x1), int(ROI_1_y1))
                bottomRightCorner=( int(ROI_1_x2), int(ROI_1_y2))
                cv.Rectangle(CHOSEN_BEST, topLeftCorner , bottomRightCorner, (0,255,0),1)


                #draw older bluen window along with CHOSEN BEST WHISKER
                topLeftCorner=( int(ROI_2_x1), int(ROI_2_y1))
                bottomRightCorner=( int(ROI_2_x2), int(ROI_2_y2))
                cv.Rectangle(CHOSEN_BEST, topLeftCorner , bottomRightCorner,(255,0,0),1)
                '''

                #cv.NamedWindow("CHOSEN_BEST", cv.CV_WINDOW_NORMAL)
		#cv.ShowImage("CHOSEN_BEST",CHOSEN_BEST)

                cv2.namedWindow("CHOSEN_BEST", cv.CV_WINDOW_NORMAL)
		cv2.imshow('CHOSEN_BEST',CHOSEN_BEST)







            #Save previous updated point
            #previous saved "was updated to" point, (in previous iteration)

            ###NOT USED .................

            TopLeftPoint=new_p_TopLeft
            BottomRightPoint=new_p_BottomRight
            prevLandmarkElement=(p_num, TopLeftPoint, BottomRightPoint)
            self.prevBestHistShapePointsList[p_num]=(prevLandmarkElement)

	    print "self.prevBestHistShapePointsList is"
            for i in self.prevBestHistShapePointsList:
		print i
            #cv.WaitKey()

            #showRectOnImage

            ###NOT USED .................


            ### if newerGreenWindow is also more similar to the targeted RED vertebrae
            ### area than the previously tested point-area(blue)-->Update to this newr point, Otherwise DON'T!

            if UpdateToNpxNpy==True:
                cv2.circle(CHOSEN_BEST, ( npx, npy ), 1, (0,255,255))

		print "2) targetImageTocheckAgainst[%d , %d] = %d"%(npy, npx, targetImageTocheckAgainst[npy, npx])
		print "max_edge = %d"%(max_edge)
                #cv.NamedWindow("CHOSEN_BEST", cv.CV_WINDOW_NORMAL)
		#cv.ShowImage("CHOSEN_BEST",CHOSEN_BEST)
                cv2.namedWindow("CHOSEN_BEST", cv.CV_WINDOW_NORMAL)
		cv2.imshow('CHOSEN_BEST',CHOSEN_BEST)
                #cv.WaitKey()
                if WHISKER_CHOICE==1:
                    cv.WaitKey()


                max_edge = targetImageTocheckAgainst[npy, npx] #greyImage[scale][y, x]
                max_pt = Point(npx, npy) #x,y   #new_p.x + t*norm[0], new_p.y + t*norm[1]

		print "POSSIBLE EDGE FOUND"
                c=cv.WaitKey(1)



                #cv.Circle(CHOSEN_BEST, ( int(mypoint.x), int(mypoint.y) ), 1, (0,255,0))
                #cv.Circle(CHOSEN_BEST, ( npx, npy ), 1, (0,0,255))


		print "updated max_edge & pax_pt based on new_pt (GREEN POINT)"
                #cv.WaitKey()
          else:
            #cv.Circle(CHOSEN_BEST, ( npx, npy ), 1, (255,0,0))
	    print "1) targetImageTocheckAgainst[%d , %d] = %d"%(npy, npx,targetImageTocheckAgainst[npy, npx])
	    print "max_edge = %d"%(max_edge)
            if WHISKER_CHOICE==1:
                cv.WaitKey()

	    print "have NOT...... updated max_edge & pax_pt based on new_pt (GREEN POINT)"
            #cv.WaitKey()


          '''
          cv.NamedWindow("CHOSEN_BEST", cv.CV_WINDOW_NORMAL)
	  cv.ShowImage("CHOSEN_BEST",CHOSEN_BEST)
          '''
          cv2.namedWindow("CHOSEN_BEST", cv.CV_WINDOW_NORMAL)
	  cv2.imshow('CHOSEN_BEST',CHOSEN_BEST)




          '''
          #max_pt = new_p

          iterationcounter=0

          #newPointedgeCheckedImg = cv.CreateImage(cv.GetSize(self.image), self.g_image[scale].depth, 1)
          #cv.Copy(self.g_image[scale], newPointedgeCheckedImg)

          newPointedgeCheckedImg = cv.CreateImage(cv.GetSize(self.image), self.image.depth, 3)
          cv.Copy(self.image, newPointedgeCheckedImg)



          # Look 6 pixels to each side too
          for t in drange(-3, 4, 1):#tangent not really (as at 1st currently, it chooses the directions side along normal and then from there onwards it searches around for nearby strong intensity edges to fit to)
	      #####print"t...........",t

              x = int(norm[0]*t + new_p.x)
              y = int(norm[1]*t + new_p.y)


              #Highlight with Green color the pixel along the whisker which matches the bestIndexProfile
              if t+eithersideOfNormalPixels==bestIndexProfile:
                  cv.Circle(newPointedgeCheckedImg, ( int(x), int(y) ), 1, (0,255,0))
              else:
                  cv.Circle(newPointedgeCheckedImg, ( int(x), int(y) ), 1, (0,0,255))


              cv.NamedWindow("newPoint_NormalLine_CheckedAgainstTargetImage", cv.CV_WINDOW_NORMAL)
	      cv.ShowImage("newPoint_NormalLine_CheckedAgainstTargetImage",newPointedgeCheckedImg)
              #c=##cv.WaitKey()
              #if c==1048603 :#whichever integer key code makes the app exit
              #    exit()


              if x < 0 or x > self.image.width or y < 0 or y > self.image.height:
                continue

              #show min and max points
              #cv.Circle(imgtmp, ( int(norm[0]*min_t + new_p.x) , int(norm[1]*min_t + new_p.y)), 10, (100,100,100))
              #cv.Circle(imgtmp, ( int(norm[0]*max_t + new_p.x) , int(norm[1]*max_t + new_p.y)), 10, (100,100,100))

	      #####printx, y, self.greyImage.width, self.greyImage.height

	      #####print"greyImage[currentSearchPyramidLevel][y, x]",self.greyImage[currentSearchPyramidLevel][y,x]

              targetImageTocheckAgainst=self.greyTargetImage[currentSearchPyramidLevel]#g_image[currentSearchPyramidLevel] #self.image
              if targetImageTocheckAgainst[y-1, x-1] > max_edge:
                max_edge = targetImageTocheckAgainst[y-1, x-1] #greyImage[scale][y, x]
                max_pt = Point(new_p.x + t*norm[0], new_p.y + t*norm[1]) #x,y   #
		####print'update max point'


		####print"p=%s"%(p)
		####print"max_pt=%s"%(max_pt)
		####print"x=%s , y=%s"%(x,y)
                iterationcounter+=1


                #show max point updated position
                cv.Circle(imgtmp, ( int(max_pt.x), int(max_pt.y) ), 2, (255,255,255))
                cv.NamedWindow("imgtmpMaxPoint", cv.CV_WINDOW_NORMAL)
		cv.ShowImage("imgtmpMaxPoint",imgtmp)
		###print 'max point shown'
                ##cv.WaitKey()

	      ####print"iterationcounter=%d"%(iterationcounter)
              tmpP = Point(new_p.x + t*norm[0], new_p.y + t*norm[1])#new_p.x + t*norm[0], new_p.y + t*norm[1]
              cv.Circle(imgtmp, ( int(tmpP.x), int(tmpP.y) ), 1, (100,100,100))
              cv.NamedWindow("imgtmpMaxPoint", cv.CV_WINDOW_NORMAL)
	      cv.ShowImage("imgtmpMaxPoint",imgtmp)
              c=cv.WaitKey(1)
              if c==1048603 :#whichever integer key code makes the app exit
                 exit()

          cv.Circle(imgtmp, ( int(max_pt.x), int(max_pt.y) ), 5, (100,100,100))
          cv.NamedWindow("imgtmpMaxPoint", cv.CV_WINDOW_NORMAL)
	  cv.ShowImage("imgtmpMaxPoint",imgtmp)
	  ###print 'max point for this landmark'
          ####cv.WaitKey()
          '''





          '''
          #NOT USED
          targetimage = cv.LoadImage(sys.argv[2])#"fluoSpine.jpg"
          mytargetImage=[]
          #greyImage will have the greyscale image marked with landmarks
          mytargetImage.append(self.__produce_gradient_image(targetimage, 2**0))

          myX = int(max_pt.x)
          myY = int(max_pt.y)


          mahalfile = open('mahalanobisChoiceCheeck/normDerIntProf.txt'+str(p_num),'w')



          #mahalfile.write("Sum of Subprofiles of all %d profile Elements of this landmark=%s"%(len(SumSubprofElemVecX), SumSubprofElemVecX)+"\n\n")

          #mahalfile.write("Sum of Subprofiles of all %d profile Elements of this landmark=%s"%(len(SumSubprofElemVecY), SumSubprofElemVecY)+"\n\n")


          for t in drange(-3, 3, 1):#for the range it check against along the whisker
              tmpPoint = Point(p.x + t*-norm[1], p.y + t*norm[0])
              myX = int(tmpPoint.x)
              myY = int(tmpPoint.y)
              gradientIntensity=mytargetImage[0][myY, myX]
	      ##print"gradientIntensity=%d"%(gra #MATRIX CHANGED
dientIntensity)

              mahalfile.write("gradientIntensity=%d"%(gradientIntensity)+"\n")

              ####cv.WaitKey()

          tmpPointChosen = Point(p.x + side*-norm[1], p.y + side*norm[0])#chosen one
          myX = int(tmpPointChosen.x)
          myY = int(tmpPointChosen.y)
          gradientIntensityChosen=mytargetImage[0][myY, myX]
          mahalfile.write("CHOSEN pixel with gradientIntensity=%d"%(gradientIntensityChosen)+"\n")

          mahalfile.close()
          iterationcounter=0
          #NOT USED
          '''





          '''
          ############################################################################################
          ###########OLD WAY OF CHOOSING BASED ON THE ASSUMPTION THE WE ARE ON THE MAX EDGE###########
          #along the whisker
          for side in range(-3,4):#along normal profile

              # Normal to normal...
	      #####print"norm",norm
              new_p = Point(p.x + side*norm[0], p.y + side*norm[1])#why ..the other way around?
	      #####print"p",(p.x,p.y)
	      ####print"new_p",(new_p)
              ######cv.WaitKey()

              iterationcounter=0

              imgtmp = cv.CreateImage(cv.GetSize(self.image), self.g_image[scale].depth, 1)
              cv.Copy(self.g_image[scale], imgtmp)

              # Look 6 pixels to each side too
              for t in drange(-6, 7, 1):#tangent
		  #####print"t...........",t

                  x = int(norm[0]*t + new_p.x)
                  y = int(norm[1]*t + new_p.y)


                  if x < 0 or x > self.image.width or y < 0 or y > self.image.height:
                    continue

                  #show min and max points
                  #cv.Circle(imgtmp, ( int(norm[0]*min_t + new_p.x) , int(norm[1]*min_t + new_p.y)), 10, (100,100,100))
                  #cv.Circle(imgtmp, ( int(norm[0]*max_t + new_p.x) , int(norm[1]*max_t + new_p.y)), 10, (100,100,100))

		  #####printx, y, self.greyImage.width, self.greyImage.height

		  #####print"greyImage[scale][y, x]",self.greyImage[scale][y,x]

                  targetImageTocheckAgainst=self.g_image[scale] #self.image
                  if targetImageTocheckAgainst[y-1, x-1] > max_edge:
                    max_edge = targetImageTocheckAgainst[y-1, x-1] #greyImage[scale][y, x]
                    max_pt = Point(new_p.x + t*norm[0], new_p.y + t*norm[1])
		    ####print'update max point'

		    ####print"p=%s"%(p)
		    ####print"max_pt=%s"%(max_pt)
		    ####print"x=%s , y=%s"%(x,y)
                    iterationcounter+=1


                    #show max point updated position
                    cv.Circle(imgtmp, ( int(max_pt.x), int(max_pt.y) ), 5, (100,100,100))
                    cv.NamedWindow("imgtmpMaxPoint", cv.CV_WINDOW_NORMAL)
		    #cv.ShowImage("imgtmpMaxPoint",imgtmp)
		    ###print 'max point shown'
                    #####cv.WaitKey()

		  ####print"iterationcounter=%d"%(iterationcounter)
                  tmpP = Point(new_p.x + t*norm[0], new_p.y + t*norm[1])
                  #cv.Circle(imgtmp, ( int(tmpP.x), int(tmpP.y) ), 3, (100,100,100))
                  #cv.NamedWindow("imgtmpMaxPoint", cv.CV_WINDOW_NORMAL)
		  #cv.ShowImage("imgtmpMaxPoint",imgtmp)
                  #c=cv.WaitKey(1)
                  #if c==1048603 :#whichever integer key code makes the app exit
                  #    exit()

              #cv.Circle(imgtmp, ( int(max_pt.x), int(max_pt.y) ), 5, (100,100,100))
              #cv.NamedWindow("imgtmpMaxPoint", cv.CV_WINDOW_NORMAL)
	      #cv.ShowImage("imgtmpMaxPoint",imgtmp)
	      ###print 'max point for this landmark'
              #cv.WaitKey(1)
              ###########OLD WAY OF CHOOSING BASED ON THE ASSUMPTION THE WE ARE ON THE MAX EDGE###########
              ############################################################################################
          '''





          if len(correctionVectorForAllShapes)<len(currentShape.pts):#write the first 30 landmarks
              correctionVectorForAllShapes.append(max_pt)
	      #print "added %d Shapes to correctionvector for this level"%(len(correctionVectorForAllShapes))

              #move to the next pyramid level, in case we reached the number of landmarks needed to be within 50% of profile length / OR number of pre-specified iterations on each level is reached
              if breaktonextpyramidlevel==True:
		  #print "break to next pyramidlevel: [%d]"%(self.currentSearchPyramidLevel)
                  f.close()
                  return correctionVectorForAllShapes
                  ###cv.WaitKey()
                  break

          else:#overwrite all the others and at the end divide by the total number of shapes
              correctionVectorForAllShapes[p_num]=max_pt
	      #print "added %d Shapes to correctionvector for this level"%(len(correctionVectorForAllShapes))

              #move to the next pyramid level, in case we reached the number of landmarks needed to be within 50% of profile length / OR number of pre-specified iterations on each level is reached
              if breaktonextpyramidlevel==True:
		  #print "break to next pyramidlevel: [%d]"%(self.currentSearchPyramidLevel)
                  f.close()
                  return correctionVectorForAllShapes
                  ###cv.WaitKey()
                  break




      f.close()
      ##print "len of correctionVectorForAllShapes=%d"%(len(correctionVectorForAllShapes))
      ####cv.WaitKey()


      #Compare the current and previous iteration, BestIndices lists
      #If different by 10% or less of the total element number of the list, which mean
      #(check if each individual list element is different from the previous iteration),
      #then stop, otherwise save the current as the previous iteration BestIndices list
      #and continue to the next search/fitting iteration
      totallandmarksAdjustedCounter=0

      print "zip(self.savedBestIndicesCurrentIteration, self.savedBestIndicesPreviousIteration)=%s"%(zip(self.savedBestIndicesCurrentIteration, self.savedBestIndicesPreviousIteration))
      #cv.WaitKey()

      for i, j in zip(self.savedBestIndicesCurrentIteration, self.savedBestIndicesPreviousIteration):

          #if element X is different the +1 to the totallandmarksAdjustedCounter
          if i!=j:
              totallandmarksAdjustedCounter+=1
	      print "i=%s"%(i)
              #cv.WaitKey()
	      print "j=%s"%(j)
              #cv.WaitKey()
	      print "i different then j"
              #cv.WaitKey()
	      print "totallandmarksAdjustedCounter=%d"%(totallandmarksAdjustedCounter)
              #cv.WaitKey()

      print "totallandmarksAdjustedCounter=%d <= float(1)/10 * self.totallandmarksAdjustedCounterPrev = %f"%(totallandmarksAdjustedCounter, float(1)/10 * self.totallandmarksAdjustedCounterPrev)
      #cv.WaitKey()

      print "self.totallandmarksAdjustedCounterPrev=%d"%(self.totallandmarksAdjustedCounterPrev)
      print "totallandmarksAdjustedCounter=%d"%(totallandmarksAdjustedCounter)
      #cv.WaitKey()




      #>5 as we want it to force it to run for a few iterations (at least 5), as it has been observed that iteration 1 is not updating (possibly cause mahalanobis spatial sitance is same as iteration 0)
      if iterationIndex>5 and (totallandmarksAdjustedCounter <= float(1)/10 * self.totallandmarksAdjustedCounterPrev):
          self.STOP_SEARCH_FITTING=True #stop fitting..
	  print "stop fitting"
          cv.WaitKey()



      #save deep copy savedBestIndicesPreviousIteration
      self.savedBestIndicesPreviousIteration = copy.deepcopy(self.savedBestIndicesCurrentIteration)
      print "savedBestIndicesPreviousIteration=%s"%(self.savedBestIndicesPreviousIteration)
      #cv.WaitKey()

      self.totallandmarksAdjustedCounterPrev=totallandmarksAdjustedCounter



      return correctionVectorForAllShapes


  '''
  ###print "len(correctionVectorForAllShapes)=%d"%len(correctionVectorForAllShapes)
  ###print "(correctionVectorForAllShapes)=%s"%(correctionVectorForAllShapes)
  ######cv.WaitKey()

for i in range(len(correctionVectorForAllShapes)):
  ###print "(correctionVectorForAllShapes[%d]) was %s"%(i,correctionVectorForAllShapes[i])
  correctionVectorForAllShapes[i]=correctionVectorForAllShapes[i]/len(AllImages_NormalizedtmpLandmarkDerivativeIntensityVector)
  ###print "(correctionVectorForAllShapes[%d]) AVERAGED is %s"%(i,correctionVectorForAllShapes[i])

###print "len(correctionVectorForAllShapes)=%d"%len(correctionVectorForAllShapes)
return correctionVectorForAllShapes
  ##########################################################################################
  ##########################################################################################
  '''



  def do_iteration(self, scale, iterationIndex, GTrainingMeansCalculated):
    """ Does a single iteration of the shape fitting algorithm.
    This is useful when we want to show the algorithm converging on
    an image

    :return shape: The shape in its current orientation
    """

    '''...a shape in the training set can be approximated using the mean shape
        and a weighted sum of the deviations obtained by x=(mean) + P*b...'''
    '''
    __get_max_along_normal, could practically start from the mean shape as it already does,
    then, for each of the image shapes (3 for now) we create an s Shape from the which would derive the update of self.shape through aligning 'new' shape to 's' with weighting
    '''

    ###print 'Iteration started'


    #for each of the points of the current shape
    #for i, pt in enumerate(self.shape.pts):

    #pass in the mean shape initially in the first iteration
    #self.shapes gets updated at every iteration
    #self.calcMaxPtFromMahalanobisModelProfile(self.shape, self.currentSearchPyramidLevel, iterationIndex, GTrainingMeansCalculated)

    if not self.STOP_SEARCH_FITTING:
        self.calcMaxPtFromMahalanobisModelProfile(self.shape, self.currentSearchPyramidLevel, iterationIndex, GTrainingMeansCalculated)




  def calcMaxPtFromMahalanobisModelProfile(self, initmeanshape, scale, iterationIndex ,GTrainingMeansCalculated):

    #create a new image based on the input image's 1st resolution (without any scaling whatsoever)
    img = cv.CreateImage(cv.GetSize(self.image), self.image.depth, 3)
    cv.Copy(self.image, img)



    #create a shape to be filled with the new correction vector (after mahalanobis measurement for this image)
    s = Shape([])

    #calculate G mean for all landmakrs just once
    if GTrainingMeansCalculated == 0:

        #now let's empty the self.g_image and start over and fill it with all images for each level as we don't need it
        self.g_image=[]

        #prevent from re-calculating Training G means (cause they should probably be based on the first mean shape)
        GTrainingMeansCalculated = 1

        #Transposed matrix to align with the iteration from bottom(tangent) left(whisker)..to..up(tangent right(whisker))
        self.gaussianMatrix=np.transpose(self.gaussianMatrix)

	####print"gaussianMatrix=%s"%(gaussianMatrix)

        #iterate the gaussian distribution matrix
        for (x,y), value in np.ndenumerate(self.gaussianMatrix):
		    print"%d, %d =%s"%(x,y,self.gaussianMatrix[x][y])
                    #######cv.WaitKey()

        #self.AllImages_NormalizedtmpLandmarkDerivativeIntensityVector is returnt to self.trainingGmean and THIS represents the trainingGmean for each level
        # g mean of all landmarks for each level-> if 20 landmarks & 3 levels..then trainingGmean containings 3 level vectors, each one of which contains 20 gmeans (1 for each landmark)
        self.trainingGmean = self.findtrainingGmean(self.testImageNameCounter, self.AllImages_NormalizedtmpLandmarkDerivativeIntensityVector, self.gaussianMatrix)
	##print "len(trainingGmean)=%s"%(len(self.trainingGmean))
        ####cv.WaitKey()



    #calculate mahalanobis distance, to derive the correction vector
    max_ptlist=self.getCorrectedLandmarkPoint(self.trainingGmean, iterationIndex, self.currentSearchPyramidLevel)

    #print 'length of max_ptlist=%d'%(len(max_ptlist))
    ###cv.WaitKey()

    #print 'max_ptlist=%s'%(max_ptlist[0])
    ###cv.WaitKey()



    f = open('maxptList/max_ptlist'+str(iterationIndex),'w')
    f.write(str(max_ptlist[0]))
    f.close()


    prevPoint=-1
    nextPoint=-1
    #for each of the points of the current shape
    for i, pt in enumerate(initmeanshape.pts):


	####print "maxpoint=%s"%(max_pt)
        #fill a tmp shape with the points returnt from the image search/fitting process in (getCorrectedLandmarkPoint function)
        s.add_point(max_ptlist[i])

        #show max points chosen along normal profile sampled -6..-6 pixels across
        #BEFORE ALIGNMENT
	####print "\BEFORE ALIGNMENT\n"
        maxpX=(int)(pt.x)
        maxpY=(int)(pt.y)
        #cv.Circle(img, (maxpX,maxpY), 3, (40,0,255))

        nextPoint = (maxpX, maxpY)
        if prevPoint != -1:
            cv.Line(img, prevPoint , nextPoint ,(0,0,255),1)
        prevPoint = nextPoint


        cv.NamedWindow("Scale", cv.CV_WINDOW_NORMAL)
	cv.ShowImage("Scale",img)


    #####cv.WaitKey()
    ####print 'Points Added'

    #align this s shape to mean shape with a weigted matrix
    new_s = s.align_to_shape(Shape.from_vector(self.asm.mean), self.asm.w)

    ####print (self.asm.evals[0])

    #calculate new shape - update the mean based on x=(mean) + P*b
    #the b parameters is to ensure we fit the current model instance to the image points correctly with the appropriate constraints applied (basic differece from snakes method, that's why they are called "smart snakes")

    #(Y - Mean)
    var = new_s.get_vector() - self.asm.mean
    #(start always from the Mean shape)..and adjust based on x=(mean) + P*b
    new = self.asm.mean
    for i in range(len(self.asm.evecs.T)):
      #b = Ptransposed (Y-Mean) - b is a k dimensional vector (k being the number of modes)
      #which expresses only a small set of parameters to be adjusted, instead of the 2n vector of points so as to fit the model to the target image
      b = np.dot(self.asm.evecs[:,i],var)

      #print "self.asm.evecs=%s"%(self.asm.evecs)
      #print "self.asm.evecs[:,%d]=%s"%(i,self.asm.evecs[:,i])

      #ADDITION of this IF statement CAUSE IT CRASHS IF  self.asm.evals[i] is < 0
      if self.asm.evals[i] > 0:
      #if True:
          #ensure doesn't vary more than 3 standard deviations

          #from 4.1 Constraint on b (from the 'Subspace Methods for Pattern Recognition in Intelligent Environment' book)

          '''In an image pyramid, because of
          subsampling, each dimension of the image on the Lth level is 1 2 L of the original
          one. x and y coordinate of landmarks are also 1 2 L . Therefore, all elements of the
          covariance matrix become 1 4 L comparing to those in the original image. When
          applying PCA now, we obtain the eigen value with 1 4 L times of that on the level
          0. If we retain the constraint as +- li , that leads to an excessive broad limit.
          Therefore, a distort shape will be generated finally.
          To solve the problem, we must modify the range limit on b. In the lower level,
          the restriction should be relaxed, whereas it should be narrower in the higher level.
          Shown mathematically, in the level L, the constraint should be -3li/4** L<= bi <= 3 li/4**L
          where li is the eigenvalue of the covariance matrix of the ith principal
          component. According to the actual conditions, constraint could be stricter in order
          to get better search result over the higher level.
          Images in the high level are blurred due to su
          '''
          max_b = 3*math.sqrt(self.asm.evals[i]/ (4**self.currentSearchPyramidLevel)) #was.. 3*math.sqrt(self.asm.evals[i])

          b = max(min(b, max_b), -max_b)

          #x=(mean) + P*b
          new = new + self.asm.evecs[:,i]*b

    ###print "previous shape=%s"%(self.shape.pts[0])
    #####cv.WaitKey()

    #align the "new" shape that has already the contraint of b applied to it, to the already existing (aligned to the mean) s
    self.shape = Shape.from_vector(new).align_to_shape(s, self.asm.w)

    ###print "current shape=%s"%(self.shape.pts[0])
    #####cv.WaitKey()

    '''
    #show max points chosen along normal profile sampled -6..-6 pixels across
    #AFTER ALIGNMENT
    ####print "\nAFTER ALIGNMENT\n"
    for i, pt in enumerate(self.shape.pts):
        maxpX=(int)(max_pt.x)
        maxpY=(int)(max_pt.y)
        #cv.Circle(img, (maxpX,maxpY), 2, (255,255,255))
        cv.NamedWindow("Scale", cv.CV_WINDOW_NORMAL)
	cv.ShowImage("Scale",img)
        ######cv.WaitKey()
    '''


    ######cv.WaitKey()



  def __get_max_along_normal(self, p_num, scale):
    print 'add code'

class ActiveShapeModel:
  """
  """
  def __init__(self, shapes = []):
    self.shapes = shapes
    # Make sure the shape list is valid
    self.__check_shapes(shapes)
    # Create weight matrix for points
    ####print "Calculating weight matrix..."
    self.w = self.__create_weight_matrix(shapes)

    ####print 'Shapes BEFORE Weighted Procrustes'
    ####print self.shapes[0].pts
    ####print "\n"
    ####print self.shapes[1].pts
    ####print "\n"
    ####print self.shapes[2].pts
    ####print "\n"

    # Align all shapes
    ####print "Aligning shapes with Procrustes analysis..."
    self.shapes = self.__procrustes(shapes)

    ####print 'Shapes AFTER Weighted Procrustes'
    ####print self.shapes[0].pts
    ####print "\n"
    ####print self.shapes[1].pts
    ####print "\n"
    ####print self.shapes[2].pts
    ####print "\n"





    ####print "Constructing model..."
    # Initialise this in constructor
    (self.evals, self.evecs, self.mean, self.modes) = \
        self.__construct_model(self.shapes)






  def __check_shapes(self, shapes):
    """ Method to check that all shapes have the correct number of
    points """
    if shapes:
      num_pts = shapes[0].num_pts
      for shape in shapes:
        if shape.num_pts != num_pts:
          raise Exception("Shape has incorrect number of points")

  def __get_mean_shape(self, shapes):
    s = shapes[0]
    for shape in shapes[1:]:
      s = s + shape
    return s / len(shapes)

  def __construct_model(self, shapes):
    """ Constructs the shape model
    """

    #get coordinates of shapes (for 3 shapes with 61 points each = 3x61x2=183 values) or
    shape_vectors = np.array([s.get_vector() for s in self.shapes])

    '''
    #LOG OUT LANDMARK POINT VECTORS OF EACH INDIVIDUAL IMAGE IN THE TRAINING SET
    #Gives the landmark points of 1 image out of the training set
    ####print ("3 training images tested = %s")%(len(shape_vectors))
    ####print ("3 training images tested = %s")%((shape_vectors))


    ####print ("each image contains %s landmark points")%(len(s.get_vector())/2)#61 sets of coordinates [x,y]
    target = open("logsOutput/trainingImagesVectorPoints", 'w')


    #for all 3 images' vectors in the training set
    for s in self.shapes:
                oneTrainingImageVector=Shape.from_vector(s.get_vector())
                for p in oneTrainingImageVector.pts:
                  x=(str(p.x))
                  y=(str(p.y))
                  target.write(x)
                  target.write(",")
                  target.write(y)
                  target.write("\n")
                target.write("\nEND IMAGE POINTS\n\n")
    target.close()
    #Gives the landmark points of 1 image out of the training set
    '''


    ##print "shape_vectors\n",shape_vectors
    c=cv.WaitKey(1)
    #####print"EXIT when ESC keycode is pressed=%d"%(c)
    if c == 1048603:
        exit()

    #the mean of the aligned shapes
    mean = np.mean(shape_vectors, axis=0)

    ####print "mean shape which is the center of the ellipsoidal Allowable Shape Domain - Before reshaping\n",mean
    ####print "\n"

    # Move mean to the origin
    # FIXME Clean this up...
    mean = np.reshape(mean, (-1,2))#turn mean array from 4x5 to 10x2 array
    min_x = min(mean[:,0])#get the min of the 1st row refering to the first X coordinate
    min_y = min(mean[:,1])#get the min of the 2nd row refering to the Second Y coordinate

    ##print "mean After reshaping\n",mean
    c=cv.WaitKey(1)
    #####print"EXIT when ESC keycode is pressed=%d"%(c)
    if c == 1048603:
        exit()


    #mean = np.array([pt - min(mean[:,i]) for i in [0,1] for pt in mean[:,i]])
    #mean = np.array([pt - min(mean[:,i]) for pt in mean for i in [0,1]])


    #http://sebastianraschka.com/Articles/2014_pca_step_by_step.html
    #Implementing a Principal Component Analysis (PCA)
    #in Python, step by step

    '''****Principal Componet Analysis****'''

    '''For Each shape in the training set
       we calculate the deviation from the mean
       for each of the dimensions PCA
    '''

    mean[:,0] = [x  for x in mean[:,0]]#- min_x
    mean[:,1] = [y  for y in mean[:,1]]#- min_y
    #mean[:,0] = [x - min_x for x in mean[:,0]]
    #mean[:,1] = [y - min_y for y in mean[:,1]]
    #max_x = max(mean[:,0])
    #max_y = max(mean[:,1])
    #mean[:,0] = [x/(2) for x in mean[:,0]]
    #mean[:,1] = [y/(3) for y in mean[:,1]]
    mean = mean.flatten()
    #####print mean


    ####print "The list of shapes (and their point coordinates:\n",shape_vectors
    ####print "\n\n"



    # Produce covariance matrix

    '''We attempt to model the shape in the Allowable Shape Domain..hence capture the relationships between position of individual landmark points'''
    '''The cloud of landmarks is approximately ellipsoidal, so we need to calculate its center (giving a mean shape and its major axes)'''
    '''Covariance indicates the level to which two variables vary together'''

    #shape_vectors=[(1,2),(3,4)]
    cov = np.cov(shape_vectors, rowvar=0)

    print "shape_vectors.shape[1:2]=%s"%(shape_vectors.shape[1:2])
    cv.WaitKey()

    ###print "cov\n",cov
    ####cv.WaitKey()

    # Find eigenvalues/vectors of the covariance matrix
    evals, evecs = np.linalg.eig(cov)

    ####print "evals\n",evals
    ####print "sum(evals)\n",sum(evals)


    ####print "evecs\n",evecs



    #=0
    # Find number of modes required to describe the shape accurately
    t = 0
    for i in range(len(evals)):

	  ####print "sum(evals[:%f])=%f\n"%(sum(evals[:i]),sum(evals[:i]))
	  #####print "sum(evals[:%f])=%f\n"%(sum(evals),sum(evals))

          #iterating through the list of evals, as soon as the sum  of evals so far divided by the total sum is >=0.99 then this is the number of modes we need
          '''Choose the first largest eigenvalues..that represent a wanted percentage of the total variance, a.i. 0.99 or 99%.
          Defines the proportion of the total variation one wishes to explain
          (for instance, 0.98 for 98%)'''

          if sum(evals[:i]) / sum(evals)< 0.99:
                #=c+1
		#####print c
                t = t + 1
          else: break


    #print "Constructed model with %d modes of variation" % t
    ###cv.WaitKey()

    #evecs = np.array([[1,2,3,4,5,6],[7,8,9,10,11,12]])
    #####print evecs[:,:1]

    ####print "evals[:%d]\n%s"%(t,evals[:t])

    #return the eigenvalues & eigenvector analogous to the number of modes required
    return (evals[:t], evecs[:,:t], mean, t)



  def generate_example(self, b):
    """ b is a vector of floats to apply to each mode of variation
    """
    # Need to make an array same length as mean to apply to eigen
    # vectors
    full_b = np.zeros(len(self.mean))
    for i in range(self.modes): full_b[i] = b[i]

    p = self.mean
    for i in range(self.modes): p = p + full_b[i]*self.evecs[:,i]

    # Construct a shape object
    return Shape.from_vector(p)

  def __procrustes(self, shapes):
    """ This function aligns all shapes passed as a parameter by using
    Procrustes analysis

    :param shapes: A list of Shape objects
    """
    # First rotate/scale/translate each shape to match first in set
    shapes[1:] = [s.align_to_shape(shapes[0], self.w) for s in shapes[1:]]

    # Keep hold of a shape to align to each iteration to allow convergence
    a = shapes[0]#1st shape
    trans = np.zeros((4, len(shapes)))#list to save all Alignment parameters after getting each transformation required for each shape

    #####print "TRANS ", trans

    converged = False
    current_accuracy = sys.maxint

    ml=[]
    lines_counter_written=1#contol flag
    lines_counter=0

    while not converged:
      # Now get mean shape
      mean = self.__get_mean_shape(shapes)

      #####print "Calculated mean:",mean.pts

      # Align to shape to stop it diverging
      mean = mean.align_to_shape(a, self.w)#NORMALIZATION

      #####print "new mean=",mean.pts

      # Now align all shapes to the mean
      for i in range(len(shapes)):
        # Get transformation required for each shape
        trans[:, i] = shapes[i].get_alignment_params(mean, self.w)
        # Apply the transformation
        shapes[i] = shapes[i].apply_params_to_shape(trans[:,i])

      # Test if the average transformation required is very close to the
      # identity transformation and stop iteration if it is
      accuracy = np.mean(np.array([1, 0, 0, 0]) - np.mean(trans, axis=1))**2



      '''
      #create a list with pairs of coordinates that represent the mean at each iteration until convergence of aligning is reached
      for i,p in enumerate(mean.pts):
                  ml.append(p.x)
                  ml.append(p.y)

                  lines_counter=int(lines_counter)
                  lines_counter+=1

          #Write a pts file with all the mean shapes calculated
      #num_mpts=str(mean.num_pts)

      #Write to pts file
      filename="MeanShapeMotion.pts"
      target = open(filename, 'w')

      #WEIRDDDDDDDDD to remove
      if (1 == 1):
                  lines_counter=str(lines_counter)
                  #target.write(lines_counter)
                  #target.write("\n")


      for i in range(len(ml)):
                  global target

                  x=str(ml[i])
                  y=str(ml[1+1])

                  target.write(x)
                  target.write(' , ')
                  target.write(y)
                  target.write(' , ')
                  #target.write("\n")
         '''

         # If the accuracy starts to decrease then we have reached limit of precision
      # possible
      if accuracy > current_accuracy: converged = True;  #####print 'accuracy=',(accuracy);     ####print 'current_accuracy=',(current_accuracy);
      else: current_accuracy = accuracy; # ####print 'accuracy=',(accuracy);     ####print 'current_accuracy=',(current_accuracy);

    #target.close()
    ####print "Final Mean Shape Points=",mean.pts
    ####print "\n"

    #####print "ACCURACY=%d, current_accuracy=%d ,  converged=%s"%(accuracy,current_accuracy, converged)
    return shapes

  def __create_weight_matrix(self, shapes):
    """ Private method to produce the weight matrix which corresponds
    to the training shapes

    :param shapes: A list of Shape objects
    :return w: The matrix of weights produced from the shapes
    """
    # Return empty matrix if no shapes
    if not shapes:
      return np.array()
    # First get number of points of each shape
    num_pts = shapes[0].num_pts

    # We need to find the distance of each point to each
    # other point in each shape.
    distances = np.zeros((len(shapes), num_pts, num_pts))
    for s, shape in enumerate(shapes):
      for k in range(num_pts):
        for l in range(num_pts):
          distances[s, k, l] = shape.pts[k].dist(shape.pts[l])

    # Create empty weight matrix
    w = np.zeros(num_pts)
    # calculate range for each point
    for k in range(num_pts):
      for l in range(num_pts):
        # Get the variance in distance of that point to other points
        # for all shapes
        w[k] += np.var(distances[:, k, l])
    # Invert weights
    return 1/w

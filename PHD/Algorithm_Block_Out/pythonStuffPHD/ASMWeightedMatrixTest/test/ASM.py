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
    cv.NamedWindow("Shape Model Variations", cv.CV_WINDOW_AUTOSIZE)
    # Get size for the window
    max_x = int(max([pt.x for shape in shapes for pt in shape.pts]))
    max_y = int(max([pt.y for shape in shapes for pt in shape.pts]))
    min_x = int(min([pt.x for shape in shapes for pt in shape.pts]))
    min_y = int(min([pt.y for shape in shapes for pt in shape.pts]))

    i = cv.CreateImage((max_x-min_x+20, max_y-min_y+20), cv.IPL_DEPTH_8U, 3)



    cv.Set(i, (0, 0, 0))
    for shape in shapes:
      r = 255#randint(0, 255)
      g = 255#randint(0, 255)
      b = 0#randint(0, 255)
      #r = 0
      #g = 0
      #b = 0
      for pt_num, pt in enumerate(shape.pts):
        # Draw normals
        #norm = shape.get_normal_to_point(pt_num)
        #cv.Line(i,(pt.x-min_x,pt.y-min_y), \
        #    (norm[0]*10 + pt.x-min_x, norm[1]*10 + pt.y-min_y), (r, g, b))
        cv.Circle(i, (int(pt.x-min_x), int(pt.y-min_y)), 1, (b, g, r), -1)
      ###print "pt=%d,%d"%(pt.x,pt.y)
      cv.ShowImage("Active shape Model",i)

  @staticmethod
  def show_modes_of_variation(model, mode):
    # Get the limits of the animation

    print "model.evals[mode]",model.evals[mode]
    cv.WaitKey()

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
      b += step

      c = cv.WaitKey()
      if c == 1048603:
          #exit()
          break


  @staticmethod
  def draw_model_fitter(f):
    cv.NamedWindow("Fit Model", cv.CV_WINDOW_AUTOSIZE)

    #c = cv.WaitKey(10)
    ###print "f.shape.pts",f.shape.pts
    #c = cv.WaitKey(10)

    # Copy image
    i = cv.CreateImage(cv.GetSize(f.image), f.image.depth, 3)
    cv.Copy(f.image, i)
    for pt_num, pt in enumerate(f.shape.pts):
      # Draw normals
      cv.Circle(i, (int(pt.x), int(pt.y)), 1, (0,255,0), -1)
      ##print "pt=%d,%d"%(pt.x,pt.y)
    #Draw the original targeted image with cyan landmark points
    ###cv.WaitKey()
    cv.ShowImage("ASM",i)
    #cv.WaitKey()
    ###print 'STAGE - key pressed\n'

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
        ###print "FFFFFfirst_line=%s"%first_line
        num_pts = int(first_line)
      for line in fh:
        if not line.startswith("}"):
          pt = line.strip().split()
          ###print "line",line
          s.add_point(Point(float(pt[0]), float(pt[1])))
    if s.num_pts != num_pts:
      ###print "Unexpected number of points in file.  "\
      "Expecting %d, got %d" % (num_pts, s.num_pts)
    return s

  @staticmethod
  def read_directory(dirname):


    """ Reads an entire directory of .pts files and returns
    them as a list of shapes
    """
    fileNumber=1
    pts = []

    totalPtsFiles=0
    for file in os.listdir(dirname):
        if file.endswith(".pts"):
            totalPtsFiles=totalPtsFiles+1

    for i in range(1,totalPtsFiles,1):
        file = glob.glob(os.path.join(dirname, "*%d*.pts"%(fileNumber) ))
        print file

        #cv.WaitKey()

        f=("%d.pts")%(fileNumber)
        f=file[0]

        print f
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
  def __init__(self, asm, image, originalShapes, t=Point(5,25)):#x,y
    #the image target we are testing against
    self.image = image
    #the array of grey_scaled target image at different resolutions (if implemented)
    self.g_image = []

    scale=0

    #vector containing the covariance matrices calculated once, when forming the grey-level profile (and it is about to be used during "image search")
    self.CovarianceMatricesVec=[]
    self.tmpCovar=0

    self.COVARIANCE_CALCULATION_ONCE = False

    #this encapsulates all images' normalized derivative profiles for all their landmarks
    self.AllImages_NormalizedtmpLandmarkDerivativeIntensityVector = []

    #for all 3 images' vectors of landmark points in the training set
    self.testImageNameCounter=0

    self.trainingGmean = []


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
    for i in range(0,4):
      self.g_image.append(self.__produce_gradient_image(image, 2**i))

    print "G_IMAGE="
    width, height = cv.GetSize(self.g_image[scale])
    print width,height
    ##cv.WaitKey()

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
    degreesOfRotation=-15

    degreesOfRotation=0
    CenterOfMass=Point(0,0)
    if CenterOfMass.__eq__(Point(0,0)):
        t= Point(0,0)

    self.asm = asm
    # Copy mean shape as starting shape and transform it to where the target shape to detect is manually (later on with template matching)
    #print "asm.mean=%s"%(asm.mean)
    ##cv.WaitKey()
    self.shape = Shape.from_vector(asm.mean).transform(t, degreesOfRotation, CenterOfMass)
    #print "asm.mean transformed=%s"%(self.shape.pts)
    ##cv.WaitKey()


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
            print "%d, %s"%(i,pt)

            cv.Circle(trainingSetImage, ( int(pt.x), int(pt.y) ), 4, (100,100,100))
            cv.NamedWindow("TEST", cv.CV_WINDOW_AUTOSIZE)
            cv.ShowImage("trainingSetImage",trainingSetImage)

            #cv.WaitKey()
    '''






    '''Transform the shape to where the target vertebra is / should be done with Template Matching'''
    shapecounter=0
    #now modifly each of the asm shapes as well, based on the desired transform
    for shape in self.asm.shapes:
        #print "self.shapes[0].pts=%s"%(shape.pts[0])
        #tmpShapeToTransform=shape.pts
        #shape.pts[0]=Point(1,1)
        #shape = np.array([shape.get_vector()])

        shapeCoordinatesList=[]
        for i, pt in enumerate(shape.pts):
            shapeCoordinatesList.append([pt.x,pt.y])

        shapeCoordinatesList = np.array(shapeCoordinatesList)
        #print "shapeCoordinatesList=%s"%(shapeCoordinatesList)

        s= shapeCoordinatesList.flatten()
        #print "shapeCoordinatesList=%s"%(s)

        print "s=%s points each vector"%(len(s))
        cv.WaitKey()

        tmpShapeManuallyTransformed=Shape.from_vector(s).transform(t,CenterOfMass)
        #print "tmpShapeManuallyTransformed=%s"%(tmpShapeManuallyTransformed.pts)
        ##cv.WaitKey()


        #print "asm.shapes[%d]=%s"%(shapecounter,asm.shapes[shapecounter].pts)
        ##cv.WaitKey()

        #now transform each asm read in shape (manually) to where the target vertebra in the image target is (just to visualize it there ..it may be placed based on tracking at a later stage or by an somehow automated procedure)
        shapePoints=asm.shapes[shapecounter].pts
        transformShapePoints=tmpShapeManuallyTransformed.pts
        for i in range(len(shapePoints)):
            #print "before=%s"%(shapePoints[i])
            ##cv.WaitKey()

            shapePoints[i].x=transformShapePoints[i].x
            shapePoints[i].y=transformShapePoints[i].y
            #print "after=%s"%(shapePoints[i])
            ##cv.WaitKey()

        #print "asm.shapes[%d] transformed=%s"%(shapecounter,asm.shapes[shapecounter])
        ##cv.WaitKey()

        shapecounter+=1




    #print "self.asm.shapes[0][0]=%s"%self.asm.shapes[0][0]
    ##cv.WaitKey()

    print "scale=%d"%(scale)
    ##cv.WaitKey()

    ShapeBeforeResizeToFitImage = cv.CreateImage(cv.GetSize(self.image), self.g_image[scale].depth, 1)
    cv.Copy(self.g_image[scale], ShapeBeforeResizeToFitImage)
    for i, pt in enumerate(self.shape.pts):
        cv.Circle(ShapeBeforeResizeToFitImage, ( int(pt.x), int(pt.y) ), 2, (100,100,100))
    cv.NamedWindow("ShapeBeforeResizeToFitImage", cv.CV_WINDOW_AUTOSIZE)
    cv.ShowImage("ShapeBeforeResizeToFitImage",ShapeBeforeResizeToFitImage)
    ##cv.WaitKey()


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
    ##print "mahalanobis distance between all g profiles of this landmark and the g mean profile is :%s"%(md)
    '''

    #X = np.vstack((IthImage_NormalizedtmpLandmarkDerivativeIntensityVector[i]), np.array(y_j_mean[0])))
    #covMat=np.cov( IthImage_NormalizedtmpLandmarkDerivativeIntensityVector[i] , np.array(y_j_mean[0]) )

    ###print "covariance matrix of  =%s"%(covMat)
    ####cv.WaitKey()

    #calculate for the ith image , each jth landmark which profile g fits best (based on mahalanobis distance measurement)
    #when done with all landmarks of this ith image..then move onto the next image to fin best fits
    #covMatInverse = np.linalg.inv(covMat)
    #mahalanobisDist = scipy.spatial.distance.mahalanobis(  np.array(IthImage_NormalizedtmpLandmarkDerivativeIntensityVector[i]).flatten(), np.array(y_j_mean[0]).flatten(), covMatInverse )
    #mahalanobisDist = scipy.spatial.distance.mahalanobis(  np.array(IthImage_NormalizedtmpLandmarkDerivativeIntensityVector[i]).flatten(), np.array(y_j_mean[0]).flatten(), covMatInverse )

    ###print "mahalanobis=%s"%(mahalanobisDist)
    ###cv.WaitKey()
    '''
    for ithImageLandmarksVector in IthImage_NormalizedtmpLandmarkDerivativeIntensityVector:#3 images x 30 landmarks each x 6 normalizedDerivativeProfiles..so 450 element vector
        landmarkAtWhiskerXMeanIndex=0
        for yij_ithLandmarkVector in ithImageLandmarksVector:#1 images x 30 landmarks x 6 normalizedDerivativeProfiles..so 180 element vector
            yj_mean=githLandmarkMeanAcrossAllImagesVector[landmarkAtWhiskerXMeanIndex]
            ##print "Calculate the Covariance Matrix between\n: yij_ithLandmarkVector=%s and yj_mean[%d]=%s \n"%(np.array(yij_ithLandmarkVector),landmarkAtWhiskerXMeanIndex, yj_mean)
            covMat=np.cov(np.array(yij_ithLandmarkVector),yj_mean)
            landmarkAtWhiskerXMeanIndex+=1
            ##print "covMat=%s"%(covMat)
            ####cv.WaitKey()
            covMatDeterminant=np.linalg.det(covMat)
            ##print "covMatDeterminant=%s"%(covMatDeterminant)
            ####cv.WaitKey()

            covMatInverse = np.linalg.inv(covMat)
            #mahalanobisDist = scipy.spatial.distance.mahalanobis(  yij_ithLandmarkVector, yj_mean, covMatInverse )
            ###print "mahalanobis=%s"%(mahalanobisDist)
            ####cv.WaitKey()
    '''


    '''
    ##print "Calculate the Covariance Matrix between\n: IthImage_NormalizedtmpLandmarkDerivativeIntensityVector[0][0]=%s and githLandmarkMeanAcrossAllImagesVector[0]=%s \n"%(np.array(IthImage_NormalizedtmpLandmarkDerivativeIntensityVector[0][0]),githLandmarkMeanAcrossAllImagesVector[0])
    covMat=np.cov(np.array(IthImage_NormalizedtmpLandmarkDerivativeIntensityVector[0][0]),githLandmarkMeanAcrossAllImagesVector[0])
    ##print "covMat=%s"%(covMat)
    ###cv.WaitKey()
    covMatDeterminant=np.linalg.det(covMat)
    ##print "covMatDeterminant=%s"%(covMatDeterminant)
    ###cv.WaitKey()
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

            ##print "%f-%f .. difference=%f"%(currentLandmarkProfiles[intensity+1], currentLandmarkProfiles[intensity], difference)

            #store derivative gray-level profile vector of all whisker point (-3 points on the one side  +landmark Itseflf + 3 points on the other side
            tmpLandmarkDerivativeIntensityVector.append( difference )

            normalizedBySumofDifference=normalizedBySumofDifference+difference

        ##print "tmpLandmarkDerivativeIntensityVector: %s"%(tmpLandmarkDerivativeIntensityVector)
        ##print "normalizedBy: %s"%(normalizedBySumofDifference)
        ####cv.WaitKey()

        #(equation 13 of the 'Subspace Methods for Pattern Recognition in Intelligent Environment' book)
        #normalize tmpProfileDerivativeIntensityVector
        NormalizedtmpLandmarkDerivativeIntensityVector=[]
        for x in tmpLandmarkDerivativeIntensityVector:
            if normalizedBySumofDifference!=0:
                x = x / normalizedBySumofDifference
                NormalizedtmpLandmarkDerivativeIntensityVector.append(x)
            else:
                ##print "tmpLandmarkDerivativeIntensityVector is: %s\n"%(tmpLandmarkDerivativeIntensityVector)
                x = x / 1
                NormalizedtmpLandmarkDerivativeIntensityVector.append(x)
                ###cv.WaitKey()

        ##print "NormalizedtmpLandmarkDerivativeIntensityVector %s\n"%(NormalizedtmpLandmarkDerivativeIntensityVector) ###.. must be for this point along the whisker
        ####cv.WaitKey()

        #now store this Normalized tmp Landmark Derivative Intensity Vector for this image for THIS landmark
        # and move onto caclulating the NormalizedtmpLandmarkDerivativeIntensityVector for this image for the NEXT landmark

        #This vector contains : for each image, all normalized derivative profile of ALL ith's image's landmarks
        ##print "image=%d\n"%(testImageNameCounter-1)
        IthImage_NormalizedtmpLandmarkDerivativeIntensityVector[testImageNameCounter-1].append(NormalizedtmpLandmarkDerivativeIntensityVector)
        ##print "Here the following vector is filled  with the  NormalizedtmpLandmarkDerivativeIntensityVector \nor each of the landmarks for this ith image in the training set"
        ##print "IthImage_NormalizedtmpLandmarkDerivativeIntensityVector %s\n"%(IthImage_NormalizedtmpLandmarkDerivativeIntensityVector[testImageNameCounter-1])
        ####cv.WaitKey()
        ##print "Total %s's landmarks tested=%d"%(currentImageName,len(IthImage_NormalizedtmpLandmarkDerivativeIntensityVector[testImageNameCounter-1]))
        totalLandmarksLeftToBeTested-=1
        ##print "now %d landmarks of %s are left to be tested..before we move the next image's landmark points\n\n"%(totalLandmarksLeftToBeTested, currentImageName)
        ####cv.WaitKey()

        #The IthImage_NormalizedtmpLandmarkDerivativeIntensityVector has been updated with IthImag's landmark profile values
        ##print "The IthImage_NormalizedtmpLandmarkDerivativeIntensityVector has been updated with %s's landmark profile values:\n\n%s\n"%(currentImageName,IthImage_NormalizedtmpLandmarkDerivativeIntensityVector)
    '''

    '''
    ##print "The IthImage_NormalizedtmpLandmarkDerivativeIntensityVector has been updated with %d image point shapes: \n\n%s\n"%(len(self.asm.shapes),IthImage_NormalizedtmpLandmarkDerivativeIntensityVector)
    ####cv.WaitKey()

    #We are going to iterate all image's vectors and sum the corresponding landmark vectors to. Every one on the top of the preexisting other vector.
    sumOflandMarksProfileAccrossAllImages=[]
    testImageNameCounter=0
    landmarkCounter=0
    for normalizedDerivativeProfileLandmarksVector in IthImage_NormalizedtmpLandmarkDerivativeIntensityVector:

        #assign the correct name to the image
        testImageNameCounter+=1
        landmarkCounter=0
        currentImageName="grey_image_"+str(testImageNameCounter)+".jpg"

        ##print "%s's landmarks' profiles=\n\n%s\n\n"%(currentImageName,normalizedDerivativeProfileLandmarksVector)
        ###cv.WaitKey()

        for landmarkNormalizedDerivativeProfile in normalizedDerivativeProfileLandmarksVector:
            landmarkCounter+=1
            ##print "%s's landmark profile%d=%s\n\n"%(currentImageName,landmarkCounter,landmarkNormalizedDerivativeProfile)
            ###cv.WaitKey()

            #calculate g-mean for this ith landmark of this ith image by iterating through all values of g vector
            #for i in landmarkNormalizedDerivativeProfile:
            #modified

            if testImageNameCounter==1:#the first time just append all 30 landmark related vectors
                sumOflandMarksProfileAccrossAllImages.append(np.array(landmarkNormalizedDerivativeProfile))
                ##print "sumOflandMarksProfileAccrossAllImages=%s"%(sumOflandMarksProfileAccrossAllImages)
                ##print "shit happened"
            else:#then just update each of the vectors' values
                sumOflandMarksProfileAccrossAllImages[landmarkCounter-1]=np.array(sumOflandMarksProfileAccrossAllImages[landmarkCounter-1]) + (np.array(landmarkNormalizedDerivativeProfile))
                ##print "sumOflandMarksProfileAccrossAllImages=%s"%(sumOflandMarksProfileAccrossAllImages)
                ##print "all good"


    #Now caclulate the mean for each one (eq. 14 of the 'Subspace Methods for Pattern Recognition in Intelligent Environment' book )
    #and put it in a vector
    githLandmarkMeanAcrossAllImagesVector=[]
    for sumYith in sumOflandMarksProfileAccrossAllImages:
        githLandmarkMeanAcrossAllImagesVector.append(sumYith / len(self.asm.shapes))#so we end up calculating 30 mean (6 element) vectors, which are each landmarks' mean across all images' in the training set
        ##print "githLandmarkMeanAcrossAllImagesVector=\n\n%s\n\n"%(githLandmarkMeanAcrossAllImagesVector)
        ####cv.WaitKey()
    ###cv.WaitKey()

    ##print "githLandmarkMeanAcrossAllImagesVector=\n\n%s\n\n"%(githLandmarkMeanAcrossAllImagesVector)
    ###cv.WaitKey()
    ##print "IthImage_NormalizedtmpLandmarkDerivativeIntensityVector[0][0].size=%d elements"%(len(IthImage_NormalizedtmpLandmarkDerivativeIntensityVector[0][0]))
    ###cv.WaitKey()
    ##print "githLandmarkMeanAcrossAllImagesVector.size=%d elements"%(len(githLandmarkMeanAcrossAllImagesVector[0]))
    ###cv.WaitKey()

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
            ##print "Calculate the Covariance Matrix between\n: yij_ithLandmarkVector=%s and yj_mean[%d]=%s \n"%(np.array(yij_ithLandmarkVector),landmarkAtWhiskerXMeanIndex, yj_mean)
            covMat=np.cov(np.array(yij_ithLandmarkVector),yj_mean)
            landmarkAtWhiskerXMeanIndex+=1
            ##print "covMat=%s"%(covMat)
            ####cv.WaitKey()
            covMatDeterminant=np.linalg.det(covMat)
            ##print "covMatDeterminant=%s"%(covMatDeterminant)
            ####cv.WaitKey()

            covMatInverse = np.linalg.inv(covMat)
            #mahalanobisDist = scipy.spatial.distance.mahalanobis(  yij_ithLandmarkVector, yj_mean, covMatInverse )
            ###print "mahalanobis=%s"%(mahalanobisDist)
            ####cv.WaitKey()

    '''


    '''
    ##print "Calculate the Covariance Matrix between\n: IthImage_NormalizedtmpLandmarkDerivativeIntensityVector[0][0]=%s and githLandmarkMeanAcrossAllImagesVector[0]=%s \n"%(np.array(IthImage_NormalizedtmpLandmarkDerivativeIntensityVector[0][0]),githLandmarkMeanAcrossAllImagesVector[0])
    covMat=np.cov(np.array(IthImage_NormalizedtmpLandmarkDerivativeIntensityVector[0][0]),githLandmarkMeanAcrossAllImagesVector[0])
    ##print "covMat=%s"%(covMat)
    ###cv.WaitKey()
    covMatDeterminant=np.linalg.det(covMat)
    ##print "covMatDeterminant=%s"%(covMatDeterminant)
    ###cv.WaitKey()
    '''



    '''
    #LandMark X Derivatives Vector Out of All Images In The Training Set
    LandMarkX_DerivativesVectorSum=nparray()

    imageCounter=0
    landmarkCounter=0
    # Loop over rows. (each tested image)
    for row in IthImage_NormalizedtmpLandmarkDerivativeIntensityVector:
        imageCounter=imageCounter+1
        ##print "Training Image:%d"%(imageCounter)

        # Loop over columns. (each landmark's normalized derivative profile)
        landmarkCounter=0
        for column in row:
            landmarkCounter=landmarkCounter+1
            ##print "derivative profile: landmark %d = %s"%(landmarkCounter,column)

            ####cv.WaitKey()
        ##print("\n")
    ###cv.WaitKey()




    imageCounter=0
    # Loop over rows. (each tested image)
    for row in IthImage_NormalizedtmpLandmarkDerivativeIntensityVector:
        imageCounter=imageCounter+1
        ##print "Training Image:%d"%(imageCounter)

        # Loop over columns. (each landmark's normalized derivative profile)
        landmarkCounter=0
        for column in row:
            landmarkCounter=landmarkCounter+1
            ##print "derivative profile: landmark %d = %s"%(landmarkCounter,column)

            #create an element for each landmark added,
            LandMarkX_DerivativesVectorSum[landmarkCounter]=LandMarkX_DerivativesVectorSum[LandMarkX_DerivativesVector] + column
            ##print

            ####cv.WaitKey()
        ##print("\n")

    '''


###print ("x=%d,y=%d")%(x,y)
#Gives the landmark points of 1 image out of the training set
###print "Gives the landmark points of this-each image out of the training set"
###print ("oneTrainingImageVector=%s")%(oneTrainingImageVector.pts)
####cv.WaitKey()






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

                ##print "\n\n\n\n\n  !!!!!!!!!!!!!!!!!!!!!! New landmark point mahalanobis calculation !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

                currentLandmarkProfilesNormalizedDerivativesVector=[]

                #along normal (whisker)
                for t in drange(-3, 3, 1):
                        # Normal to normal...
                        ###print "norm",norm


                        ###print "p",(p.x,p.y)
                        ###print "new_p",(new_p)
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

                        ##print "tmpProfileIntensityVector:%s"%tmpProfileIntensityVector


                        #(equation 12 of the 'Subspace Methods for Pattern Recognition in Intelligent Environment' book)
                        normalizedBy=0
                        for intensity in range(len(tmpProfileIntensityVector)-1):

                                #calculate derivative profile intensity g_image[i] - g_image[i-1]...and so on.... dg vector
                                #....................
                                #....................
                                difference=tmpProfileIntensityVector[intensity+1]-tmpProfileIntensityVector[intensity]

                                ##print "%d-%d"%(tmpProfileIntensityVector[intensity+1],tmpProfileIntensityVector[intensity])

                                #store derivative gray-level profile vector
                                tmpProfileDerivativeIntensityVector.append( difference )

                                normalizedBy=normalizedBy+difference

                        ##print "tmpProfileDerivativeIntensityVector: %s"%(tmpProfileDerivativeIntensityVector)
                        ##print "normalizedBy: %s"%(normalizedBy)

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

                        ##print "NormalizedTmpProfileDerivativeIntensityVector %s\n"%(NormalizedTmpProfileDerivativeIntensityVector)
                        cv.WaitKey(10)


                        #fill in each normalized derivative profile vector for this landmark and create the currentLandmarkProfilesNormalizedDerivativesVector
                        #essentially containing all normalized derivative profiles for current landmark point, of which we should calculate the mean

                        currentLandmarkProfilesNormalizedDerivativesVector.append(NormalizedTmpProfileDerivativeIntensityVector)
                        ##print "currentLandmarkProfilesNormalizedDerivativesVector total: %s"%(len(currentLandmarkProfilesNormalizedDerivativesVector))
                        ##print "currentLandmarkProfilesNormalizedDerivativesVector: %s"%(currentLandmarkProfilesNormalizedDerivativesVector)

                        #(equation 14 of the 'Subspace Methods for Pattern Recognition in Intelligent Environment' book)

                # now sum all currentLandmarkProfilesNormalizedDerivativesVector elements together and divide by len(currentLandmarkProfilesNormalizedDerivativesVector)
                # which is the total number of dormalized derivative profile vectors for this landmark point
                #/////////////currentLandmarkProfilesNormalizedDerivativesVector/len(currentLandmarkProfilesNormalizedDerivativesVector)
        '''

    '''
                        ###print "Gmean",Gmean.pts
                        currentLandmarkProfiles.append(tmpProfileIntensityVector)

                for j in range(len(currentLandmarkProfiles)):
                        ##print "profile %d with %d points: %s"%(j, len(currentLandmarkProfiles[j]), currentLandmarkProfiles[j])
                cv.WaitKey(1000)
         '''




                ###print "total profiles",len(profiles)
                #cv.WaitKey(1000)
                #now calculate the derivative profile dg

    '''
                for i in range(len(profiles)):
                        ##print "i",profiles[0].pts
                        ###cv.WaitKey()
    '''

                #now calculate the meanG profile for this landmark , as well as the covariance matrix for this landmark



                ###print "profile number:",len(profiles)
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
    ##print "width=%d,height=%d"%(width,height)

    #Save out the image and move on to the new cv2 way of doing things so as to gray scale it
    cv.SaveImage("InputToGrayScale.jpg", i)
    InputToGrayScaleImg=cv2.imread("InputToGrayScale.jpg")
    ####cv.WaitKey()

    grey_image = np.zeros((width,height,1), np.uint8)


    #cv.WaitKey(1000)

    width=width/scale
    height=height/scale

    grey_image_small = np.zeros((width,height,1), np.uint8)

    grey_image = cv2.cvtColor(InputToGrayScaleImg, cv2.COLOR_BGR2GRAY)
    #cv.Resize(grey_image, grey_image_small)#, interpolation=cv.CV_INTER_NN)
    #cv.Resize(grey_image_small, grey_image)#, interpolation=cv.CV_INTER_NN)
    #grey_image = cv2.GaussianBlur(grey_image,(3,3),0)

    ###print grey_image[0][:]
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


    cv.CvtColor(grey_imageFormattedToOldCvVersion, grey_imageFor, cv2.COLOR_BGR2GRAY)

    return grey_imageFor


  def calculateIthImageLandmarkDerivIntensityVec(self, p, norm, greyImage, gaussianMatrix, _searchProfiles, IthImage_NormalizedtmpLandmarkDerivativeIntensityVector, gaussianWindow):

      scale=0
      print 'greyImage.channels=%s'%(greyImage.channels)

      #c=cv.WaitKey()
      #if c==1048603 :#whichever integer key code makes the app exit
      #    exit()

      #along the whisker
      for side in range(-3,4):#along normal profile
          # Normal to normal...
          ###print"norm",norm

          #CHANGE BACK to the FOLLOWING LINE if it does not work
          #new_p = Point(p.x + side*-norm[1], p.y + side*norm[0])#why ..the other way around? best way to form the search window 12 x 7 = 84 pixels wide window
          new_p = Point(p.x + side*norm[0], p.y + side*norm[1])


          ###print"p",(p.x,p.y)
          ###print"new_p",(new_p)

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
              #y = int((norm[1]*t + new_p.y))#*math.cos(t*(math.pi/180)))
              x = int((new_p.x + t*-norm[1]))#*math.sin(t*(math.pi/180)))
              y = int((new_p.y + t*norm[0]))#*math.cos(t*(math.pi/180)))


              cv.Circle(gaussianWindow, ( int(x), int(y) ), 2, (255,0,50))
              cv.NamedWindow("gaussianWindow", cv.CV_WINDOW_AUTOSIZE)
              cv.ShowImage("gaussianWindow",gaussianWindow)

              #c=cv.WaitKey()
              #if c==1048603 :#whichever integer key code makes the app exit
              #    exit()



              #(equation 11 of the 'Subspace Methods for Pattern Recognition in Intelligent Environment' book)
              scale=0


              #add to g (the intensity value of next pixel along the perpendicular to the whisker.. search profile)
              gradientIntensity=greyImage[y,x] * self.gaussianMatrix[t+6][side+3] #MATRIX CHANGED

              #Add weights properly based on gaussian distribution generated matrix
              ##print"gaussianMatrix[%d][%d]=%s gaussian matrix component was multiplied with.."%(t+6,side+3,gaussianMatrix[t+6][side+3])
              ##print"greyImage[%d][%d,%d]=%s"%(scale,y,x,greyImage[scale][y,x])

              #c=cv.WaitKey(1)
              ###print"EXIT when ESC keycode is pressed=%d"%(c)
              #if c == 1048603:
              #    exit()

              #store gray-level profile in tmpProfileIntensityVector ..g vector
              tmpProfileIntensityVector.append(gradientIntensity)
              ##print"tmpProfileIntensityVector=%f"%(tmpProfileIntensityVector[-1])

          ##print"point =%d along the whisker:\n , tmpProfileIntensityVector:%s\n"%(side,tmpProfileIntensityVector)

          '''store g# intensity profile'''
          _searchProfiles.append(tmpProfileIntensityVector)
          ##print"_searchProfiles =%s\n"%(_searchProfiles)
          ####cv.WaitKey()


          #Landmark Search Profile with Weights is calculated
          ##print"Landmark Search Profile with Weights was calculated\n"
          ##print"tmpProfileIntensityVector:%s\n"%(tmpProfileIntensityVector)
          ####cv.WaitKey()

          '''
          sumGith=0
          averageGith=0
          #now average each gith and save it to currentLandmarkProfiles vector
          for i in tmpProfileIntensityVector:
              sumGith=sumGith + i
          ##print"sumGith :%f\n"%(sumGith)
          ####cv.WaitKey()


          #save the averageGith in currentLandmarkProfiles vector, which corresponds to this point along the whisker
          averageGith=sumGith/len(tmpProfileIntensityVector)
          ##print"averageGith=%s"%(averageGith)
          ####cv.WaitKey()
          currentLandmarkProfiles.append(averageGith)
          ##print"currentLandmarkProfiles=%s"%(currentLandmarkProfiles)
          ####cv.WaitKey()
          '''
      #find g mean profile vector for this landmark
      g_mean=[]


      print"_searchProfiles=%s"%(len(_searchProfiles))
      #cv.WaitKey()
      for profile in _searchProfiles:
          if not g_mean:
              g_mean.append(np.array(profile))
          else:
              g_mean[0]=g_mean[0]+np.array(profile)
          ###print"np.array(profile)=%s"%(np.array(profile))
          ##print"g_mean=%s"%(g_mean)

      #Calculate g_mean, mean of all _searchprofiles
      for i in g_mean:#just one element since its 1 numpy array
          g_mean[0]=i/len(_searchProfiles)#this is executed only once
          ##print"I: %s=%s/%s"%(i,i,len(_searchProfiles))#divide all numpy array elements in place
          ##print"g_mean averaged=%s"%(g_mean)


      #now calculate dg- subtract each profile from the other to find the derivative profile
      dg_landmarkDerivativeProfiles=[]
      for profileIndex in range(len(_searchProfiles)-1):

          #calculate derivative profile intensity g_image[i] - g_image[i-1]...and so on.... dg vector
          #....................
          #....................


          difference=np.array(_searchProfiles[profileIndex+1])-np.array(_searchProfiles[profileIndex])
          ##print"\n_searchProfiles[profileIndex+1]=%s\n - _searchProfiles[profileIndex]=%s\n .. difference=%s\n"%(_searchProfiles[profileIndex+1], _searchProfiles[profileIndex], difference)
          dg_landmarkDerivativeProfiles.append( (difference) )#np.fabs
          ####cv.WaitKey()




      #print"\dg vector=%s\n"%(dg_landmarkDerivativeProfiles)
      #cv.WaitKey()

      #print"dg_landmarkDerivativeProfiles=%s"%(len(dg_landmarkDerivativeProfiles))
      #cv.WaitKey()

      #calculate normalized dg FIX THIS
      sum_dg=[]#contains only one numpy array
      for profileIndex in range(len(dg_landmarkDerivativeProfiles)):
          if not sum_dg:
              sum_dg.append( np.fabs(np.array(dg_landmarkDerivativeProfiles[profileIndex])) )
          else:
              sum_dg[0]=sum_dg[0]+np.fabs(np.array(dg_landmarkDerivativeProfiles[profileIndex]))
          ###print"np.array(profile)=%s"%(np.array(profile))
          ###print"\sum_dg=%s\n"%(sum_dg)
          ####cv.WaitKey()

      ##print"\sum_dg vector=%s\n"%(sum_dg[0])
      ####cv.WaitKey()



      #an array of 6 by13 elements derived by the normalization of each dg
      yij_normalizedDerivProfile=[]

      #Calculate dg magnitude based on: |a| = sqrt((ax * ax) + (ay * ay) + (az * az))
      sum=0

      for profileIndex in range(len(dg_landmarkDerivativeProfiles)):

          #square all elements of dg (derivative profiles)
          squared=np.power(dg_landmarkDerivativeProfiles[profileIndex] , 2)
          print "math.pow(dg_landmarkDerivativeProfiles[profileIndex] , 2) = %s"%(np.power(dg_landmarkDerivativeProfiles[profileIndex] , 2))
          #c=cv.WaitKey()
          #sum them up
          summed=np.sum(squared)
          print "summed = %s"%(summed)
          #c=cv.WaitKey()
          #sqrt them to derive this dg's magnitude
          sqrooted=np.sqrt(summed)
          print "sqrooted = %s"%(sqrooted)
          #c=cv.WaitKey()

          dg_magnitude=sqrooted

          print "dg_landmarkDerivativeProfiles[profileIndex] = %s"%(dg_landmarkDerivativeProfiles[profileIndex])
          #c=cv.WaitKey()
          normalizedDg = dg_landmarkDerivativeProfiles[profileIndex]/dg_magnitude
          print "normalizedDg = %s"%(normalizedDg)
          #c=cv.WaitKey()

          yij_normalizedDerivProfile.append(normalizedDg)



      yij_normalizedDerivProfile=np.asarray(yij_normalizedDerivProfile)



      #for this ith image for this jth landmark derive calculate the normalized derivative profile
      ###################################################################################################################################################
      ###################################################################################################################################################
      '''TOOK THE NEXT LINE OUT as now Normalization of each of the derivative profiles for this landmark is calculated properly above in the for loop'''
      #yij_normalizedDerivProfile=np.array(dg_landmarkDerivativeProfiles) / np.array(sum_dg[0]) #equation 13 of 'Subspace Methods for Pattern Recognition in Intelligent Environment' book
      ###################################################################################################################################################
      ###################################################################################################################################################

      print"\y_ij vector=%s\n"%(yij_normalizedDerivProfile)
      #c=cv.WaitKey()
      #if c==1048603 :#whichever integer key code makes the app exit
      #   exit()


      '''save this & every other landmark's normalized derivative profile for this image '''
      IthImage_NormalizedtmpLandmarkDerivativeIntensityVector.append(yij_normalizedDerivProfile)
      ##print"\n%d , IthImage_NormalizedtmpLandmarkDerivativeIntensityVector=%s\n"%(len(IthImage_NormalizedtmpLandmarkDerivativeIntensityVector),IthImage_NormalizedtmpLandmarkDerivativeIntensityVector)
      ####cv.WaitKey()


  #find the G mean for all landmarks in the training set of shapes
  def findLandmarkNormIIntesityVec(self, gMeanFlag, oneTrainingImageVector, shape, trainingSetImage, IthImage_NormalizedtmpLandmarkDerivativeIntensityVector,gaussianMatrix, greyImage):

      gaussianWindow = cv.CreateImage(cv.GetSize(greyImage), greyImage.depth, greyImage.channels)
      cv.Copy(greyImage, gaussianWindow)

      '''for each of each shape's landmarks'''
      for i,p in enumerate(shape.pts):

          #print "p.x=%d ,p.y=%d"%(p.x,p.y)
          ##cv.WaitKey()

          ##if g mean of training set
          if gMeanFlag==1:
              '''draw them in a window named trainingSetImage'''
              tmpP = Point(p.x, p.y)
              cv.Circle(trainingSetImage, ( int(tmpP.x), int(tmpP.y) ), 4, (100,100,100))
              cv.NamedWindow("trainingSetImage", cv.CV_WINDOW_AUTOSIZE)
              cv.ShowImage("trainingSetImage",trainingSetImage)
              #cv.WaitKey()
          elif gMeanFlag==0:
              '''draw them in a window named targetImage'''
              tmpP = Point(p.x, p.y)
              cv.Circle(trainingSetImage, ( int(tmpP.x), int(tmpP.y) ), 4, (100,100,100))
              cv.NamedWindow("targetImage", cv.CV_WINDOW_AUTOSIZE)
              cv.ShowImage("targetImage",trainingSetImage)
              #cv.WaitKey()

          #this image's current landmark point; for each of these landmarks calculate a 2d windows search profile
          #x=(p.x)
          #y=(p.y)

          #create a list of the g profiles for each landmark point
          currentLandmarkProfiles = []

          #store point
          p = p

          #for each point.. get normal direction of this the point based on the 2 adjacent point
          norm =  shape.get_normal_to_point(i)#??? self.shape

          ##print"\n\n\n\n\n  !!!!!!!!!!!!!!!!!!!!!! New landmark point SEARCH PROFILE calculation !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

          tmpLandmarkDerivativeIntensityVector=[]
          currentLandmarkProfilesNormalizedDerivativesVector=[]

          #g vectors
          _searchProfiles=[]

          '''calculate, update-append to IthImage_NormalizedtmpLandmarkDrivativeIntesityVector (for each landmark for this shape)'''
          self.calculateIthImageLandmarkDerivIntensityVec(p, norm, greyImage, gaussianMatrix, _searchProfiles, IthImage_NormalizedtmpLandmarkDerivativeIntensityVector, gaussianWindow)

      print "IthImage_NormalizedtmpLandmarkDerivativeIntensityVector LENGTH %d"%(len(IthImage_NormalizedtmpLandmarkDerivativeIntensityVector[0]))
      #c=cv.WaitKey()
      #if c==1048603 :#whichever integer key code makes the app exit
      #   exit()

      #this vector contains the normalized derivative profiles for all landmarks of this shape
      return IthImage_NormalizedtmpLandmarkDerivativeIntensityVector



  def findtrainingGmean(self, testImageNameCounter, AllImages_NormalizedtmpLandmarkDerivativeIntensityVector, gaussianMatrix):

      '''for each image/shape'''
      gMeanFlag=1;
      for asmshape in self.originalShapes:#self.asm.shapes !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 3 for now

          #add another list of derivative profile vectors for all landmarks which correspond to each image
          #IthImage_NormalizedtmpLandmarkDerivativeIntensityVector.append([])
          IthImage_NormalizedtmpLandmarkDerivativeIntensityVector=[]

          #1)for each image, convert to grayscale, iterate over the  corresponding points, and multiply profile pixel intensity with corresponding kernel gaussian distr. matrix element,
          #..then get the average of them for each tangential search profile and store it in the gith position of the g vector of elements along the whisker.


          #just test pixel value
          ###print'my greyImage_gradient:'
          ###printgreyImage[0][207,282]


          ####cv.WaitKey()
          ##print"testing image=%s"%("grey_image_"+str(testImageNameCounter)+".jpg")
          ####cv.WaitKey()


          #convert this test_grey_image to grayscale
          greyImage = []
          testImageNameCounter=testImageNameCounter+1
          '''so for shape 1..load  gray_image_1, for shape 2..load  gray_image_2 etc'''

          currentImageName="grey_image_"+str(testImageNameCounter)+".jpg"
          test_grey_image = cv.LoadImage(currentImageName)


          '''greyImage will have the greyscale image marked with landmarks'''
          greyImage.append(self.__produce_gradient_image(test_grey_image, 2**0))


          '''create a copy of the greyscaled image to put the new landmarks onto'''
          trainingSetImage = cv.CreateImage(cv.GetSize(greyImage[0]), greyImage[0].depth, 1)
          cv.Copy(greyImage[0], trainingSetImage)


          #print "len greyImage=%s"%(len(greyImage[0]))

          #s=[]
          #asmshape.add_point(Point(100,100))
          #asmshape.add_point(Point(200,200))
          #asmshape.add_point(Point(300,300))

          #print "asm shape x&y length flattend (..so divide by 2 to get the total coordinate points) : %s"%( len(asmshape.get_vector()) )
          #print "asm shape : %s"%( asmshape.get_vector() )
          #cv.WaitKey()

          '''get a vector of all landmark points of this shape'''
          asmshapePointVector=Shape.from_vector(asmshape.get_vector())#[320,185,309,196,292,203,276,209,264,217,256,237,261,252,275,255,294,243,308,238,327,240,345,241,342,224,336,217,335,209,348,203,359,210,369,214,376,206,392,211,400,215,404,214,402,198,399,189,390,181,389,172,383,160,373,147,354,145,338,161]
          totalLandmarksLeftToBeTested=len(asmshapePointVector.pts)

          #find g mean of each of the 30 landmarks of each asmshape and save it in IthImage_NormalizedtmpLandmarkDerivativeIntensityVector
          IthImage_NormalizedtmpLandmarkDerivativeIntensityVector=self.findLandmarkNormIIntesityVec (gMeanFlag, asmshapePointVector, asmshape, trainingSetImage, IthImage_NormalizedtmpLandmarkDerivativeIntensityVector,gaussianMatrix, trainingSetImage)

          #c=cv.WaitKey()
          #if c==1048603 :#whichever integer key code makes the app exit
          #   exit()

          #save each image's  normalized derivative profile which contains all  normalized derivative profiles for each of its landmarks
          AllImages_NormalizedtmpLandmarkDerivativeIntensityVector.append(IthImage_NormalizedtmpLandmarkDerivativeIntensityVector)
          #print"\n\length of IthImage_NormalizedtmpLandmarkDerivativeIntensityVector vector=%s\n"%(len(IthImage_NormalizedtmpLandmarkDerivativeIntensityVector))
          #print"\n\IthImage_NormalizedtmpLandmarkDerivativeIntensityVector vector=%s\n"%((IthImage_NormalizedtmpLandmarkDerivativeIntensityVector))


      #every AllImages_NormalizedtmpLandmarkDerivativeIntensityVector is a vector of 3 vectors
      #..each of which contains 30 elements each of which contains 6 profiles each of which contains 13 elements in the tangential direction

      #print"\n\length of IthImage_NormalizedtmpLandmarkDerivativeIntensityVector vector=%s\n"%(len(AllImages_NormalizedtmpLandmarkDerivativeIntensityVector[0][0]))
      #cv.WaitKey()


      #Calculate yj_mean of the training set
      y_j_mean=[]
      meancounter=0

      correctionVectorForAllShapes=[]

      #cv.WaitKey()

      for i in range(len(AllImages_NormalizedtmpLandmarkDerivativeIntensityVector[0])):#for each landmark in the set (30 for now)
        #Calculate yj_mean
        y_j_mean=[]
        meancounter=0


        print '\n'
        for index in range(len(AllImages_NormalizedtmpLandmarkDerivativeIntensityVector)):#for each image in the set (3 for now)

            print "at shape %d"%(index)
            print "landmark %d\n"%(i)


            meancounter+=1
            if not y_j_mean:
                y_j_mean.append( np.array(AllImages_NormalizedtmpLandmarkDerivativeIntensityVector[index][i]))#append 6 profiles with 13 elements each one
            else:
                y_j_mean[0]=y_j_mean[0]+ np.array(AllImages_NormalizedtmpLandmarkDerivativeIntensityVector[index][i])#add the next 6 profiles of the of the same landmarks in training set of images
            ###print"np.array(profile)=%s"%(np.array(profile))
            ##print"\y_j_mean=%s\n"%(y_j_mean[0])
            ##cv.WaitKey()

        #print "len(AllImages_NormalizedtmpLandmarkDerivativeIntensityVector)=%s"%(len(AllImages_NormalizedtmpLandmarkDerivativeIntensityVector))
        #print "len(AllImages_NormalizedtmpLandmarkDerivativeIntensityVector[index])=%s"%(len(AllImages_NormalizedtmpLandmarkDerivativeIntensityVector[index]))
        #cv.WaitKey()

        #print "meancounter=%d"%(meancounter)
        y_j_mean = np.array(y_j_mean) / meancounter#len(self.asm.shapes)   #equation 14 of 'Subspace Methods for Pattern Recognition in Intelligent Environment' book

        #print "y_j_mean=%s"%(y_j_mean)
        #cv.WaitKey()

        correctionVectorForAllShapes.append(y_j_mean)#THIS IS storing THE MEAN OF THE NORMALIZED DERIVATIVE PROFILE FOR EACH LANDMARK accross the training set

        print "len(correctionVectorForAllShapes)=%s"%(len(correctionVectorForAllShapes))
        #cv.WaitKey()

      return correctionVectorForAllShapes


  def findcurrentShapeGmean(self, currentshape, One_NormalizedtmpLandmarkDerivativeIntensityVector, gaussianMatrix):

      scale=0
      gMeanFlag=0

      '''for each image/shape'''
      #for asmshape in self.asm.shapes:#self.originalShapes

      #add another list of derivative profile vectors for all landmarks which correspond to each image
      #curShape_NormalizedtmpLandmarkDerivativeIntensityVector.append([])
      curShape_NormalizedtmpLandmarkDerivativeIntensityVector=[]

      #1)for each image, convert to grayscale, iterate over the  corresponding points, and multiply profile pixel intensity with corresponding kernel gaussian distr. matrix element,
      #..then get the average of them for each tangential search profile and store it in the gith position of the g vector of elements along the whisker.


      #just test pixel value
      ###print'my greyImage_gradient:'
      ###printgreyImage[0][207,282]


      ####cv.WaitKey()
      ##print"testing image=%s"%("grey_image_"+str(testImageNameCounter)+".jpg")
      ####cv.WaitKey()

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


      #print "len greyImage=%s"%(len(greyImage[0]))

      #s=[]
      #currentshape.add_point(Point(100,100))
      #currentshape.add_point(Point(200,200))
      #currentshape.add_point(Point(300,300))

      #print "asm shape x&y length flattend (..so divide by 2 to get the total coordinate points) : %s"%( len(currentshape.get_vector()) )
      #print "asm shape : %s"%( currentshape.get_vector() )
      #cv.WaitKey()

      '''get a vector of all landmark points of this shape'''
      currentshapePointVector=Shape.from_vector(currentshape.get_vector())#[320,185,309,196,292,203,276,209,264,217,256,237,261,252,275,255,294,243,308,238,327,240,345,241,342,224,336,217,335,209,348,203,359,210,369,214,376,206,392,211,400,215,404,214,402,198,399,189,390,181,389,172,383,160,373,147,354,145,338,161]
      totalLandmarksLeftToBeTested=len(currentshapePointVector.pts)

      #trainingSetImage = cv.CreateImage(cv.GetSize(self.g_image[scale]), self.g_image[scale].depth, 1)
      #cv.Copy(self.g_image[scale], trainingSetImage)

      #find g mean of 30 landmarks of each shape and save it in curShape_NormalizedtmpLandmarkDerivativeIntensityVector
      self.findLandmarkNormIIntesityVec (gMeanFlag, currentshapePointVector, currentshape, self.image, curShape_NormalizedtmpLandmarkDerivativeIntensityVector,gaussianMatrix, self.g_image[scale])


      #save each image's  normalized derivative profile which contains all  normalized derivative profiles for each of its landmarks
      One_NormalizedtmpLandmarkDerivativeIntensityVector.append(curShape_NormalizedtmpLandmarkDerivativeIntensityVector)
      #print"\n\length of curShape_NormalizedtmpLandmarkDerivativeIntensityVector vector=%s\n"%(len(curShape_NormalizedtmpLandmarkDerivativeIntensityVector))
      #print"\n\curShape_NormalizedtmpLandmarkDerivativeIntensityVector vector=%s\n"%((curShape_NormalizedtmpLandmarkDerivativeIntensityVector))


      #every One_NormalizedtmpLandmarkDerivativeIntensityVector is a vector of 3 vectors
      #..each of which contains 30 elements each of which contains 6 profiles each of which contains 13 elements in the tangential direction

      #print"\n\length of curShape_NormalizedtmpLandmarkDerivativeIntensityVector vector=%s\n"%(len(One_NormalizedtmpLandmarkDerivativeIntensityVector[0][0]))
      #cv.WaitKey()

      print"\nlength of One_NormalizedtmpLandmarkDerivativeIntensityVector[0]=%s\n"%(len(One_NormalizedtmpLandmarkDerivativeIntensityVector[0]))
      #cv.WaitKey()

      print"\n\length of One_NormalizedtmpLandmarkDerivativeIntensityVector=%s\n"%(len(One_NormalizedtmpLandmarkDerivativeIntensityVector))
      #cv.WaitKey()


      #Calculate yj_mean of the training set
      currentShape_y_j_mean=[]
      meancounter=0

      correctionVectorForAllShapes=[]

      #cv.WaitKey()
      #f = open('landmarksTestGChosen/landmark_g_Chosen','w')
      for i in range(len(One_NormalizedtmpLandmarkDerivativeIntensityVector[0])):#for each landmark in the set (30 for now)
        #Calculate yj_mean
        currentShape_y_j_mean=[]
        meancounter=0

        for index in range(len(One_NormalizedtmpLandmarkDerivativeIntensityVector)):#for each image in the set (3 for now)

            print "at shape %d"%(index)
            print "landmark %d"%(i)
            #cv.WaitKey()


            meancounter+=1
            if not currentShape_y_j_mean:
                currentShape_y_j_mean.append( np.array(One_NormalizedtmpLandmarkDerivativeIntensityVector[index][i]))#append 6 profiles with 13 elements each one
            else:
                currentShape_y_j_mean[0]=currentShape_y_j_mean[0]+ np.array(One_NormalizedtmpLandmarkDerivativeIntensityVector[index][i])#add the next 6 profiles of the of the same landmarks in training set of images
            ###print"np.array(profile)=%s"%(np.array(profile))
            print"\currentShape_y_j_mean=%s\n"%(currentShape_y_j_mean[0])
            #cv.WaitKey()

        #print "len(One_NormalizedtmpLandmarkDerivativeIntensityVector)=%s"%(len(One_NormalizedtmpLandmarkDerivativeIntensityVector))
        #print "len(One_NormalizedtmpLandmarkDerivativeIntensityVector[index])=%s"%(len(One_NormalizedtmpLandmarkDerivativeIntensityVector[index]))
        #cv.WaitKey()

        #print "meancounter=%d"%(meancounter)
        #cv.WaitKey()
        currentShape_y_j_mean = np.array(currentShape_y_j_mean) / meancounter#len(self.asm.shapes)
        correctionVectorForAllShapes.append(currentShape_y_j_mean)

      return correctionVectorForAllShapes


  def getCorrectedLandmarkPoint(self,trainingGmean):

      ###print"asm.mean",asm.mean
      scale=0

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

#      ##print"gaussianMatrix=%s"%(gaussianMatrix)

#      #iterate the gaussian distribution matrix
#      for (x,y), value in np.ndenumerate(gaussianMatrix):
#                  print"%d, %d =%s"%(x,y,gaussianMatrix[x][y])
#                  ####cv.WaitKey()


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
      #print "len(trainingGmean)=%s"%(len(trainingGmean))
      #cv.WaitKey()


      #Now find y_j_mean for the current shape and then perform mahalanobis check between the mean_obtained_from_training and the current mean for each landmark
      print 'Now find y_j_mean for the current shape and then perform mahalanobis check between the mean_obtained_at_training and the each of the current g profiles of each landmark to find match mathc than minimizes..'
      #cv.WaitKey()

      print"\n trainingGmean vector=\n%s\n"%(trainingGmean)
      #cv.WaitKey()


      #target image
      self.image
      self.g_image[scale]
      #get a vector of all current landmark points
      currentShape=self.shape#Shape.from_vector(self.asm.shapes[index].get_vector())#self.originalShapes
      #find y_j_mean for the current shape
      One_NormalizedtmpLandmarkDerivativeIntensityVector = []
      #probably currentShapeGmean is not needed
      currentShapeGmean = self.findcurrentShapeGmean(currentShape, One_NormalizedtmpLandmarkDerivativeIntensityVector, self.gaussianMatrix)


      print "len(One_NormalizedtmpLandmarkDerivativeIntensityVector)=%s"%(len(One_NormalizedtmpLandmarkDerivativeIntensityVector[0]))
      #cv.WaitKey()
      print"\n One_NormalizedtmpLandmarkDerivativeIntensityVector=\n%s\n"%(One_NormalizedtmpLandmarkDerivativeIntensityVector)
      #cv.WaitKey()


      '''CALCULATE ALL COVARIANCE MATRICES ONLY ONCE AT THE START SO AS TO FORM THE GREY LEVEL PROFILE & ...
       and now THAT we have for each landmark a  certain GREY profile yj ,  we can use for the image search,
       and particularly finding the  desired movements of the landmarks that take xi to xi + dxi .'''


      if self.COVARIANCE_CALCULATION_ONCE != True:
          self.COVARIANCE_CALCULATION_ONCE=True

          #x=np.array(One_NormalizedtmpLandmarkDerivativeIntensityVector[0][0])
          #self.tmpCovar = np.cov(x, rowvar=0)#SAME

          for p_num in range(len(One_NormalizedtmpLandmarkDerivativeIntensityVector[0])):#so for all landmarks ..30 for now
              #tmpCovar= cv2.calcCovarMatrix(One_NormalizedtmpLandmarkDerivativeIntensityVector[0][p_num], cv.CV_COVAR_COLS,cv.CV_64F | cv.CV_COVAR_NORMAL | cv.CV_COVAR_USE_AVG, trainingGmean[p_num][0])
              x=np.array(One_NormalizedtmpLandmarkDerivativeIntensityVector[0][p_num])
              tmpCovar=np.cov(x, rowvar=0)
              self.CovarianceMatricesVec.append(tmpCovar)
          cv.WaitKey()




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
      Geometrically, the determinant is the volume of the N dimensional vector space implied by the covariance matrix.  Minimizing the ellipsoid is equivalent to minimizing the volume.
      '''

      if totalLandmarkNumber==-1:
          totalLandmarkNumber=len(self.AllImages_NormalizedtmpLandmarkDerivativeIntensityVector[0])#for 30 totalLandmarkNumber

      #for s in self.asm.shapes:

      correctionVectorForAllShapes=[]
      #for p_num in range(totalLandmarkNumber):#0..29 iterate only for 30 elements, but for a different asmshapePointVector
      #for index in range(len(AllImages_NormalizedtmpLandmarkDerivativeIntensityVector)):


      #Now we need to find the g profile from each landmark of the current shape that minimizes the mahalanobis distance between this and the one found as the Gmean of training set (for this landmark)
      print 'Now we need to find the g profile from each landmark of the current shape that minimizes the mahalanobis distance between this and the one found as the Gmean of training set (for this landmark)'
      #cv.WaitKey()

      #print 'asmshapePointVector=%s'%(asmshapePointVector.pts[0])
      print 'len(One_NormalizedtmpLandmarkDerivativeIntensityVector)=%s'%(len(One_NormalizedtmpLandmarkDerivativeIntensityVector[0][0]))
      #cv.WaitKey()

      f = open('landmarksTestGChosen/landmark_g_Chosen','w')
      for p_num in range(len(One_NormalizedtmpLandmarkDerivativeIntensityVector[0])):
      ###print"\n np.array(IthImage_NormalizedtmpLandmarkDerivativeIntensityVector[p_num])=%s"%(np.array(IthImage_NormalizedtmpLandmarkDerivativeIntensityVector[p_num]).flatten())
      ###print"\n np.array(y_j_mean)=%s"%(np.array(y_j_mean[0]).flatten())

          #x is the intensity vector for this profile , g , for each landmark (contains 6 search profiles x 13 subprofiles)
          x=np.array(One_NormalizedtmpLandmarkDerivativeIntensityVector[0][p_num])
          #y is the mean intensity vector for the 6 profiles (shouldn't this be 7 instead of 6 profiles ??? )..--> no, cause we are taking the derivative! so there will be n-1. Hence ..6
          y=np.array(trainingGmean[p_num][0]) #What am i supposed to get here?? The ith landmark's training G mean (containing 2d intensity info 6 (7-1 dg) whisker points and 13 tangent to normal points)
          #x=x.tolist()
          #y=y.tolist()

          print "x=%s"%(x)
          #cv.WaitKey()

          print "y=%s"%(y)
          #cv.WaitKey()


          #print "len trainingGmean=%s"%(len(trainingGmean))
          #cv.WaitKey()
          #print "len trainingGmean[0]=%s"%(len(trainingGmean[0]))
          #cv.WaitKey()
          #print "len trainingGmean[0][0]=%s"%(len(trainingGmean[0][0]))
          #cv.WaitKey()


          #Now for each landmark one MYCOVAR ..does it relate to mahalanobis calculation correctly


          #################################
          #not currently used bu maybe should constider using it
          #mycovar = np.cov(x, y, rowvar=0)
          #print "mycovar=%s"%(mycovar)
          #print "mycovar=%s"%(len(mycovar))
          #c=cv.WaitKey()
          #if c==1048603 :#whichever integer key code makes the app exit
          #    exit()

          #mycovar=0
          #mycovar=cv2.calcCovarMatrix(x, mycovar, y, cv.CV_COVAR_NORMAL )#| cv.CV_COVAR_COLS



          #print "mycovar=%s"%(mycovar)
          #cv.WaitKey()
          #################################

          ##print"AllImages_NormalizedtmpLandmarkDerivativeIntensityVector[%d][%d]=%s\n"%(index,p_num,x)
          ##print"y_j_mean[0]=%s\n"%(y)


          #calculate Mahalanobis best smaller distance
          covar = np.cov(x, rowvar=0)
          #print "covar=%s"%(covar)
          #print "covar=%s"%(len(covar))
          #c=cv.WaitKey()
          #if c==1048603 :#whichever integer key code makes the app exit
          #    exit()
          #cv.WaitKey()

          #invcovar = np.linalg.inv(covar.reshape((13,13)))

          #invcovar=0
          #invcovar=cv2.invert(mycovar, invcovar, cv2.DECOMP_SVD)#'''covar'''
          #invcovar=invcovar[1]

          #print "invcovar=%s"%(invcovar)
          #print "invcovar=%s"%(len(invcovar))
          #c=cv.WaitKey()
          #if c==1048603 :#whichever integer key code makes the app exit
          #    exit()

          ##print"inverse covariance matrix shape=%s"%(str(invcovar.shape))

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
          #print x
          #y = cv.fromarray(y)
          #print y

          ##MAJOR CHANGE...........
          #cv.CalcCovarMatrix(x, mycovar, y, cv.CV_COVAR_COLS | cv.CV_COVAR_NORMAL | cv.CV_COVAR_USE_AVG)
          #mycovar= cv2.calcCovarMatrix(x, cv.CV_COVAR_COLS,cv.CV_64F | cv.CV_COVAR_NORMAL | cv.CV_COVAR_USE_AVG, y[0])#http://www.programmershare.com/3791731/

          mycovar=self.CovarianceMatricesVec[p_num]#self.tmpCovar
          print "mycovar=%s"%(mycovar)
          print "mycovar=%s"%(len(mycovar))
          #c=cv.WaitKey()
          #if c==1048603 :#whichever integer key code makes the app exit
          #    exit()

          ##MAJOR CHANGE...........

          #divide covariance by row number
          print "mycovar before element division\n"
          print (mycovar[0])
          for i in mycovar[0]:
              i /= 6 #6 is the number of rows


          print "mycovar\n"
          print (mycovar[0])
          #print "mycovar=%s"%(mycovar)
          #print "mycovar=%s"%(len(mycovar))
          #c=cv.WaitKey()
          #if c==1048603 :#whichever integer key code makes the app exit
          #    exit()

          invcovar=0
          invcovar=cv2.invert(mycovar, invcovar, cv2.DECOMP_SVD)#mycovar[0] when used with cv2.calcCovarMatrix #'''OR covar'''
          invcovar=invcovar[1]

          print "invcovar=%s"%(invcovar)
          #c=cv.WaitKey()
          #if c==1048603 :#whichever integer key code makes the app exit
          #  exit()


          print "invcovar=%s"%(invcovar)
          print "invcovar=%s"%(len(invcovar))
          #c=cv.WaitKey()
          #if c==1048603 :#whichever integer key code makes the app exit
          #    exit()


          #print"total_of %d g search profile elements\n"%(x.shape[0])
          ###cv.WaitKey()

          #mahalfile = open('mahalanobisChoiceCheeck/normDerIntProf.txt'+str(p_num),'w')


          #Save all sums of sub tangential subprofiles, of the whisker points --->(all 13 tangential elements' sum)
          SumSubprofElemVecX=[]
          SumSubprofElemVecY=[]

          for j in range(x.shape[0]):#for this landmark the are 6 profiles (13 elements each) of which we find the best mahalanobis for each search profile to select the bestIndexProfile accordingly
          #for j in range(len(x)):


              #s = np.array([[20,55,119],[123,333,11],[113,321,11],[103,313,191],[123,3433,1100]])
              #print "x[j]=%s"%(x[j])
              #cv.WaitKey()
              #co = np.cov(s[0],s[1], rowvar=0)
              #print "co=%s"%(co)
              #cv.WaitKey()




              #print "x[j]=%s"%(x[j])
              #cv.WaitKey()
              print "y=%s"%(y)
              #cv.WaitKey()

              #print "x[j]-y[j]=%s"%(x[j]-y[j])
              #cv.WaitKey()


              #mycovar = np.cov(x[j], y[j], rowvar=0)

              #diff=x-y
              #mycovar = np.cov(diff, rowvar=0)

              #print "mycovar=%s"%(mycovar)
              #print "mycovar=%s"%(len(mycovar))
              #c=cv.WaitKey()
              #if c==1048603 :#whichever integer key code makes the app exit
              #    exit()

              #invcovar=0
              #invcovar=cv2.invert(mycovar, invcovar, cv2.DECOMP_SVD)#'''covar'''
              #invcovar=invcovar[1]

              #print "invcovar=%s"%(invcovar)
              #print "invcovar=%s"%(len(invcovar))
              #c=cv.WaitKey()
              #if c==1048603 :#whichever integer key code makes the app exit
              #    exit()


              print "x=%s"%(x)
              #cv.WaitKey()
              print "y=%s"%(y)
              #cv.WaitKey()

              ##print"test g%d %s...with...gmean%d %s"%(j, x[j],j, y[j])
              tmpMahalDist=scipy.spatial.distance.mahalanobis( x[j], y[j], invcovar)# currently works with fabs applied to inv covar matrix..however this should not be the case
              #tmpMahalDist=scipy.spatial.distance.mahalanobis(x,y,invcovar);
              ##print"g%d MahalanobisDistance=%s\n"%(j,tmpMahalDist)


              #mahalfile.write(str(x[j])+"\n")

              #first time it runs for each landmark
              if bestProfileMatchWithSmallerMahalanobisDistanceFrom_gProf_to_meanProf==-1:
                  bestProfileMatchWithSmallerMahalanobisDistanceFrom_gProf_to_meanProf= tmpMahalDist
                  bestIndexProfile=j


              else:
                  if tmpMahalDist < bestProfileMatchWithSmallerMahalanobisDistanceFrom_gProf_to_meanProf:
                      bestProfileMatchWithSmallerMahalanobisDistanceFrom_gProf_to_meanProf= tmpMahalDist
                      bestIndexProfile=j
                      #print "bestProfileMatchWithSmallerMahalanobisDistanceFrom_gProf_to_meanProf for this landmark=%s"%(bestProfileMatchWithSmallerMahalanobisDistanceFrom_gProf_to_meanProf)


              print "tmpMahalDist for this landmark=%s"%(tmpMahalDist)



              #Add up all tangential subprofile elements
              #Check x[i] & y[i] sums with regards to the intensity of the final pixel chosen
              #x[i]
              #              SumSubprofElem=0
              #              for subprofElem in x[j]:
              #                  SumSubprofElem+=subprofElem
              #                  print "subprofElemX=%f"%(subprofElem)
              #                  #cv.WaitKey()
              #              print "\nSumSubprofElemX=%f"%(SumSubprofElem)
              #              SumSubprofElemVecX.append(SumSubprofElem)
              #              #y[i]
              #              SumSubprofElem=0
              #              for subprofElem in y[j]:
              #                  SumSubprofElem+=subprofElem
              #                  print "subprofElemY=%f"%(subprofElem)
              #                  #cv.WaitKey()
              #              print "\nSumSubprofElemY=%f"%(SumSubprofElem)
              #              SumSubprofElemVecY.append(SumSubprofElem)





          print"profile g%d has the bestProfileMatchWithSmallerMahalanobisDistanceFrom_gProf_to_meanProf=%s"%(bestIndexProfile,bestProfileMatchWithSmallerMahalanobisDistanceFrom_gProf_to_meanProf)
          #cv.WaitKey()

          ##print"g%d model profile chosen:\n%s"%(bestIndexProfile,x[bestIndexProfile])
          ##print"landmark%d to be corrected based on g%d best profile model"%(p_num,bestIndexProfile)
          ###cv.WaitKey()


          ##########################################################################################
          ##########################################################################################
          #calculate the normal corresponding to point p_num of the mean shape

          ##print"totalLandmarkNumber=%d"%(totalLandmarkNumber)
          ##print"currentShape=%d"%(len(currentShape.pts))
          ##print"\n\npoint number=%d\n\n"%(p_num)
          norm = currentShape.get_normal_to_point(p_num)
          ##print"norm\n",norm

          #if p_num==29 :
          #    ##cv.WaitKey()

          #get p, each of the 2d points of the mean shape shape in the training model
          p = currentShape.pts[p_num]#1st point ..2nd point.. and so on
          ##print"p_num%d"%(p_num)
          ###cv.WaitKey()
          ##print"p=%s"%(p)
          ###cv.WaitKey()

          ##print"Points of the mean shape of the Trained Set",currentShape.pts

          # Scan over the whole line
          max_pt = p
          max_edge = 0

          #based on: 2(m - k) + 1  , from http://www.face-rec.org/algorithms/AAM/app_models.pdf
          #m from 0<=m<=5
          #k 3 pixels on each side
          side= bestIndexProfile-3# 2*(bestIndexProfile-3)+1


          print"side=%d"%(side)
          ###cv.WaitKey()


          f.write('for landmark '+str(p_num)+') '+str(side)+'     is chosen')
          f.write("\n")


          imgtmp = cv.CreateImage(cv.GetSize(self.image), self.g_image[scale].depth, 1)
          cv.Copy(self.g_image[scale], imgtmp)


          #along the whisker
          '''for side in range(-3,4):#along normal profile'''
          # Normal to normal...

          new_p = Point(p.x + side*-norm[1], p.y + side*norm[0])#why ..the other way around?
          #new_p = Point(new_p.x + side*norm[0], new_p.y + side*norm[1])
          max_pt = new_p


          targetimage = cv.LoadImage("fluoSpine.jpg")
          mytargetImage=[]
          '''greyImage will have the greyscale image marked with landmarks'''
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
              print"gradientIntensity=%d"%(gradientIntensity)

              mahalfile.write("gradientIntensity=%d"%(gradientIntensity)+"\n")

              #cv.WaitKey()

          tmpPointChosen = Point(p.x + side*-norm[1], p.y + side*norm[0])#chosen one
          myX = int(tmpPointChosen.x)
          myY = int(tmpPointChosen.y)
          gradientIntensityChosen=mytargetImage[0][myY, myX]
          mahalfile.write("CHOSEN pixel with gradientIntensity=%d"%(gradientIntensityChosen)+"\n")

          mahalfile.close()




          iterationcounter=0



          '''
          # Look 6 pixels to each side too
          for t in drange(-6, 7, 1):#tangent
              ###print"t...........",t

              x = int(norm[0]*t + new_p.x)
              y = int(norm[1]*t + new_p.y)


              if x < 0 or x > self.image.width or y < 0 or y > self.image.height:
                continue

              #show min and max points
              #cv.Circle(imgtmp, ( int(norm[0]*min_t + new_p.x) , int(norm[1]*min_t + new_p.y)), 10, (100,100,100))
              #cv.Circle(imgtmp, ( int(norm[0]*max_t + new_p.x) , int(norm[1]*max_t + new_p.y)), 10, (100,100,100))

              ###printx, y, self.greyImage.width, self.greyImage.height

              ###print"greyImage[scale][y, x]",self.greyImage[scale][y,x]

              targetImageTocheckAgainst=self.g_image[scale] #self.image
              if targetImageTocheckAgainst[y-1, x-1] > max_edge:
                max_edge = targetImageTocheckAgainst[y-1, x-1] #greyImage[scale][y, x]
                max_pt = Point(new_p.x + side*norm[0], new_p.y + side*norm[1])
                ##print'update max point'

                ##print"p=%s"%(p)
                ##print"max_pt=%s"%(max_pt)
                ##print"x=%s , y=%s"%(x,y)
                iterationcounter+=1


                #show max point updated position
                #cv.Circle(imgtmp, ( int(max_pt.x), int(max_pt.y) ), 5, (100,100,100))
                #cv.NamedWindow("imgtmpMaxPoint", cv.CV_WINDOW_AUTOSIZE)
                #cv.ShowImage("imgtmpMaxPoint",imgtmp)
                #print 'max point shown'
                #cv.WaitKey()

              ##print"iterationcounter=%d"%(iterationcounter)
              #Show all the points on the normal to choose from..
              tmpP = Point(new_p.x + t*norm[0], new_p.y + t*norm[1])
              cv.Circle(imgtmp, ( int(tmpP.x), int(tmpP.y) ), 3, (100,100,100))
              cv.NamedWindow("imgtmpMaxPoint", cv.CV_WINDOW_AUTOSIZE)
              cv.ShowImage("imgtmpMaxPoint",imgtmp)
              c=cv.WaitKey(10)
              if c ==1048603:
                  exit()
          '''



          #shows the whisker from which which chose points
          for t in range(-6,7,1):
              #tmpP = Point(p.x + t*-norm[1], p.y + t*norm[0])
              #tmpP = Point(p.x + t*-norm[1], p.y + t*norm[0])



              #tmpP = Point(new_p.x + t*norm[0], new_p.y + t*norm[1])

              #the possible points to chose from
              #tmpP = Point(p.x + t*norm[0], p.y + t*norm[1])
              tmpP = Point(new_p.x + t*norm[0], new_p.y + t*norm[1])

              #the point chosen by mahalanobis
              #tmpP = max_pt


              #cv.Circle(imgtmp, ( int(tmpP.x), int(tmpP.y) ), 3, (100,100,100))
              #cv.NamedWindow("imgtmpMaxPoint", cv.CV_WINDOW_AUTOSIZE)
              #cv.ShowImage("imgtmpMaxPoint",imgtmp)

              #c=cv.WaitKey(10)
              #if c==1048603 :#whichever integer key code makes the app exit
              #  exit()



          ############################################################################################
          ###########OLD WAY OF CHOOSING BASED ON THE ASSUMPTION THE WE ARE ON THE MAX EDGE###########
          #along the whisker
          '''for side in range(-3,4):#along normal profile'''
          # Normal to normal...
          ###print"norm",norm
          new_p = Point(p.x + side*norm[0], p.y + side*norm[1])#why ..the other way around?
          ###print"p",(p.x,p.y)
          ##print"new_p",(new_p)
          ###cv.WaitKey()

          iterationcounter=0

          imgtmp = cv.CreateImage(cv.GetSize(self.image), self.g_image[scale].depth, 1)
          cv.Copy(self.g_image[scale], imgtmp)

          # Look 6 pixels to each side too
          for t in drange(-6, 7, 1):#tangent
              ###print"t...........",t

              x = int(norm[0]*t + new_p.x)
              y = int(norm[1]*t + new_p.y)


              if x < 0 or x > self.image.width or y < 0 or y > self.image.height:
                continue

              #show min and max points
              #cv.Circle(imgtmp, ( int(norm[0]*min_t + new_p.x) , int(norm[1]*min_t + new_p.y)), 10, (100,100,100))
              #cv.Circle(imgtmp, ( int(norm[0]*max_t + new_p.x) , int(norm[1]*max_t + new_p.y)), 10, (100,100,100))

              ###printx, y, self.greyImage.width, self.greyImage.height

              ###print"greyImage[scale][y, x]",self.greyImage[scale][y,x]

              targetImageTocheckAgainst=self.g_image[scale] #self.image
              if targetImageTocheckAgainst[y-1, x-1] > max_edge:
                max_edge = targetImageTocheckAgainst[y-1, x-1] #greyImage[scale][y, x]
                max_pt = Point(new_p.x + t*norm[0], new_p.y + t*norm[1])
                ##print'update max point'

                ##print"p=%s"%(p)
                ##print"max_pt=%s"%(max_pt)
                ##print"x=%s , y=%s"%(x,y)
                iterationcounter+=1


                #show max point updated position
                cv.Circle(imgtmp, ( int(max_pt.x), int(max_pt.y) ), 5, (100,100,100))
                cv.NamedWindow("imgtmpMaxPoint", cv.CV_WINDOW_AUTOSIZE)
                #cv.ShowImage("imgtmpMaxPoint",imgtmp)
                #print 'max point shown'
                ##cv.WaitKey()

              ##print"iterationcounter=%d"%(iterationcounter)
              tmpP = Point(new_p.x + t*norm[0], new_p.y + t*norm[1])
              #cv.Circle(imgtmp, ( int(tmpP.x), int(tmpP.y) ), 3, (100,100,100))
              cv.NamedWindow("imgtmpMaxPoint", cv.CV_WINDOW_AUTOSIZE)
              cv.ShowImage("imgtmpMaxPoint",imgtmp)
              c=cv.WaitKey()
              if c==1048603 :#whichever integer key code makes the app exit
                  exit()

          cv.Circle(imgtmp, ( int(max_pt.x), int(max_pt.y) ), 5, (100,100,100))
          cv.NamedWindow("imgtmpMaxPoint", cv.CV_WINDOW_AUTOSIZE)
          cv.ShowImage("imgtmpMaxPoint",imgtmp)
          #print 'max point for this landmark'
          #cv.WaitKey(1)
          ###########OLD WAY OF CHOOSING BASED ON THE ASSUMPTION THE WE ARE ON THE MAX EDGE###########
          ############################################################################################







          if len(correctionVectorForAllShapes)<len(currentShape.pts):#write the fist 30 landmarks
              correctionVectorForAllShapes.append(max_pt)
          else:#overwrite all the othes and at the end divide by the total number of shapes
              correctionVectorForAllShapes[p_num]+=max_pt

      f.close()
      print "len of correctionVectorForAllShapes=%d"%(len(correctionVectorForAllShapes))
      #cv.WaitKey()
      return correctionVectorForAllShapes


  '''
  #print "len(correctionVectorForAllShapes)=%d"%len(correctionVectorForAllShapes)
  #print "(correctionVectorForAllShapes)=%s"%(correctionVectorForAllShapes)
  ###cv.WaitKey()

for i in range(len(correctionVectorForAllShapes)):
  #print "(correctionVectorForAllShapes[%d]) was %s"%(i,correctionVectorForAllShapes[i])
  correctionVectorForAllShapes[i]=correctionVectorForAllShapes[i]/len(AllImages_NormalizedtmpLandmarkDerivativeIntensityVector)
  #print "(correctionVectorForAllShapes[%d]) AVERAGED is %s"%(i,correctionVectorForAllShapes[i])

#print "len(correctionVectorForAllShapes)=%d"%len(correctionVectorForAllShapes)
return correctionVectorForAllShapes
  ##########################################################################################
  ##########################################################################################
  '''



  def do_iteration(self, scale, i, GTrainingMeansCalculated):
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

    #print 'Iteration started'


    #for each of the points of the current shape
    #for i, pt in enumerate(self.shape.pts):

    #pass in the mean shape initially in the first iteration
    self.calcMaxPtFromMahalanobisModelProfile(self.shape,scale,i, GTrainingMeansCalculated)



  def calcMaxPtFromMahalanobisModelProfile(self, initmeanshape, scale, i ,GTrainingMeansCalculated):

    #create a new image based on the input image's 1st resolution (without any scaling whatsoever)
    img = cv.CreateImage(cv.GetSize(self.image), self.g_image[scale].depth, 1)
    cv.Copy(self.g_image[scale], img)

    #create a shape to be filled with the new correction vector (after mahalanobis measurement for this image)
    s = Shape([])

    #calculate G mean for all landmakrs just once
    if GTrainingMeansCalculated == 0:

        #prevent from re-calculating Training G means (cause they should probably be based on the first mean shape)
        GTrainingMeansCalculated = 1

        #Transposed matrix to align with the iteration from bottom(tangent) left(whisker)..to..up(tangent right(whisker))
        self.gaussianMatrix=np.transpose(self.gaussianMatrix)

        ##print"gaussianMatrix=%s"%(gaussianMatrix)

        #iterate the gaussian distribution matrix
        for (x,y), value in np.ndenumerate(self.gaussianMatrix):
                    print"%d, %d =%s"%(x,y,self.gaussianMatrix[x][y])
                    ####cv.WaitKey()


        self.trainingGmean = self.findtrainingGmean(self.testImageNameCounter, self.AllImages_NormalizedtmpLandmarkDerivativeIntensityVector, self.gaussianMatrix)
        print "len(trainingGmean)=%s"%(len(self.trainingGmean))
        #cv.WaitKey()



    #calculate mahalanobis distance, to derive the correction vector
    max_ptlist=self.getCorrectedLandmarkPoint(self.trainingGmean)
    #print 'max_ptlist=%s'%(max_ptlist[0])
    #cv.WaitKey()

    f = open('maxptList/max_ptlist'+str(i),'w')
    f.write(str(max_ptlist[0]))
    f.close()

    #for each of the points of the current shape
    for i, pt in enumerate(initmeanshape.pts):


        ##print "maxpoint=%s"%(max_pt)
        s.add_point(max_ptlist[i])

        #show max points chosen along normal profile sampled -6..-6 pixels across
        #BEFORE ALIGNMENT
        ##print "\BEFORE ALIGNMENT\n"
        maxpX=(int)(pt.x)
        maxpY=(int)(pt.y)
        cv.Circle(img, (maxpX,maxpY), 1, (255,255,255))
        cv.NamedWindow("Scale", cv.CV_WINDOW_AUTOSIZE)
        cv.ShowImage("Scale",img)


    ##cv.WaitKey()
    ##print 'Points Added'

    #align this s shape to mean with a weigted matrix
    new_s = s.align_to_shape(Shape.from_vector(self.asm.mean), self.asm.w)

    ##print (self.asm.evals[0])

    #calculate new shape - update the mean based on x=(mean) + P*b
    var = new_s.get_vector() - self.asm.mean
    new = self.asm.mean
    for i in range(len(self.asm.evecs.T)):
      b = np.dot(self.asm.evecs[:,i],var)

      #ADDITION of this IF statement CAUSE IT CRASHS IF  self.asm.evals[i] is < 0
      if self.asm.evals[i] > 0:
          max_b = 3*math.sqrt(self.asm.evals[i])
          b = max(min(b, max_b), -max_b)
          new = new + self.asm.evecs[:,i]*b

    #print "previous shape=%s"%(self.shape.pts[0])
    ##cv.WaitKey()

    #align the new shape to the already existing (aligned to the mean) s
    self.shape = Shape.from_vector(new).align_to_shape(s, self.asm.w)

    #print "current shape=%s"%(self.shape.pts[0])
    ##cv.WaitKey()

    '''
    #show max points chosen along normal profile sampled -6..-6 pixels across
    #AFTER ALIGNMENT
    ##print "\nAFTER ALIGNMENT\n"
    for i, pt in enumerate(self.shape.pts):
        maxpX=(int)(max_pt.x)
        maxpY=(int)(max_pt.y)
        #cv.Circle(img, (maxpX,maxpY), 2, (255,255,255))
        cv.NamedWindow("Scale", cv.CV_WINDOW_AUTOSIZE)
        cv.ShowImage("Scale",img)
        ###cv.WaitKey()
    '''


    ###cv.WaitKey()



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
    ##print "Calculating weight matrix..."
    self.w = self.__create_weight_matrix(shapes)

    ##print 'Shapes BEFORE Weighted Procrustes'
    ##print self.shapes[0].pts
    ##print "\n"
    ##print self.shapes[1].pts
    ##print "\n"
    ##print self.shapes[2].pts
    ##print "\n"

    # Align all shapes
    ##print "Aligning shapes with Procrustes analysis..."
    self.shapes = self.__procrustes(shapes)

    ##print 'Shapes AFTER Weighted Procrustes'
    ##print self.shapes[0].pts
    ##print "\n"
    ##print self.shapes[1].pts
    ##print "\n"
    ##print self.shapes[2].pts
    ##print "\n"





    ##print "Constructing model..."
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
    ##print ("3 training images tested = %s")%(len(shape_vectors))
    ##print ("3 training images tested = %s")%((shape_vectors))


    ##print ("each image contains %s landmark points")%(len(s.get_vector())/2)#61 sets of coordinates [x,y]
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


    print "shape_vectors\n",shape_vectors
    c=cv.WaitKey(1)
    ###print"EXIT when ESC keycode is pressed=%d"%(c)
    if c == 1048603:
        exit()

    #the mean of the aligned shapes
    mean = np.mean(shape_vectors, axis=0)

    ##print "mean shape which is the center of the ellipsoidal Allowable Shape Domain - Before reshaping\n",mean
    ##print "\n"

    # Move mean to the origin
    # FIXME Clean this up...
    mean = np.reshape(mean, (-1,2))#turn mean array from 4x5 to 10x2 array
    min_x = min(mean[:,0])#get the min of the 1st row refering to the first X coordinate
    min_y = min(mean[:,1])#get the min of the 2nd row refering to the Second Y coordinate

    print "mean After reshaping\n",mean
    c=cv.WaitKey(1)
    ###print"EXIT when ESC keycode is pressed=%d"%(c)
    if c == 1048603:
        exit()


    #mean = np.array([pt - min(mean[:,i]) for i in [0,1] for pt in mean[:,i]])
    #mean = np.array([pt - min(mean[:,i]) for pt in mean for i in [0,1]])


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
    ###print mean


    ##print "The list of shapes (and their point coordinates:\n",shape_vectors
    ##print "\n\n"



    # Produce covariance matrix

    '''We attempt to model the shape in the Allowable Shape Domain..hence capture the relationships between position of individual landmark points'''
    '''The cloud of landmarks is approximately ellipsoidal, so we need to calculate its center (giving a mean shape and its major axes)'''
    '''Covariance indicates the level to which two variables vary together'''

    #shape_vectors=[(1,2),(3,4)]
    cov = np.cov(shape_vectors, rowvar=0)
    print "shape_vectors",shape_vectors
    #cv.WaitKey()
    #print "cov\n",cov
    #cv.WaitKey()

    # Find eigenvalues/vectors of the covariance matrix
    evals, evecs = np.linalg.eig(cov)

    ##print "evals\n",evals
    ##print "sum(evals)\n",sum(evals)


    ##print "evecs\n",evecs



    #=0
    # Find number of modes required to describe the shape accurately
    t = 0
    for i in range(len(evals)):

          ##print "sum(evals[:%f])=%f\n"%(sum(evals[:i]),sum(evals[:i]))
          ###print "sum(evals[:%f])=%f\n"%(sum(evals),sum(evals))

          #iterating through the list of evals, as soon as the sum  of evals so far divided by the total sum is >=0.99 then this is the number of modes we need
          '''Choose the first largest eigenvalues..that represent a wanted percentage of the total variance, a.i. 0.99 or 99%.
          Defines the proportion of the total variation one wishes to explain
          (for instance, 0.98 for 98%)'''

          if sum(evals[:i]) / sum(evals)< 0.99:
                #=c+1
                ###print c
                t = t + 1
          else: break


    print "Constructed model with %d modes of variation" % t
    #cv.WaitKey()

    #evecs = np.array([[1,2,3,4,5,6],[7,8,9,10,11,12]])
    ###print evecs[:,:1]

    ##print "evals[:%d]\n%s"%(t,evals[:t])

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

    ###print "TRANS ", trans

    converged = False
    current_accuracy = sys.maxint

    ml=[]
    lines_counter_written=1#contol flag
    lines_counter=0

    while not converged:
      # Now get mean shape
      mean = self.__get_mean_shape(shapes)

      ###print "Calculated mean:",mean.pts

      # Align to shape to stop it diverging
      mean = mean.align_to_shape(a, self.w)#NORMALIZATION

      ###print "new mean=",mean.pts

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
      if accuracy > current_accuracy: converged = True;  ###print 'accuracy=',(accuracy);     ##print 'current_accuracy=',(current_accuracy);
      else: current_accuracy = accuracy; # ##print 'accuracy=',(accuracy);     ##print 'current_accuracy=',(current_accuracy);

    #target.close()
    ##print "Final Mean Shape Points=",mean.pts
    ##print "\n"

    ###print "ACCURACY=%d, current_accuracy=%d ,  converged=%s"%(accuracy,current_accuracy, converged)
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

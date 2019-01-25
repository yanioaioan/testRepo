#!/usr/bin/env python

import sys
import os
import cv
import glob
import math
import numpy as np
from random import randint
import scipy.spatial
import cv2
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from Sampling import *
import os.path

#CHOOSE A WORKING SETUP
WORKING_SETUP=0
skipToNextLevel=False

#SLOW MANUAL PER PIXEL PROCESSING
ENHANCE_IMAGE=True#increase contrast & make vertebras differer from background

#WRITE_OUT SEGMENTED RESULT AS SET OF POINTS under sys.argv[1]/edgeSegmentedPoints
'''
This flag is supposed to be '1'/ON when we want to write out the segmented points as pts after we have probably segmented with WORKING_SETUP=1 !
Next, we can turn it off and run by pointing to "sys.argv[1]/edgeSegmentedPoints"
'''
WRITE_OUT_SEGMENTED_RESULT_AS_SET_OF_POINTS=0

TOTALLEVELS=4#only 3 used currently

DEBUG_LINES=0
STOP_AT_THE_END_OF_EVERY_LANDMARK=0
SHOW_EVERY_POSSIBLE_POINT=0
SHOW_EVERY_POSSIBLE_STATISTICAL_SAMPLE=0

USE_1D_PROFILE=1
USE_2D_PROFILE=0

USE_MAX_EDGE=0#max edge or profile model


#fluoroTest --> STEP=10 and PIXELS_SEARCH=3 # path is under images/fluoroTest
#round shape --> STEP=7 and PIXELS_SEARCH=2
#frank --> STEP=6 and PIXELS_SEARCH=2 or4
#hands --> STEP=6 and PIXELS_SEARCH=2 or4

class Dataset:
    fluoroTest=0,#fluoroTest
    round= 1,#round shape
    Frank= 2,#Frank
    hands= 3,#hands
    Cootes= 4,#Cootes
    incisor=5,#fluoroTest
    HD_FluroVertebra=6

#CHOOSE which training set to work with!!!

#7 step..3 pixel_search overall good
#11 step..9 pixel_search good for fluoro somehow
#datasetChosen = Dataset.fluoroTest #terminal COMMAND: --> python segmentImage.py everything/images/fluoroTEST/ everything/images/fluoroTEST/grey_image_8.png
#datasetChosen = Dataset.round #terminal COMMAND: --> python segmentImage.py everything/points/S_plus_Round_Shape_works/Round_Shape everything/images/S_plus_Round_Shape_works/Round_Shape/grey_image_1.jpg
#datasetChosen = Dataset.Frank #terminal COMMAND: --> python segmentImage.py /home/yioannidis/Desktop/PHD/myphd/PHD/Algorithm_Block_Out/pythonStuffPHD/ASMWeightedMatrixTest/test/WORKING_TEST_RECTANGLE/working_fluoroSpine/FRANK_TEST/points/properlyname30/ /home/yioannidis/Desktop/PHD/myphd/PHD/Algorithm_Block_Out/pythonStuffPHD/ASMWeightedMatrixTest/test/WORKING_TEST_RECTANGLE/working_fluoroSpine/FRANK_TEST/images/properlynamed30images/grey_image_15.jpg
#datasetChosen = Dataset.Cootes #terminal COMMAND: -->
#datasetChosen = Dataset.hands #terminal COMMAND: -->python segmentImage.py everything/otherTests/testHandsBones/pts/ everything/otherTests/testHandsBones/images/grey_image_1.png
#datasetChosen = Dataset.incisor #terminal COMMAND: -->python segmentImage.py everything/points/X-RaysTeeth/_Data/Landmarks/original/1st_Incisor/ everything/images/X-RaysTeeth/_Data/Radiographs/grey_image_2.jpg
datasetChosen = Dataset.HD_FluroVertebra #terminal COMMAND: -->python segmentImage.py /home/yioannidis/Desktop/PHD/myphd/PHD/Algorithm_Block_Out/pythonStuffPHD/ASMWeightedMatrixTest/test/WORKING_TEST_RECTANGLE/DESKTOP_LAST_WORKINGTESTS_TEST_STATISTICAL_SHAPE_MODEL_FIT/TEST_STATISTICAL_SHAPE_MODEL_FIT/everything/images/HD-Fluoro/08AlBl_Flex/pts /home/yioannidis/Desktop/PHD/myphd/PHD/Algorithm_Block_Out/pythonStuffPHD/ASMWeightedMatrixTest/test/WORKING_TEST_RECTANGLE/DESKTOP_LAST_WORKINGTESTS_TEST_STATISTICAL_SHAPE_MODEL_FIT/TEST_STATISTICAL_SHAPE_MODEL_FIT/everything/images/HD-Fluoro/08AlBl_Flex/images/10only/grey_image_5.jpg




#How many extra pixels in each direction
STEP=3# NEEDS TO BE 9 when working with 10 frames sequence (10only_v1_half_res).   ----   sample every "STEP" pixels along the whisker # was 10 WORKS FOR FRANK-->6 works for fluoroTest

#Normal Profile range on each side
PIXELS_SEARCH=6# NEEDS TO BE 5 when working with 10 frames sequence (10only_v1_half_res).   ----    #COULD BE 2 # was 4    #6 work with   __createStatisticalProfileModel() & and self.__get_MAHALANOBIS()
#NOTE: Stasm used 4 for PIXELS_SEARCH on each side & 2 for STEP

#use the sampling class & not the old   __createStatisticalProfileModel & __get_MAHALANOBIS (they are different as the new ones under Sampling.py are performing STASM equivalent sampling on smoothed only images not sobeled)
SAMPLING_PY_ENABLED=1

#When this flag is ON it only smoothes & doesn't take the 1st derivative of the image-->as we are going to calculate it ourselves
JUST_SMOOTH_NO_SOBEL=1

#When this flag is ON it uses pyramid cv2 smoothing instead of the __produce_gradient_image function (old style smoothing)
USE_PYRAMID_SMOOTHING=1


#CHOOSE A WORKING SETUP, works with max edge and proper initialization
if WORKING_SETUP==1:
    TOTALLEVELS=4#only 3 used currently

    DEBUG_LINES=0
    STOP_AT_THE_END_OF_EVERY_LANDMARK=0
    SHOW_EVERY_POSSIBLE_POINT=0
    SHOW_EVERY_POSSIBLE_STATISTICAL_SAMPLE=0

    USE_1D_PROFILE=1
    USE_2D_PROFILE=0

    USE_MAX_EDGE=1#max edge or profile model

    #How many extra pixels in each direction
    STEP=2#sample every "STEP" pixels along the whisker

    #Normal Profile range on each side
    PIXELS_SEARCH=6 #6 work with   __createStatisticalProfileModel() & and self.__get_MAHALANOBIS()
    #NOTE: Stasm used 4 for PIXELS_SEARCH on each side & 2 for STEP

    #use the sampling class & not the old   __createStatisticalProfileModel & __get_MAHALANOBIS
    SAMPLING_PY_ENABLED=0

    #When this flag is ON it only smoothes & doesn't take the 1st derivative of the image-->as we are going to calculate it ourselves
    JUST_SMOOTH_NO_SOBEL=0

    #When this flag is ON it uses pyramid cv2 smoothing instead of the __produce_gradient_image function (old style smoothing)
    USE_PYRAMID_SMOOTHING=0



#gaussianMatrix=np.array([
#[0.000158,	0.000608,        0.00183,	0.004312,	0.007953,	0.011483,	0.012978,	0.011483,	0.007953,	0.004312,	0.00183,	0.000608,	0.000158],
#[0.000291,	0.001121,	0.003375,	0.007953,	0.014669,	0.021179,	0.023938,	0.021179,	0.014669,	0.007953,	0.003375,	0.001121,	0.000291],
#[0.000421,	0.001618,	0.004873,	0.011483,	0.021179,	0.030579,	0.034561,	0.030579,	0.021179,	0.011483,	0.004873,	0.001618,	0.000421],

#[0.000476,	0.001829,	0.005508,	0.012978,	0.023938,	0.034561,	0.039062,	0.034561,	0.023938,	0.012978,	0.005508,	0.001829,	0.000476],

#[0.000421,	0.001618,	0.004873,	0.011483,	0.021179,	0.030579,	0.034561,	0.030579,	0.021179,	0.011483,	0.004873,	0.001618,	0.000421],
#[0.000291,	0.001121,	0.003375,	0.007953,	0.014669,	0.021179,	0.023938,	0.021179,	0.014669,	0.007953,	0.003375,	0.001121,	0.000291],
#[0.000158,	0.000608,	0.00183,	0.004312,	0.007953,	0.011483,	0.012978,	0.011483,	0.007953,	0.004312,	0.00183,	0.000608,	0.000158]
#                                                            ])

#gaussianMatrix = np.array([0.000476,	0.001829,	0.005508,	0.012978,	0.023938,	0.034561,	0.039062,	0.034561,	0.023938,	0.012978,	0.005508,	0.001829,	0.000476])
#gaussianMatrix = np.array([0.006, 0.61, 0.242, 0.383, 0.242, 0.61, 0.006])
gaussianMatrix = np.array([0.002406  ,0.009255,      0.027867,       0.065666        ,0.121117,      0.174868,       0.197641        ,0.174868       ,0.121117       ,0.065666       ,0.027867       ,0.009255       ,0.002406])


#the length of the following 2 lists is equal to the number of annotated landmarks for each image (but as we have multiple gaussian pyramid levels)
TrainingMean=[]
TrainingCovarianceMatrices=[]
giAllPointsShapeVecSet = []

def myprint(arg):
    if DEBUG_LINES==1:
        print "arg=%s"%(arg)



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

  def __div__(self, i):#division by a constant
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

  def __abs__(self):
    """ Return a new point which is the absolute of p
    :param p: The other point
    """
    return Point( abs(self.x), abs(self.y) )

  '''
  def __idiv__(self, p):#division by another point
    """ Return a new point which is the division of bothe x and y correspondingly with p
    :param p: The other point
    """
    return Point( self.x/float(p.x), self.y/float(p.y) )
  '''

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
  '''
  def add_point(self, p):
    self.pts.append(p)
    self.num_pts += 1


  def transform(self, t):
    s = Shape([])
    for p in self.pts:
      s.add_point(p + t)
    return s
  '''


  #https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
  def add_point(self, p, degreesOfRotation=0, CenterOfMass=Point(0,0)):
      xx=p.x
      yy=p.y
      if CenterOfMass.x != 0 and CenterOfMass.y != 0 and degreesOfRotation!=0:
          #degreesOfRotation=-15
          angle=degreesOfRotation*(math.pi/180)

          """
          Rotate a point counterclockwise by a given angle around a given origin.

          The angle should be given in radians.
          """
          ox, oy = CenterOfMass.x, CenterOfMass.y
          print "CenterOfMass=",CenterOfMass

          xx  = ox + math.cos(angle) * (p.x - ox) - math.sin(angle) * (p.y - oy)
          yy  = oy + math.sin(angle) * (p.x - ox) + math.cos(angle) * (p.y - oy)


      self.pts.append(Point(xx,yy))
      self.num_pts += 1


  #not working properly ..investigate when you have free time..so never!!
  def add_point_prev(self, p, degreesOfRotation=0, CenterOfMass=Point(0,0)):


    if CenterOfMass.x != 0 and CenterOfMass.y != 0 and degreesOfRotation!=0:
        #degreesOfRotation=-15
        f=degreesOfRotation*(math.pi/180)

        '''
        p.x = p.x - CenterOfMass.x
        p.y = p.y - CenterOfMass.y

        p.x = p.x * math.cos(f) - p.y *math.sin(f)
        p.y = p.y * math.sin(f) + p.x *math.cos(f)

        p.x = p.x + CenterOfMass.x
        p.y = p.y + CenterOfMass.y
        '''
        print "CenterOfMass=",CenterOfMass

        p.x = CenterOfMass.x + ((p.x-CenterOfMass.x)*math.cos(f)-(p.y-CenterOfMass.y)*math.sin(f))
        p.y = CenterOfMass.y + ((p.x-CenterOfMass.x)*math.sin(f)+(p.y-CenterOfMass.y)*math.cos(f))



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
      #original
      '''
      x = self.pts[1].x - self.pts[0].x
      y = self.pts[1].y - self.pts[0].y
      '''
      #landmark position dependent normal calculation

      x = self.pts[1].x - self.pts[-1].x
      y = self.pts[1].y - self.pts[-1].y

    # Normal to last point
    elif p_num == len(self.pts)-1:
      #original
      '''
      x = self.pts[-1].x - self.pts[-2].x
      y = self.pts[-1].y - self.pts[-2].y
      '''

      #landmark position dependent normal calculation

      x = self.pts[0].x - self.pts[-2].x
      y = self.pts[0].y - self.pts[-2].y

    # Must have two adjacent points, so...
    else:
      x = self.pts[p_num+1].x - self.pts[p_num-1].x
      y = self.pts[p_num+1].y - self.pts[p_num-1].y
    mag = math.sqrt(x**2 + y**2)

    #HACKED TO AVOID DIVISION BY zero when /norm[0] or when /norm[1]

    if y==0:
        y=1
    if x==0:
        x=1

    if mag!=0:
      return (-y/mag, x/mag)
    else:
        mag=1
        return (-y/mag, x/mag)

  @staticmethod
  def from_vector(vec):
    s = Shape([])
    for i,j in np.reshape(vec, (-1,2)):
      s.add_point(Point(i, j))
    return s



#drawing from mean shape
def drawMeanShape(meansvec, image):
    meanshape = Shape.from_vector(meansvec)
    for pt_num, pt in enumerate(meanshape.pts):

        nextPoint=-1
        prevPoint=-1

        # Draw normals

        cv.Circle(image, (int(round(pt.x)), int(round(pt.y))), 1, (0,0,255), -1)
        nextPoint = (int(round(pt.x)), int(round(pt.y)))
        '''
        if prevPoint != -1:#draw line connecting every other point with the next one , except for the last to the first segment
            cv.Line(i, prevPoint , nextPoint ,(0,0,255),1)

        else:#draw line segment connecting the last to the first landmark
            prevPoint = (int(round(f.shape.pts[-1].x)), int(round(f.shape.pts[-1].y)))
            cv.Line(i, prevPoint , nextPoint ,(0,0,255),1)
        '''

        prevPoint = nextPoint
        #cv2.imshow("image", image)
        #print "drawing mean point..%d..%s"%(pt_num,pt)
        #cv2.waitKey(100)


#ManualTempateMatching

dx=0
dy=0
degreesOfRotation=0
new_degrees=0

meanshapeVec = Shape([])
finalPlacedShape = Shape([])
initVecMean=[]

def recalculateTransform(new_dx, new_dy, new_degrees, direction, meanvec):

    #keep reseting the image and avoid having multiple shape instances drawn on it

    global initVecMean
    if  len(initVecMean)==0:
        initVecMean = meanvec[:]


    print 'Transforming %s..'%(direction)

    #Transformation recalculation
    meanshapeVec=Shape.from_vector(meanvec)

    sumx=0
    sumy=0
    for i, pt in enumerate(meanshapeVec.pts):
        sumx+=pt.x
        sumy+=pt.y
    sumx/=len(meanshapeVec.pts)
    sumy/=len(meanshapeVec.pts)

    from testInitASM import Point
    CenterOfMass = Point(sumx,sumy)
    global degreesOfRotation
    degreesOfRotation=new_degrees#-15

    #CenterOfMass=Point(0,0)
    global dx
    global dy

    dx=new_dx
    dy=new_dy
    print dx,dy
    t=Point(dx,dy)
    if CenterOfMass.__eq__(Point(0,0)):
        t= Point(0,0)

    global finalPlacedShape
    finalPlacedShape = Shape.from_vector(initVecMean).transform(t, degreesOfRotation, CenterOfMass)
    print "meanshapeVec.pts=",meanshapeVec.pts

    #update meanvec after the transformation
    print "before",meanvec
    cv.WaitKey()
    meanvecNew = finalPlacedShape.get_vector()
    print "meanvecNew=",meanvecNew
    cv.WaitKey()
    return meanvecNew


l_r=0
u_d=0
rot=0
meanshapeVectorNew=0




class ShapeViewer ( object ):
  """ Provides functionality to display a shape in a window
  """
  @staticmethod
  def show_shapes(shapes):
    """ Function to show all of the shapes which are passed to it
    """
    cv.NamedWindow("Shape Model",  cv.CV_WINDOW_NORMAL)#cv.CV_WINDOW_AUTOSIZE
    # Get size for the window
    max_x = int(max([pt.x for shape in shapes for pt in shape.pts]))
    max_y = int(max([pt.y for shape in shapes for pt in shape.pts]))
    min_x = int(min([pt.x for shape in shapes for pt in shape.pts]))
    min_y = int(min([pt.y for shape in shapes for pt in shape.pts]))

    i = cv.CreateImage((max_x-min_x+20, max_y-min_y+20), cv.IPL_DEPTH_8U, 3)
    cv.Set(i, (0, 0, 0))
    for shape in shapes:
      r = randint(0, 255)
      g = randint(0, 255)
      b = randint(0, 255)
      #r = 0
      #g = 0
      #b = 0

      nextPoint=-1
      prevPoint=-1

      for pt_num, pt in enumerate(shape.pts):
        # Draw normals
        #norm = shape.get_normal_to_point(pt_num)
        #cv.Line(i,(pt.x-min_x,pt.y-min_y), \
        #    (norm[0]*10 + pt.x-min_x, norm[1]*10 + pt.y-min_y), (r, g, b))



        #cv.Circle(i, (int(pt.x-min_x), int(pt.y-min_y)), 2, (r, g, b), -1)
        nextPoint = (int(pt.x-min_x), int(pt.y-min_y))
        if prevPoint != -1:
            cv.Line(i, prevPoint , nextPoint ,(0,255,255),1)

        prevPoint = nextPoint

    cv.ShowImage("Shape Model",i)

  @staticmethod
  def show_modes_of_variation(model, mode):
    # Get the limits of the animation
    start = -2*math.sqrt(model.evals[mode])
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
      c = cv.WaitKey(10)
      if chr(255&c) == 'q': break



  @staticmethod
  def draw_model_fitter(f,scale):
    cv.NamedWindow("Model Fitter", cv.CV_WINDOW_NORMAL)
    # Copy image

    i = cv.CreateImage(cv.GetSize(f.target), f.target.depth, 3)

    #Shows original target
    cv.Copy(f.target, i)

    ##Shows derivatives of target image
    #convertedToColouredImg = cv.CreateImage(cv.GetSize( f.g_target_image[scale]), cv.IPL_DEPTH_8U, 3)
    #cv.CvtColor(f.g_target_image[scale], convertedToColouredImg, cv.CV_GRAY2BGR)
    #cv.Copy(convertedToColouredImg, i)



    nextPoint=-1
    prevPoint=-1

    for pt_num, pt in enumerate(f.shape.pts):
      # Draw normals

      cv.Circle(i, (int(pt.x), int(pt.y)), 1, (0,0,255), -1)#Un Hash to have points
      nextPoint = (int(pt.x), int(pt.y))

      if prevPoint != -1:#draw line connecting every other point with the next one , except for the last to the first segment
          cv.Line(i, prevPoint , nextPoint ,(0,0,255),1)

      else:#draw line segment connecting the last to the first landmark
          prevPoint = (int(f.shape.pts[-1].x), int(f.shape.pts[-1].y))                    
          cv.Line(i, prevPoint , nextPoint ,(0,0,255),1)


      prevPoint = nextPoint
      cv.NamedWindow("Model Fitter", cv.CV_WINDOW_NORMAL)
      cv.ShowImage("Model Fitter",i)


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
        num_pts = int(first_line)
      for line in fh:
        if not line.startswith("}"):
          pt = line.strip().split()
          print pt
          s.add_point(Point(float(pt[0]), float(pt[1])))
    if s.num_pts != num_pts:
      #print "Unexpected number of points in file.  "\
      "Expecting %d, got %d" % (num_pts, s.num_pts)
    return s

  @staticmethod
  def read_directory(dirname):
    """ Reads an entire directory of .pts files and returns
    them as a list of shapes
    """
    pts = []
    for file in glob.glob(os.path.join(dirname, "*.pts")):
      pts.append(PointsReader.read_points_file(file))
    return pts

class FitModel:
  """
  Class to fit a model to an image

  :param asm: A trained active shape model
  :param image: An OpenCV image
  :param t: A transformation to move the shape to a new origin
  """

  pointsWithin50PerentOfProfile = []

  #S Shape Works 140,45.0
  #round shape 140,45.0
  #fluoroTest 230,145.0
  #girl face at images/faces  180.0,320
  # Frank 270,200
  #Simple rectangle 252,110.0 (grey_image_2.jpg)
  #Simple rectangle 256,105.0 (grey_image_3.jpg)

  #Create a Sampling instance
  sampling = Sampling( TOTALLEVELS,#only 3 used currently
                      DEBUG_LINES,
                      STOP_AT_THE_END_OF_EVERY_LANDMARK,
                      SHOW_EVERY_POSSIBLE_POINT,
                      SHOW_EVERY_POSSIBLE_STATISTICAL_SAMPLE,
                      USE_1D_PROFILE,
                      USE_2D_PROFILE,
                      USE_MAX_EDGE,
                      STEP,#sample every "STEP" pixels along the whisker
                      PIXELS_SEARCH, #6+1
                      JUST_SMOOTH_NO_SOBEL)

  def pyramid(self, image, level, imagenumber=0, totaleLevel=0):

      print "imagenumber=%s"%(imagenumber)
      print "totaleLevel=%s"%(totaleLevel)
      img = cv2.imread(image,cv2.IMREAD_GRAYSCALE)
      h,w = img.shape[:2]
      img = cv2.pyrDown( img,dstsize = (w/2, h/2) )
      h,w = img.shape[:2]
      img = cv2.pyrUp(img,dstsize =  (2*w, 2*h) )
      cv2.imwrite("testPyr_%s.jpg"%(level),img)
      testPyr = cv.LoadImage("testPyr_%s.jpg"%(level))
      #cv.NamedWindow("testPyr_%s"%(level), cv.CV_WINDOW_NORMAL)
      #cv.ShowImage("testPyr_%s"%(level),testPyr)
      #cv.WaitKey()
      return testPyr


  def pythonicSwitch(x):
      return {
          Dataset.fluoroTest: Point(240,175),#(10,75), #fluoroTest
          Dataset.round: Point(145,50.0),#round shape
          Dataset.Frank: Point(280,210),#Frank
          Dataset.hands: Point(55,70),#hands
          Dataset.Cootes: Point(280,100),#Cootes
          Dataset.incisor: Point(1350,800.0),#round shape
          Dataset.HD_FluroVertebra: Point(400,200.0),#round shape
      }.get(x, (145,50.0))    # (145,50.0)..so the 'round shape' target coordinates is default that's going to be used if x not found


  initPoint=pythonicSwitch(datasetChosen)

  #def __init__(self, asm, images, imageNames, target,   originalShapes, t=initPoint):#fluoroTest
  def __init__(self, asm, images, imageNames, target,   originalShapes, t=initPoint):#round shape
  #def __init__(self, asm, images, imageNames, target,   originalShapes, t=initPoint):#Frank
  #def __init__(self, asm, images, imageNames, target,   originalShapes, t=initPoint):#Cootes
  #def __init__(self, asm, images, imageNames, target,   originalShapes, t=initPoint):#hands

  #def __init__(self, asm, image, t=Point(140,45.0)):
    self.images = images
    self.g_images = []


    #save all landmark points
    self.originalShapes = originalShapes
    #for shape in (asm.shapes):#for each shape
    #	self.shapepoints = shape.pts


    '''
    self.pyramid(sys.argv[2],1)
    self.pyramid("testPyr_%s.jpg"%(1),2)
    self.pyramid("testPyr_%s.jpg"%(2),3)
    sobelTestPyr = cv.LoadImage("testPyr_%s.jpg"%(3),3)

    scale=2
    result_sobelTestPyr=self.__produce_gradient_image(sobelTestPyr, 2**scale)

    cv.NamedWindow("result_sobelTestPyr_level%s"%(scale), cv.CV_WINDOW_NORMAL)
    cv.ShowImage("result_sobelTestPyr_level%s"%(scale),result_sobelTestPyr)
    cv.WaitKey()

    scale=0
    result_sobelTestPyr=self.__produce_gradient_image(sobelTestPyr, 2**scale)

    cv.NamedWindow("result_sobelTestPyr_level%s"%(scale), cv.CV_WINDOW_NORMAL)
    cv.ShowImage("result_sobelTestPyr_level%s"%(scale),result_sobelTestPyr)
    cv.WaitKey()
    '''



    self.target = target
    self.g_target_image = []

    self.convertedToColouredImg = []

    for i in range(0,TOTALLEVELS):#for 4 different levels of sobel derivative smoothing
      if not USE_PYRAMID_SMOOTHING:
        self.g_target_image.append(self.__produce_gradient_image(target, 2**i))
      else:

        if i==0:
          #sys.argv[2] is the target image name
          currentSmoothedLevelTarget=self.pyramid(sys.argv[2],i+1)
        else:
          currentSmoothedLevelTarget=self.pyramid("testPyr_%s.jpg"%(1),i+1)


        result_sobelTestPyr=self.__produce_gradient_image(currentSmoothedLevelTarget,2**i)
        #cv.NamedWindow("result_sobelTestPyr_level%s"%(i+1), cv.CV_WINDOW_NORMAL)
        #cv.ShowImage("result_sobelTestPyr_level%s"%(i+1),result_sobelTestPyr)

        self.g_target_image.append(self.__produce_gradient_image(currentSmoothedLevelTarget,2**i))
        #cv.WaitKey()


    for imagenumber in range(0,len(images),1):#for each image
        for i in range(0,TOTALLEVELS):#for 4 different levels of sobel derivative smoothing

              print "imagenumber=%s"%(imagenumber)
              print "totaleLevel=%s"%(i)

              if not USE_PYRAMID_SMOOTHING:
                self.g_images.append(self.__produce_gradient_image(images[imagenumber], 2**i))
              else:

                if i==0:
                  #imageNames[imagenumber] is the current training imag name
                  currentSmoothedLevelTarget=self.pyramid(imageNames[imagenumber],i+1,  imagenumber, i)
                else:
                  currentSmoothedLevelTarget=self.pyramid("testPyr_%s.jpg"%(1),i+1)


                result_sobelTestPyr=self.__produce_gradient_image(currentSmoothedLevelTarget,2**i)
                #cv.NamedWindow("result_sobelTestPyr_level%s"%(i+1), cv.CV_WINDOW_NORMAL)
                #cv.ShowImage("result_sobelTestPyr_level%s"%(i+1),result_sobelTestPyr)

                self.g_images.append(result_sobelTestPyr)
                #cv.WaitKey()




    self.asm = asm
    # Copy mean shape as starting shape and transform it to origin
    #self.shape = Shape.from_vector(asm.mean).transform(t)

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
    degreesOfRotation=0#-15

    #degreesOfRotation=0
    #CenterOfMass=Point(0,0)
    if CenterOfMass.__eq__(Point(0,0)):
        t= Point(0,0)

    meanshapeVectorNew = asm.mean

    l_r=t.x
    u_d=t.y

    meanshapeVectorNew = recalculateTransform(l_r, u_d, degreesOfRotation, "",  meanshapeVectorNew)


    targetCopy = cv.CreateImage(cv.GetSize(self.target), self.target.depth, 3)
    #Shows original target
    cv.Copy(self.target, targetCopy)


    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress

        targetCopy = cv.CreateImage(cv.GetSize(self.target), self.target.depth, 3)
        cv.Copy(self.target, targetCopy)


        key = cv2.waitKey(1) & 0xFF
        global l_r, u_d, rot

        # if the 'up' key is pressed, transform up
        if key == (82):

            u_d+=-5
            print 'RIGHT'
            cv.WaitKey()
            global meanshapeVectorNew
            meanshapeVectorNew = recalculateTransform(l_r, u_d, rot, "Up",  meanshapeVectorNew)

        # if the 'down' key is pressed, transform up
        if key == (84):
            u_d+=5
            global meanshapeVectorNew
            meanshapeVectorNew = recalculateTransform(l_r, u_d, rot, "Down", meanshapeVectorNew)

        # if the 'left' key is pressed, transform up
        if key == (81):
            l_r+=-5
            global meanshapeVectorNew
            meanshapeVectorNew = recalculateTransform(l_r, u_d, rot, "Left",  meanshapeVectorNew)

        # if the 'right' key is pressed, transform up
        if key == (83):
            l_r+=5
            global meanshapeVectorNew
            meanshapeVectorNew = recalculateTransform(l_r, u_d, rot, "Right", meanshapeVectorNew)


        #ROTATE
        # if the 'a' key is pressed, rotate left
        if key == (97):
            rot+=-1
            global meanshapeVectorNew
            meanshapeVectorNew = recalculateTransform(l_r, u_d, rot, "_rotating", meanshapeVectorNew)
        # if the 'd' key is pressed, rotate right
        if key == (100):
            rot+=1
            global meanshapeVectorNew
            meanshapeVectorNew = recalculateTransform(l_r, u_d, rot, "_rotating", meanshapeVectorNew)

        #draw the shape again on the target image

        drawMeanShape(meanshapeVectorNew, targetCopy)

        cv.NamedWindow("targetCopy", cv.CV_WINDOW_NORMAL)
        cv.ShowImage("targetCopy",targetCopy)


        if key == ord("e"):

            #Write out all the contour landmarks points x,y
            print finalPlacedShape.pts
            cv.WaitKey()

            for i, pt in enumerate (finalPlacedShape.pts):
                print pt.x, pt.y

            break




    finalPlacedVec = finalPlacedShape.get_vector()
    #self.shape = Shape.from_vector(asm.mean).transform(t, degreesOfRotation, CenterOfMass)
    self.shape = Shape.from_vector(finalPlacedVec)#.transform(t, degreesOfRotation, CenterOfMass)


    # And resize shape to fit image if required
    if self.__shape_outside_image(self.shape, self.images[0]):
      self.shape = self.__resize_shape_to_fit_image(self.shape, self.images[0])

  def __shape_outside_image(self, s, i):
    for p in s.pts:
      if p.x > i.width or p.x < 0 or p.y > i.height or p.y < 0:
        return True
    return False

  def __resize_shape_to_fit_image(self, s, i):
    # Get rectagonal boundary orf shape
    min_x = min([pt.x for pt in s.pts])
    min_y = min([pt.y for pt in s.pts])
    max_x = max([pt.x for pt in s.pts])
    max_y = max([pt.y for pt in s.pts])

    # If it is outside the image then we'll translate it back again
    if min_x > i.width: min_x = 0
    if min_y > i.height: min_y = 0
    ratio_x = (i.width-min_x) / (max_x - min_x)
    ratio_y = (i.height-min_y) / (max_y - min_y)
    new = Shape([])
    for pt in s.pts:
      new.add_point(Point(pt.x*ratio_x if ratio_x < 1 else pt.x, \
                          pt.y*ratio_y if ratio_y < 1 else pt.y))
    return new

  def sobel(self, imageToSobel, scale):
    import cv2
    import numpy as np

    #scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    img = cv2.imread(imageToSobel)
    img = cv2.GaussianBlur(img,(3,3),0)
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

    cv2.imshow('dst',dst)
    cv2.waitKey(0)
    cv2.imwrite("AfterDerivative-grey_image.jpg",dst)



  def __produce_gradient_image_modified(self, i, scale):


    size = cv.GetSize(i)
    grey_image = cv.CreateImage(size, 8, 1)

    size = [s/scale for s in size]
    grey_image_small = cv.CreateImage(size, 8, 1)

    cv.CvtColor(i, grey_image, cv.CV_RGB2GRAY)

    cv.NamedWindow("BeforeDerivative-grey_image", cv.CV_WINDOW_NORMAL)
    cv.ShowImage("BeforeDerivative-grey_image",grey_image)
    cv.SaveImage("BeforeDerivative-grey_image.jpg", grey_image)

    self.sobel("BeforeDerivative-grey_image.jpg",scale)
    deriv=cv.LoadImage("AfterDerivative-grey_image.jpg")
    cv.CvtColor(deriv, grey_image, cv.CV_RGB2GRAY)
    return grey_image

    '''
    #calculate image derivative
    df_dx = cv.CreateImage(cv.GetSize(i), cv.IPL_DEPTH_16S, 1)
    cv.Sobel( grey_image, df_dx, 1, 1 )
    cv.Convert(df_dx, grey_image)#convert back to 8 bit depth
    #re scale

    cv.NamedWindow("AfterDerivative-grey_image", cv.CV_WINDOW_NORMAL)
    cv.ShowImage("AfterDerivative-grey_image",grey_image)
    #cv.WaitKey()

    #resize to "/2**scale" times & back to original size
    cv.Resize(grey_image, grey_image_small)#, interpolation=cv.CV_INTER_NN)
    cv.Resize(grey_image_small, grey_image)#, interpolation=cv.CV_INTER_NN)

    cv.NamedWindow("ResizedAfterDerivative-grey_image", cv.CV_WINDOW_NORMAL)
    cv.ShowImage("ResizedAfterDerivative-grey_image",grey_image)
    #cv.WaitKey()

    return grey_image
    '''


  def __produce_gradient_image(self, i, scale):


    size = cv.GetSize(i)
    grey_image = cv.CreateImage(size, 8, 1)

    size = [s/scale for s in size]
    grey_image_small = cv.CreateImage(size, 8, 1)

    cv.CvtColor(i, grey_image, cv.CV_RGB2GRAY)

    #decrease brightness of the image, increase the contrast of the image - To enhace the edges
    if ENHANCE_IMAGE:
        for i in range(grey_image.height):
            for j in range(grey_image.width):
                #print 'decreasing brightness'
                grey_image[i,j] = grey_image[i,j] - 50
                #print 'increasing contrast'
                grey_image[i,j] = grey_image[i,j] * 1.5



    #cv.NamedWindow("BeforeDerivative-grey_image", cv.CV_WINDOW_NORMAL)
    #cv.ShowImage("BeforeDerivative-grey_image",grey_image)
    #cv.SaveImage("BeforeDerivative-grey_image.jpg", grey_image)
    #cv.WaitKey()

    #calculate image derivative
    #JUST_SMOOTH_NO_SOBEL- don't get 1d derivatives as we want to calculate them ourselves, cause we want to get the signed gradient alon gthe profile normal
    if not JUST_SMOOTH_NO_SOBEL:
      df_dx = cv.CreateImage(cv.GetSize(i), cv.IPL_DEPTH_16S, 1)
      cv.Sobel( grey_image, df_dx, 1, 1 )
      cv.Convert(df_dx, grey_image)#convert back to 8 bit depth
    #re scale

    #cv.NamedWindow("AfterDerivative-grey_image", cv.CV_WINDOW_NORMAL)
    #cv.ShowImage("AfterDerivative-grey_image",grey_image)
    #cv.WaitKey()

    #resize to "/2**scale" times & back to original size
    cv.Resize(grey_image, grey_image_small)#, interpolation=cv.CV_INTER_NN)
    cv.Resize(grey_image_small, grey_image)#, interpolation=cv.CV_INTER_NN)


    #cv.EqualizeHist(grey_image, grey_image)

    #cv.NamedWindow("ResizedAfterDerivative-grey_image", cv.CV_WINDOW_NORMAL)
    #cv.ShowImage("ResizedAfterDerivative-grey_image",grey_image)
    #cv.WaitKey()



    return grey_image

    '''

    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    #img = cv2.imread('messi5.jpg')
    size = cv.GetSize(i)
    grey_image = cv.CreateImage(size, 8, 1)

    cv.CvtColor(i, grey_image, cv.CV_RGB2GRAY)

    #print ("i= %s"%(grey_image))
    img = np.asarray(grey_image[:,:])


    gray = cv2.GaussianBlur(img,(3,3),0)
    #gray = img
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Gradient-X
    #grad_x = cv2.Sobel(gray,ddepth,1,0,ksize = 3, scale = scale, delta = delta,borderType = cv2.BORDER_DEFAULT)
    grad_x = cv2.Scharr(gray,ddepth,1,0)

    # Gradient-Y
    #grad_y = cv2.Sobel(gray,ddepth,0,1,ksize = 3, scale = scale, delta = delta, borderType = cv2.BORDER_DEFAULT)
    grad_y = cv2.Scharr(gray,ddepth,0,1)

    abs_grad_x = cv2.convertScaleAbs(grad_x)   # converting back to uint8
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    dst = cv2.addWeighted(abs_grad_x,0.5,abs_grad_y,0.5,0)
    #dst = cv2.add(abs_grad_x,abs_grad_y)

    #cv2.imshow('dst',dst)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


    source = dst # source is numpy array
    bitmap = cv.CreateImageHeader((source.shape[1], source.shape[0]), cv.IPL_DEPTH_8U, 1)
    cv.SetData(bitmap, source.tostring(),
               source.dtype.itemsize * 1 * source.shape[1])

    return bitmap
    '''





  def do_iteration(self, scale, reset_giAllPointsShapeVecSet):

    '''reset_giAllPointsShapeVecSet is set when we want to recalculate training Gmean & Training Covariance Matrices for Higher Pyramid Level'''

    #build statistical shape model BEFORE ANYTHING ELSE
    #for each landmark create a set based on all training images
    #so if 25 landmakrs and 50 images-->then each gi set will have 50 sampled profiles, 1 for each image

    #sample either side of this landmark and put in vector "gPointIthImage"
    #calc total sum of absolute elements of the vector
    #divide vector by the sum of  absolute elements
    #repeat this for each training image, to get a set of normalised samples
    #{"giAllPointsShapeVecSet" } for the given model point

    #then calculate for each giAllImagesSet's element its "mean" and "covariance matrix" to compare agains
    #using mahalanobis distance during image search


    #Calculate Training Gmean & Covariance matrices once for each scale level

    if USE_MAX_EDGE:
        USE_PROFILE_MODEL=0

    else:
        USE_PROFILE_MODEL=1

        if reset_giAllPointsShapeVecSet==1:
            #global curBest
            #curBest=-1

            #?????? SHOULD I CALCULATE THE FOLLOWING STATISTICAL MODEL ALL SCALE LEVELS ??????
            #print 'Refill Vectors'
            cv.WaitKey()

            #reset:
               #1) the list that holds all points' sets accross training set of images ,as we want to fill it with the new intensities based on a different scale/level
               #2) the list that holds all points' sets Gmean accross training set of images
               #3) the list that holds all points' sets CovarianceMatrices accross training set of images
            giAllPointsShapeVecSet=[]
            tmpList=[]
            del giAllPointsShapeVecSet[:]
            del tmpList[:]

            del TrainingMean[:]
            del TrainingCovarianceMatrices[:]

            #for shapenum in range(len(self.originalShapes)):#for each shape--------

            fileName = sys.argv[2].replace("/", "_")
            print fileName
            fileName=fileName.split("_")[:-3]
            stringfileName=""
            for i in fileName:
                stringfileName+=i+"_"
            print stringfileName
            cv.WaitKey()

            # If any of the stringfileName+"trainingSetLevel_%d.txt"%(scale) DOESN'T already exist, then procuce it

            #create trainingSet dir and put training profile vectors of all levels there
            if not os.path.isdir("trainingSetProfiles"):
                os.mkdir("trainingSetProfiles")

            if not os.path.exists( "trainingSetProfiles/"+stringfileName+"_"+str(STEP)+"_"+str(PIXELS_SEARCH)+"_trainingSetLevel_%d.txt"%(scale) ):
                with open("trainingSetProfiles/"+stringfileName+"_"+str(STEP)+"_"+str(PIXELS_SEARCH)+"_trainingSetLevel_%d.txt"%(scale), 'w') as f:#write out the writeOutStatisticalTrainingProfileModel

                    for i in range(len(self.originalShapes[0].pts)):#for each point of any shape (we used originalShapes[0] to get the length of the list containing the landmarks)
                        gPointIthImageShapeVec = []
                        for imageShapenumber in range(0,len(self.images),1):#for each training image/shape
                            #print "\nsample point %d at training image %s"%(i,imageShapenumber)

                            #REMOVE
                            #scale=2

                            #Memory Structured (each consecutive 4 elements contain each image's all lavels)

                            #self.g_images[0]-->scale 0 , 'image 1'
                            #..
                            #self.g_images[3]-->scale 3 , 'image 1'
                            #
                            #... ...
                            #
                            #self.g_images[4]-->scale 0 , 'image 2'
                            #..
                            #self.g_images[7]-->scale 3 , 'image 2'



                            cv.SaveImage("SAMPLING TEST.jpg", self.g_images[imageShapenumber*TOTALLEVELS+scale])
                            #cv.SaveImage("SAMPLING TEST.jpg", self.g_target_image[scale])
                            cv.NamedWindow("SAMPLING TEST Statistical model sampling", cv.CV_WINDOW_NORMAL)
                            cv.ShowImage("SAMPLING TEST Statistical model sampling", self.g_images[imageShapenumber*TOTALLEVELS+scale])
                            #cv.WaitKey()

                            path = 'SAMPLING TEST.jpg'
                            mat = cv.LoadImage(path, cv.CV_LOAD_IMAGE_UNCHANGED)

                            if SAMPLING_PY_ENABLED:
                              gPointIthImageShapeVec.append(self.sampling.createStatisticalProfileModel(\
                                                                    i, scale, self.g_images[imageShapenumber*TOTALLEVELS+scale],\
                                                                    self.originalShapes[imageShapenumber],mat))

                              writeOutStatisticalTrainingProfileModel=self.sampling.createStatisticalProfileModel(\
                                                                      i, scale, self.g_images[imageShapenumber*TOTALLEVELS+scale],\
                                                                      self.originalShapes[imageShapenumber],mat)
                              #print "writeOutStatisticalTrainingProfileModel=.."
                              #print writeOutStatisticalTrainingProfileModel

                              for s in writeOutStatisticalTrainingProfileModel:
                                  f.write(str(s) + ' ')
                              #f.write(str(writeOutStatisticalTrainingProfileModel))
                              f.write('\n')#image shape written out
                              #cv.WaitKey()


                            else:
                              gPointIthImageShapeVec.append(self.  __createStatisticalProfileModel(\
                                                                    i, scale, self.g_images[imageShapenumber*TOTALLEVELS+scale],\
                                                                    self.originalShapes[imageShapenumber],mat))
                            if SHOW_EVERY_POSSIBLE_STATISTICAL_SAMPLE==1:
                                print "\nsamplePointsList AFTER normalization = gPointIthImageShapeVec[imageShapenumber]=%r"%(gPointIthImageShapeVec[imageShapenumber])
                                #cv.WaitKey()

                        #each element of the following vector contains a distinct set of samples for each landmark, each of which sample corresponds to a training image
                        #giAllPointsShapeVecSet[0] contains landmark 0 set across all training images

                        giAllPointsShapeVecSet.append(gPointIthImageShapeVec)
                        if SHOW_EVERY_POSSIBLE_STATISTICAL_SAMPLE==1:
                            print "\ngiAllPointsShapeVecSet%r = "%(giAllPointsShapeVecSet)
                            #print len(giAllPointsShapeVecSet[0][0])
                            #cv.WaitKey()

                    #iterate over the list of sets for each landmark  / each pointset has this many elements as the number of images
                    #25 points, each point has 25points x 16images = 400 individual elements
                    ##print "giAllPointsShapeVecSet=%r"%(len(giAllPointsShapeVecSet))
                    ##print "giAllPointsShapeVecSet=%r"%(len(giAllPointsShapeVecSet[0]))
                    ##print "giAllPointsShapeVecSet=%r"%(giAllPointsShapeVecSet)
                    ##print "giAllPointsShapeVecSet=%r"%(giAllPointsShapeVecSet[0])
                    ##print "giAllPointsShapeVecSet=%r"%(giAllPointsShapeVecSet[0])
                    #ar=np.array(giAllPointsShapeVecSet[0])
                    ##print "ar=%r"%(ar)
                    #meanOfPoint0=np.mean(np.array(giAllPointsShapeVecSet[0]))
                    ##print "meanOfPoint0=%r"%(meanOfPoint0)
                    #covarOfPoint0=np.cov(np.array(giAllPointsShapeVecSet[0]))
                    ##print "covarOfPoint0=%r"%(covarOfPoint0)#each covariance matrix will have a size of "totalLandmarkPoints * totalLandmarkPoints"


            else:#if any of the "trainingSetProfiles/"+stringfileName+"_"+str(STEP)+"_"+str(PIXELS_SEARCH)+"_trainingSetLevel_%d.txt"%(scale) is found just read it, don't reproduce it
                print "Reading..trainingSetLevel_%d.txt, instead of reproducing"%(scale)
                #cv.WaitKey()
                print "trainingSetProfiles/"+stringfileName+"_"+str(STEP)+"_"+str(PIXELS_SEARCH)+"_trainingSetLevel_%d.txt"%(scale)
                #cv.WaitKey()
                with open("trainingSetProfiles/"+stringfileName+"_"+str(STEP)+"_"+str(PIXELS_SEARCH)+"_trainingSetLevel_%d.txt"%(scale), 'r') as f:
                    lines = f.readlines()

                counter=0
                perShapeList=[]
                for line in lines:

                    #print counter
                    counter+=1

                    tmpList=[]
                    line = map(float, line.split())
                    tmpList.append(line)

                    #print tmpList
                    #cv.WaitKey()

                    if (counter%len(self.images))==0:
                        perShapeList.append(tmpList[0])
                        #print 'len(perShapeList)=%r'%(len(perShapeList))
                        #print 'perShapeList=%r'%(perShapeList)

                        #print giAllPointsShapeVecSet
                        #cv.WaitKey()

                    if (counter% (len(lines)/len(self.originalShapes[0].pts)) )==0:
                        giAllPointsShapeVecSet.append(perShapeList)
                        #print (giAllPointsShapeVecSet)
                        #print len(giAllPointsShapeVecSet)
                        #cv.WaitKey()



                '''
                for i in lines:
                    print i
                    tmpList.append(i.strip())
                    #cv.WaitKey()
                '''
                #print tmpList


                #giAllPointsShapeVecSet=tmpList
                #print 'len(tmpList)=%r'%(len(tmpList))
                #cv.WaitKey()

            #print 'len(giAllPointsShapeVecSet)=%r'%(len(giAllPointsShapeVecSet))
            #cv.WaitKey()
            #print 'giAllPointsShapeVecSet=%r'%(giAllPointsShapeVecSet)
            #cv.WaitKey()


            #so now it's time to find the mean of each the sets for each landmark point, as well as each of their covariance matrix
            #so if total number of annotated landmarks for each image is 25, then we are going to end up with 25 means and 25 covariance matrices (1 for each point landmarked)
            for pointSet in ((giAllPointsShapeVecSet)):


                #for each landmark point find "mean & covariance" matrix across training images set
                #for el in range (len(giAllPointsShapeVecSet[pointSet])):


                if STOP_AT_THE_END_OF_EVERY_LANDMARK==1:
                  #print "for point %d, giAllPointsShapeVecSet[%d]=%r"%(pointSet,pointSet,giAllPointsShapeVecSet[pointSet])
                  cv.WaitKey()


                  #meanOfPointX=np.mean(np.array(giAllPointsShapeVecSet[pointSet]))
                  ##print "meanOfPoint_%d=%r"%(pointSet,meanOfPointX)
                  #cv.WaitKey()
                  #print "giAllPointsShapeVecSet[%d]=%r"%(pointSet,giAllPointsShapeVecSet[pointSet])
                  cv.WaitKey()



                '''
                for i in (pointSet):
                    if len(i)<13:
                        print 'STOP'
                        cv.WaitKey()
                #
                #  #print "[pointSet]", (pointSet)
                #  print len(i)

                print "AAAAAAAAAAAAAAAAAAAaa,\n",pointSet
                '''

                print "pointSet = %r "%(pointSet)
                cv.WaitKey()

                meanOfPointX=np.mean(np.array(pointSet), axis=0, dtype=np.float64)
                #print "Correct meanOfPoint_%d=%r"%(pointSet,meanOfPointX)

                #cv.WaitKey()
                covarOfPointX=np.cov(np.array(pointSet), rowvar=0)
                ##print "covarOfPoint_%d=%r"%(covarOfPointX)#each covariance matrix will have a size of "totalLandmarkPoints * totalLandmarkPoints"
                TrainingMean.append(meanOfPointX)
                TrainingCovarianceMatrices.append(covarOfPointX)

                if STOP_AT_THE_END_OF_EVERY_LANDMARK==1:
                    #print "TrainingMean=%r"%(TrainingMean)
                    #print "TrainingCovarianceMatrices=%r"%(TrainingCovarianceMatrices)
                    cv.WaitKey()






    """ Does perform single iteration of the shape fitting algorithm.
    This is useful when we want to show the algorithm converging on
    an image


    :return shape: The shape in its current orientation
    """

    # Build new shape from max points along normal to current
    # shape
    s = Shape([])

    counter=0

    createImageOncePerIteration=True
    for i, pt in enumerate(self.shape.pts):
      if USE_MAX_EDGE:
        s.add_point(self.__get_max_along_normal(i, scale,createImageOncePerIteration))#, curBest

      #REMOVE
      #scale=2      
      if USE_PROFILE_MODEL:
        if SAMPLING_PY_ENABLED:
            point, savedOffset = self.sampling.get_MAHALANOBIS(self.shape, self.g_target_image, TrainingCovarianceMatrices, TrainingMean, i, scale, createImageOncePerIteration)
            s.add_point(point)#, curBest

            print "savedOffset=%r"%(savedOffset)
            print "0.5 * STEP=%r"%(0.5 * STEP)
            if abs(savedOffset) <= 0.5 * STEP:
                counter+=1
        else:
            s.add_point(self.__get_MAHALANOBIS(i, scale, createImageOncePerIteration))#, curBest
      createImageOncePerIteration=False

    #Check if convergence has occured, if the 90% of the number of points from the new 's' shape picked up are withing 50% of the STEP profile ...

    print "counter=%r"%(counter)
    print "0.9*len(s.pts)=%r"%(0.9*len(s.pts))
    #cv.WaitKey()

    if counter > 0.9 * len(s.pts):
        print "CONVERGED BECAUSE 90% of the total number of points from the s new shape picked up during the last search \nare withing 50% of the STEP profile ... "
        if scale-1 >= 0:
            print "Skip to next level -->%d"%(scale-1)
        else:
            print "Done Segmenting!"
        cv.WaitKey()
        #Skip to next level, finer resolution (scale)
        skipToNextLevel=True
        return skipToNextLevel
    else:
        #Don't skip to next level (scale)
        skipToNextLevel=False


    '''
    pointIdsChanged = [0] * len(self.prevEquivalentPointIdsChanged)

    if len(self.prevShapeList) ==0:#first time it runs..0th iteration, then fill it in
        self.prevShapeList = s.pts[:]
        print self.prevShapeList
    else:
        #compare prevShapeList with s shape
        for i in range (len(self.prevShapeList)):
            print self.prevShapeList[i]
            #cv.WaitKey()
            print s.pts[i]

            if self.prevShapeList[i] != s.pts[i]:
                print "%d point was different"%(i)
                pointIdsChanged.append(i)#these point ids have changed
                #cv.WaitKey()
        #count
        counter=0
        for i in range (len(self.prevEquivalentPointIdsChanged)):
            if (pointIdsChanged[i] != self.prevEquivalentPointIdsChanged[i]) and pointIdsChanged[i]!=0:
                counter+=1

        print "(10/100) * len(self.prevEquivalentPointIdsChanged)=%r"%( (0.1) * len(self.prevEquivalentPointIdsChanged))
        print "counter=%d"%(counter)

        if counter < (10/100) * len(self.prevEquivalentPointIdsChanged):
            print 'CONVERGENCE REACHED'
            exit(0)





        #add the changed landmark ids as long as the convergence criterion hasn't been reached
        if self.prevShapeList[i] != s.pts[i]:
            print "%d point was different"%(i)
            self.prevEquivalentPointIdsChanged.append(i)
            #cv.WaitKey()

        #save it as previous
        self.prevShapeList = s.pts[:]
     '''






    #align new found shape point positions to the mean shape, and then use this to calculate the filt parameterst of the current shape
    new_shape = s.align_to_shape(Shape.from_vector(self.asm.mean), self.asm.w)


    #start from mean shape
    new = self.asm.mean

    for i in range(len(self.asm.evecs.T)):
      #update parameters to fit the current shape
      b = np.dot(self.asm.evecs[:,i], new_shape.get_vector() - self.asm.mean)
      #max_b = 2*math.sqrt(self.asm.evals[i])
      #constrain parameters to not deviate more the 3 standart deviations
      max_b = 3*math.sqrt(self.asm.evals[i])#/TOTALLEVELS**scale #scale represents the level
      b = max(min(b, max_b), -max_b)
      #use parameters to calculate new shape model
      new = new + self.asm.evecs[:,i]*b


    #apply parameters to shape..
    #basically, the our self.shape with be resulted from:
    # 1) s shape's alignment to the new shape (which is the mean shape updated/constrained to the new pose/fit parameters calulcated)
    # these parameters are continuously calculted from aligning the newly found/optimal positions (resulting from the profile model in this case) to the mean applying certain precalulated asm model weights

    self.shape = Shape.from_vector(new).align_to_shape(s, self.asm.w)

  def __get_max_along_normal(self, p_num, scale,createImageOncePerIteration):
    """ Gets the max edge response along the normal to a point

    :param p_num: Is the number of the point in the shape
    """

    norm = self.shape.get_normal_to_point(p_num)
    p = self.shape.pts[p_num]

    # Find extremes of normal within the image
    # Test x first

    min_t = -p.x / norm[0]
    if p.y + min_t*norm[1] < 0:
      min_t = -p.y / norm[1]
    elif p.y + min_t*norm[1] >  self.g_target_image[scale].height:
      min_t = ( self.g_target_image[scale].height - p.y) / norm[1]

    # X first again
    max_t = ( self.g_target_image[scale].width - p.x) / norm[0]
    if p.y + max_t*norm[1] < 0:
      max_t = -p.y / norm[1]
    elif p.y + max_t*norm[1] >  self.g_target_image[scale].height:
      max_t = ( self.g_target_image[scale].height - p.y) / norm[1]
    '''

    min_t = -p.x / norm[0]
    if p.y + min_t*norm[1] < 0:
      min_t = -p.y / norm[1]
    elif p.y + min_t*norm[1] > self.target.height:
      min_t = (self.target.height - p.y) / norm[1]

    # X first again
    max_t = (self.target.width - p.x) / norm[0]
    if p.y + max_t*norm[1] < 0:
      max_t = -p.y / norm[1]
    elif p.y + max_t*norm[1] > self.target.height:
      max_t = (self.target.height - p.y) / norm[1]
    '''


    # Swap round if max is actually larger...
    tmp = max_t
    max_t = max(min_t, max_t)
    min_t = min(min_t, tmp)

    # Get length of the normal within the image
    x1 = min(p.x+max_t*norm[0], p.x+min_t*norm[0])
    x2 = max(p.x+max_t*norm[0], p.x+min_t*norm[0])
    y1 = min(p.y+max_t*norm[1], p.y+min_t*norm[1])
    y2 = max(p.y+max_t*norm[1], p.y+min_t*norm[1])
    l = math.sqrt((x2-x1)**2 + (y2-y1)**2)


    cv.NamedWindow("targetImageToWorkAgainst-grey_image", cv.CV_WINDOW_NORMAL)
    cv.ShowImage("targetImageToWorkAgainst-grey_image",self.g_target_image[scale])
    #cv.WaitKey()

    img = cv.CreateImage(cv.GetSize(self.target), self.g_target_image[scale].depth, 1)
    cv.Copy(self.g_target_image[scale], img)


    if createImageOncePerIteration:
      convertedToColouredImg = cv.CreateImage(cv.GetSize( self.g_target_image[scale]), cv.IPL_DEPTH_8U, 3)
      cv.CvtColor(img, convertedToColouredImg, cv.CV_GRAY2BGR)
      self.convertedToColouredImg = convertedToColouredImg

    convertedToColouredImg = self.convertedToColouredImg


    #convertedToColouredImg = cv.CreateImage(cv.GetSize(self.g_target_image[scale]), cv.IPL_DEPTH_8U, 3)
    #cv.CvtColor(img, convertedToColouredImg, cv.CV_GRAY2BGR)


    #cv.Circle(img, \
    #    (int(norm[0]*min_t + p.x), int(norm[1]*min_t + p.y)), \
    #    5, (0, 0, 0))
    #cv.Circle(img, \
    #    (int(norm[0]*max_t + p.x), int(norm[1]*max_t + p.y)), \
    #    5, (0, 0, 0))

    # Scan over the whole line
    max_pt = p
    max_edge = 0
    max_edge_contr = 0

    # Now check over the vector
    #v = min(max_t, -min_t)
    #for t in drange(min_t, max_t, (max_t-min_t)/l):
    #search = 2*scale+1
    search = PIXELS_SEARCH+1

    #Assigning Gaussian Weights to gradient profile elements
    #Precalculate the total number of whisker elements so as to .. remove noise based on gausian kernel contribution (4.2   H. Lu and F. Yang    Width of Search Profile)
    whiskerElements=0
    #for t in drange(-search+1 if -search > min_t else min_t+1, \
    #		     search if search < max_t else max_t , 1):

    if -search+1==0:#make sure -search+1 doesn't become 0, by hacking/setting search to 2
        search=2
    for t in range(-search+1, search,1):
        whiskerElements+=1
    #print "\nwhiskerElements=%d"%(whiskerElements)

    if DEBUG_LINES==1:
        cv.WaitKey()
    #create 1D convolution filter for each whisker line
    x = np.arange(-whiskerElements/2+1, (whiskerElements/2)+1, 1)
    myprint(x)
    stdDev=10
    #convolution array
    conv1D = 1 / np.sqrt(2 * np.pi) * stdDev*np.exp(-x ** 2 / (2.*stdDev**2))
    #conv1D = 1 / stdDev * np.sqrt(2 * np.pi) * stdDev*np.exp( (-1/2.0) * (x/stdDev)**2)
    myprint(conv1D)


    ##print "search=%s"%(search)
    ##print "2*(scale+1)=%s"%( 2*(scale+1))
    #cv.WaitKey()

    # Look 6 pixels to each side too
    for side in range(-2*(scale+1), 2*(scale+1) ):#profile WIDTH

      # Normal to normal...
      ##print "side=%s"%(side)
      #cv.WaitKey()

      #profile LENGTH
      new_p = Point(p.x + side*-norm[1], p.y + side*norm[0])
      smoothedContribution=0

      for t in drange(-search+1 if -search > min_t else min_t+1, \
                       search if search < max_t else max_t-1 , 1):


          ##print "TATA=%s"%(t)
          #cv.WaitKey()

          #normal for this side (side starts spreads across the window width, controlled by 'search' variable)
          x = int(new_p.x + t*norm[0])
          y = int(new_p.y + t*norm[1])


          #Make sure it's within the image boundaries
          if x < 0 or x > self.target.width or y < 0 or y > self.target.height:
            continue

          if DEBUG_LINES==1:

              #Tangent line to landmark point (which is orthogonal to the whisker/normal)
              cv.Circle(convertedToColouredImg, (int(new_p.x), int(new_p.y)), 1, (0,255,255))
              #Whisker/normal
              cv.Circle(convertedToColouredImg, (x, y), 1, (255,255,0))#individually tested window points

          if SHOW_EVERY_POSSIBLE_POINT==1:
              cv.NamedWindow("convertedToColouredImg", cv.CV_WINDOW_NORMAL)
              cv.ShowImage("convertedToColouredImg",convertedToColouredImg)
              cv.WaitKey(1)

          ##print x, y, self.g_target_image.width, self.g_target_image.height

          contributionCoefficient = conv1D [t+(whiskerElements/2)]
          ##print "conv1D[%d]=%s"%(t+(whiskerElements/2), contributionCoefficient)

          smoothedContribution= contributionCoefficient * self.g_target_image[scale][y-1, x-1]
          #print "%s * %s"%(contributionCoefficient , self.g_target_image[scale][y-1, x-1])


          #if DEBUG_LINES==1:
          #  cv.WaitKey()


          #if self.g_target_image[scale][y-1, x-1] > max_edge:	#choice based on 'Simple maximum minima edge' test
          if smoothedContribution > max_edge_contr:		#choice based on 'Gaussian Spread Contribution of possible individually tested window points' test

            #last best point chosen - for this side .. for this normal

            if SHOW_EVERY_POSSIBLE_POINT==1:
                cv.Circle(convertedToColouredImg, (x, y), 0, (255,255,255))
                cv.WaitKey(1)

            #print "new max_edge_contr=%s"%(smoothedContribution)
            max_edge_contr = smoothedContribution
            max_edge = self.g_target_image[scale][y-1, x-1]
            #max_pt = Point(new_p.x, new_p.y)
            max_pt = Point(x, y)


            #max_pt = Point(new_p.x + norm[0], new_p.y + norm[1])

    #for point in self.shape.pts:
    #  cv.Circle(img, (int(point.x), int(point.y)), 3, (255,255,255))

    if STOP_AT_THE_END_OF_EVERY_LANDMARK==1:
        #FINAL best point chosen - for this iteration
        cv.Circle(convertedToColouredImg, (int(max_pt.x), int(max_pt.y)), 0, (0,0,255))
        cv.NamedWindow("convertedToColouredImg", cv.CV_WINDOW_NORMAL)
        cv.ShowImage("convertedToColouredImg",convertedToColouredImg)

        #print "max_pt=%s"%(max_pt)
        cv.WaitKey()

    return max_pt


  def   __createStatisticalProfileModel(self, p_num, scale , image, shape, mat):
      """ Gets the max edge response along the normal to a point

      :param p_num: Is the number of the point in the shape
      """


      norm = shape.get_normal_to_point(p_num)
      #print norm
      p = shape.pts[p_num]

      # Find extremes of normal within the image
      # Test x first
      min_t = -p.x / norm[0]
      if p.y + min_t*norm[1] < 0:
        min_t = -p.y / norm[1]
      elif p.y + min_t*norm[1] > image.height:
        min_t = (image.height - p.y) / norm[1]

      # X first again
      max_t = (image.width - p.x) / norm[0]
      if p.y + max_t*norm[1] < 0:
        max_t = -p.y / norm[1]
      elif p.y + max_t*norm[1] > image.height:
        max_t = (image.height - p.y) / norm[1]

      # Swap round if max is actually larger...
      tmp = max_t
      max_t = max(min_t, max_t)
      min_t = min(min_t, tmp)

      # Get length of the normal within the image
      x1 = min(p.x+max_t*norm[0], p.x+min_t*norm[0])
      x2 = max(p.x+max_t*norm[0], p.x+min_t*norm[0])
      y1 = min(p.y+max_t*norm[1], p.y+min_t*norm[1])
      y2 = max(p.y+max_t*norm[1], p.y+min_t*norm[1])
      l = math.sqrt((x2-x1)**2 + (y2-y1)**2)


      cv.NamedWindow("Statistical model sampling", cv.CV_WINDOW_NORMAL)
      #cv.ShowImage("Statistical model sampling", image)
      #cv.WaitKey()

      img = cv.CreateImage(cv.GetSize(image), image.depth, 1)
      cv.Copy(image, img)

      convertedToColouredImg = cv.CreateImage(cv.GetSize(image), cv.IPL_DEPTH_8U, 3)
      cv.CvtColor(img, convertedToColouredImg, cv.CV_GRAY2BGR)

      # Scan over the whole line
      max_pt = p
      max_edge = 0
      max_edge_contr = 0

      #search = 2*scale+1
      search = PIXELS_SEARCH+1

      #Assigning Gaussian Weights to gradient profile elements
      #Precalculate the total number of whisker elements so as to .. remove noise based on gausian kernel contribution (4.2   H. Lu and F. Yang    Width of Search Profile)
      whiskerElements=0
      #for t in drange(-search+1 if -search > min_t else min_t+1, \
      #		     search if search < max_t else max_t , 1):

      if -search+1==0:#make sure -search+1 doesn't become 0, by hacking/setting search to 2
          search=2
      for t in range(-search+1, search,1):
          whiskerElements+=1
      #print "\nwhiskerElements=%d"%(whiskerElements)

      if DEBUG_LINES==1:
          cv.WaitKey()
      #create 1D convolution filter for each whisker line
      x = np.arange(-whiskerElements/2+1, (whiskerElements/2)+1, 1)
      myprint(x)
      stdDev=10
      #convolution array
      conv1D = 1 / np.sqrt(2 * np.pi) * stdDev*np.exp(-x ** 2 / (2.*stdDev**2))
      #conv1D = 1 / stdDev * np.sqrt(2 * np.pi) * stdDev*np.exp( (-1/2.0) * (x/stdDev)**2)
      myprint(conv1D)

      # Look 6 pixels to each side too
      '''
      for side in range(-2*(scale+1), 2*(scale+1) ):#profile WIDTH

        # Normal to normal...
        ##print "side=%s"%(side)
        #cv.WaitKey()

        #profile LENGTH
        new_p = Point(p.x + side*-norm[1], p.y + side*norm[0])
      '''
      side=0
      #samplePointsList=Shape([])
      #sampled intensity on the derivative image
      samplePointsList= []

      if USE_2D_PROFILE==1:

          smoothedContribution=0

          #print "creating Statistical Shape 1D Profile"

          side=0
          new_p = Point(p.x + side*-norm[1], p.y + side*norm[0])

          #calculate contributionCoef at range
          tmpIntensitiesArray= np.zeros((2*search-1, 2*search-1))
          for side in range(-search+1, search ):#profile WIDTH
              new_p = Point(p.x + side*-norm[1], p.y + side*norm[0])
              #print "side",side
              #cv.WaitKey()



              for t in drange(-search+1 if -search > min_t else min_t+1, \
                             search if search < max_t else max_t-1 , 1):
                     x = int(new_p.x + (t)*norm[0])
                     y = int(new_p.y + (t)*norm[1])
                     ##print "side=%d,t=%d",side+6,t+6
                     ##print "tmpIntensitiesArray side=%d,t=%d",tmpIntensitiesArray[side+6][t+6]
                     ##print "tmpIntensitiesArray=%r",tmpIntensitiesArray
                     #cv.WaitKey()
                     tmpIntensitiesArray[side+(search-1)][t+(search-1)] = image[y-1, x-1]

                     ##print "t",t
                     #cv.WaitKey()
                     ##print "[%d][%d]"%(side+(search-1),t+(search-1))
                     ##print "(tmpIntensitiesArray)=%r",(tmpIntensitiesArray)
                     #cv.WaitKey()


          ##print "gaussian_filter(tmpIntensitiesArray, 3)=%r",gaussian_filter(tmpIntensitiesArray, 3)
          #cv.WaitKey()


          # Look 6 pixels to each side too
          for side in range(-search+1, search ):#profile WIDTH
              new_p = Point(p.x + side*-norm[1], p.y + side*norm[0])


              for t in drange(-search+1 if -search > min_t else min_t+1, \
                             search if search < max_t else max_t-1 , 1):

                    #normal for this side (side starts spreads across the window width, controlled by 'search' variable)
                    x = int(new_p.x + (t)*norm[0])
                    y = int(new_p.y + (t)*norm[1])

                    '''
                    for a given point we sample along a profile k pixels either side of
                    the model point in the ith training image. We have 2k + 1 samples which can be
                    put in a vector gi .
                    '''

                    cv.Circle(convertedToColouredImg, (x, y), 1, (0,255,0))#individually sampled points along the normal

                    #samplePointsList.add_point(Point(x,y))
                    #ADD THE INTENSITY ..not the points themselves
                    #samplePointsList.append(image[y-1, x-1])

                    contributionCoefficient=gaussian_filter(tmpIntensitiesArray, 3)[side+(search-1)][t+(search-1)]
                    ##print "side=%d,t=%d",side+(search-1),t+(search-1)
                    ##print "gaussian_filter(tmpIntensitiesArray, 3)[side+search][t+search]",gaussian_filter(tmpIntensitiesArray, 3)[side+(search-1)][t+(search-1)]
                    ##print "gaussian_filter(tmpIntensitiesArray, 3)=%r",gaussian_filter(tmpIntensitiesArray, 3)
                    #cv.WaitKey()

                    smoothedContribution = contributionCoefficient# * image[y-1, x-1]
                    samplePointsList.append(int(smoothedContribution))


                    ##print "x=%r, y=%r"%(x-1,y-1)

                    if SHOW_EVERY_POSSIBLE_POINT==1:
                        cv.NamedWindow("convertedToColouredImgTMP", cv.CV_WINDOW_NORMAL)
                        cv.ShowImage("convertedToColouredImgTMP",convertedToColouredImg)
                        cv.WaitKey(1)


          #normalize the sample by dividing by the sum of the absolute element values
          #absPointSum=Point(0,0)
          absPointSum=0

          #for i in samplePointsList.pts:
          for sampledIntensity in samplePointsList:

              #print "sampledIntensity=%r"%(sampledIntensity)
              absPointSum+=abs(sampledIntensity)
              ##print abs(i)
              ##print absPointSum
              #cv.WaitKey()


          #print "\nsamplePointsList before normalization=%r"%(samplePointsList)
          for i in range(len(samplePointsList)):
              if SHOW_EVERY_POSSIBLE_STATISTICAL_SAMPLE==1:
                  print samplePointsList.pts[i]
                  ##print absPointSum
                  #print i
                  #cv.WaitKey()
              #normalize point
              #samplePointsList.pts[i] *= 1/absPointSum
              if absPointSum!=0:
                  samplePointsList[i] *= 1/absPointSum




          if SHOW_EVERY_POSSIBLE_STATISTICAL_SAMPLE==1:
              print samplePointsList
              #cv.WaitKey()


          return samplePointsList

      elif USE_1D_PROFILE==1:

              smoothedContribution=0

              #print "creating Statistical Shape 1D Profile"

              side=0
              new_p = Point(p.x + side*-norm[1], p.y + side*norm[0])

              #calculate contributionCoef at range
              tmpIntensitiesArray=[]
              #for t in drange(-search+1 if -search > min_t else min_t+1, \
                             #search if search < max_t else max_t-1 , 1):
              #CHANGED THE ABOVE 2 LINES to THE BELOW 1 LINE (could cause out of bounds exception while accessing the image pixels)
              for t in range(-search+1, search , 1):
                         x = int(new_p.x + (STEP*t)*norm[0])
                         y = int(new_p.y + (STEP*t)*norm[1])
                         tmpIntensitiesArray.append(image[y-1, x-1])


              #print "Before tmpIntensitiesArray =%r"%(tmpIntensitiesArray)
              #cv.WaitKey()
              #print "multiplied with =%r"%(gaussianMatrix[offsetT+(search-1)])
              #cv.WaitKey()
              #tmpIntensitiesArray= [0.034561*el for el in tmpIntensitiesArray]
              #print "After tmpIntensitiesArray =%r"%(tmpIntensitiesArray)
              #cv.WaitKey()


              #for t in drange(-search+1 if -search > min_t else min_t+1, \
                             #search if search < max_t else max_t-1 , 1):
              #CHANGED THE ABOVE 2 LINES to THE BELOW 1 LINE (could cause out of bounds exception while accessing the image pixels)
              for t in range(-search+1, search , 1):

                    #normal for this side (side starts spreads across the window width, controlled by 'search' variable)
                    x = int(new_p.x + (STEP*t)*norm[0])
                    y = int(new_p.y + (STEP*t)*norm[1])

                    '''
                    for a given point we sample along a profile k pixels either side of
                    the model point in the ith training image. We have 2k + 1 samples which can be
                    put in a vector gi .
                    '''

                    cv.Circle(convertedToColouredImg, (x, y), 1, (0,255,0))#individually sampled points along the normal

                    #samplePointsList.add_point(Point(x,y))
                    #ADD THE INTENSITY ..not the points themselves
                    #use the sobeled image
                    #samplePointsList.append(image[y-1, x-1])

                    #use this if no gaussian distribution is taken into account
                    #use the sobeled image read from 'SAMPLING_TEST.jpg' back in again
                    #samplePointsList.append(mat[y-1, x-1])

                    #Here we add to out current sampled intensity vector,
                    #the gaussian contribution instead of the actual intensity
                    #contributionCoefficient = conv1D [t+(whiskerElements/2)]
                    #contributionCoefficient=gaussianMatrix[t+(search-1)]

                    #contributionCoefficient=gaussianMatrix[t+(search-1)]*tmpIntensitiesArray[t+(search-1)]
                    #contributionCoefficient=gaussian_filter1d(tmpIntensitiesArray, 1)[t+(search-1)]
                    contributionCoefficient=tmpIntensitiesArray[t+(search-1)]

                    ##print "tmpIntensitiesArray=",tmpIntensitiesArray
                    ##print "gaussian_filter1d",gaussian_filter1d(tmpIntensitiesArray, 1)
                    ##print "gaussian_filter1d t",gaussian_filter1d(tmpIntensitiesArray,  3)[t+(search-1)]
                    ##print "t",t
                    #cv.WaitKey()


                    ##print "contributionCoefficient=",contributionCoefficient
                    ##print "gaussianMatrix=",gaussianMatrix
                    ##print "t=",t
                    #cv.WaitKey()

                    smoothedContribution = contributionCoefficient# * image[y-1, x-1]

                    samplePointsList.append(smoothedContribution)


                    '''
                    if not smoothedContribution:

                        print "tmpIntensitiesArray=",tmpIntensitiesArray
                        print "smoothedContribution=",smoothedContribution

                        cv.WaitKey()
                    '''


                    #print "x=%r, y=%r"%(x-1,y-1)

                    #Was
                    ##print "%r..intensity value added to sampleIntensity"%(image[y-1, x-1])
                    #print "%r..intensity value added to sampleIntensity"%(image[y-1, x-1])

                    if SHOW_EVERY_POSSIBLE_POINT==1:
                        cv.NamedWindow("convertedToColouredImgTMP", cv.CV_WINDOW_NORMAL)
                        cv.ShowImage("convertedToColouredImgTMP",convertedToColouredImg)
                        #cv.WaitKey()


              #normalize the sample by dividing by the sum of the absolute element values
              #absPointSum=Point(0,0)
              absPointSum=0

              #for i in samplePointsList.pts:
              for sampledIntensity in samplePointsList:

                  #print "sampledIntensity=%r"%(sampledIntensity)
                  absPointSum+=abs(sampledIntensity)
                  ##print abs(i)
                  ##print absPointSum
                  #cv.WaitKey()

              #print "\nsamplePointsList before normalization=%r"%(samplePointsList)
              for i in range(len(samplePointsList)):
                  if SHOW_EVERY_POSSIBLE_STATISTICAL_SAMPLE==1:
                      print samplePointsList
                      ##print absPointSum
                      #print i
                      #cv.WaitKey()
                  #normalize point
                  #samplePointsList.pts[i] *= 1/absPointSum
                  if absPointSum!=0:
                      samplePointsList[i] *= 1/absPointSum


              if SHOW_EVERY_POSSIBLE_STATISTICAL_SAMPLE==1:
                  print samplePointsList
                  #cv.WaitKey()




              if not samplePointsList:
                  print "samplePointsList=",samplePointsList
                  cv.WaitKey()

              '''
              if len(samplePointsList)<13:
                  print 'STOP'
                  cv.WaitKey()
              print len(samplePointsList)
              '''

              return samplePointsList


  def __get_MAHALANOBIS(self, p_num, scale, createImageOncePerIteration):#, curBest
      """ Gets the optimum fit based on the Mahalanobis for this particular point on this scale/level

      :param p_num: Is the number of the point in the shape
      """

      norm = self.shape.get_normal_to_point(p_num)
      p = self.shape.pts[p_num]

      # Find extremes of normal within the image
      # Test x first


      min_t = -p.x / norm[0]
      if p.y + min_t*norm[1] < 0:
        min_t = -p.y / norm[1]
      elif p.y + min_t*norm[1] >  self.g_target_image[scale].height:
        min_t = ( self.g_target_image[scale].height - p.y) / norm[1]

      # X first again
      max_t = ( self.g_target_image[scale].width - p.x) / norm[0]
      if p.y + max_t*norm[1] < 0:
        max_t = -p.y / norm[1]
      elif p.y + max_t*norm[1] >  self.g_target_image[scale].height:
        max_t = ( self.g_target_image[scale].height - p.y) / norm[1]
      '''

      min_t = -p.x / norm[0]
      if p.y + min_t*norm[1] < 0:
        min_t = -p.y / norm[1]
      elif p.y + min_t*norm[1] > self.target.height:
        min_t = (self.target.height - p.y) / norm[1]

      # X first again
      max_t = (self.target.width - p.x) / norm[0]
      if p.y + max_t*norm[1] < 0:
        max_t = -p.y / norm[1]
      elif p.y + max_t*norm[1] > self.target.height:
        max_t = (self.target.height - p.y) / norm[1]
      '''


      # Swap round if max is actually larger...
      tmp = max_t
      max_t = max(min_t, max_t)
      min_t = min(min_t, tmp)

      # Get length of the normal within the image
      x1 = min(p.x+max_t*norm[0], p.x+min_t*norm[0])
      x2 = max(p.x+max_t*norm[0], p.x+min_t*norm[0])
      y1 = min(p.y+max_t*norm[1], p.y+min_t*norm[1])
      y2 = max(p.y+max_t*norm[1], p.y+min_t*norm[1])
      l = math.sqrt((x2-x1)**2 + (y2-y1)**2)


      cv.NamedWindow("targetImageToWorkAgainst-grey_image", cv.CV_WINDOW_NORMAL)
      cv.ShowImage("targetImageToWorkAgainst-grey_image",self.g_target_image[scale])
      #cv.WaitKey()

      img = cv.CreateImage(cv.GetSize( self.g_target_image[scale]), self.g_target_image[scale].depth, 1)
      cv.Copy(self.g_target_image[scale], img)


      if createImageOncePerIteration:
        convertedToColouredImg = cv.CreateImage(cv.GetSize( self.g_target_image[scale]), cv.IPL_DEPTH_8U, 3)
        cv.CvtColor(img, convertedToColouredImg, cv.CV_GRAY2BGR)
        self.convertedToColouredImg = convertedToColouredImg

      convertedToColouredImg = self.convertedToColouredImg


      # Scan over the whole line
      max_pt = p
      max_edge = 0
      max_edge_contr = 0

      # Now check over the vector

      #search = 2*scale+1
      search = PIXELS_SEARCH+1

      #Assigning Gaussian Weights to gradient profile elements
      #Precalculate the total number of whisker elements so as to .. remove noise based on gausian kernel contribution (4.2   H. Lu and F. Yang    Width of Search Profile)
      whiskerElements=0
      #for t in drange(-search+1 if -search > min_t else min_t+1, \
      #		     search if search < max_t else max_t , 1):

      if -search+1==0:#make sure -search+1 doesn't become 0, by hacking/setting search to 2
          search=2
      for t in range(-search+1, search,1):
          whiskerElements+=1
      #print "\nwhiskerElements=%d"%(whiskerElements)

      if DEBUG_LINES==1:
          cv.WaitKey()
      #create 1D convolution filter for each whisker line
      x = np.arange(-whiskerElements/2+1, (whiskerElements/2)+1, 1)
      myprint(x)
      stdDev=10
      #convolution array
      conv1D = 1 / np.sqrt(2 * np.pi)*stdDev*  np.exp(-x ** 2 / (2.*stdDev**2))
      #conv1D = 1 / stdDev * np.sqrt(2 * np.pi) * stdDev*np.exp( (-1/2.0) * (x/stdDev)**2)
      myprint(conv1D)


      # Look 6 pixels to each side too
      '''
      for side in range(-2*(scale+1), 2*(scale+1) ):#profile WIDTH

        # Normal to normal...
        ##print "side=%s"%(side)
        #cv.WaitKey()

        #profile LENGTH
        new_p = Point(p.x + side*-norm[1], p.y + side*norm[0])
      '''


      #sampled intensity on the derivative image
      curBest=-1
      bestPoint=0

      #at level 2-> runs from -6 to 6, then at level 1 runs from -3 to 3, and then at level 0 from -1 to 1
      rangeAtThisLevel=0
      rangeAtThisLevel=-search+1 if -search > min_t else min_t+1
      #print "rangeAtThisLevel=%d"%(rangeAtThisLevel)
      rangeAtThisLevel=abs(rangeAtThisLevel)


      if USE_2D_PROFILE==1:

        smoothedContribution=0

        #print "creating Statistical Shape 1D Profile"
        offsetT=0
        side=0
        new_p = Point(p.x + side*-norm[1], p.y + side*norm[0])

        xxx = int(new_p.x + (offsetT)*norm[0])
        yyy = int(new_p.y + (offsetT)*norm[1])

        hack=0


        for offsetT in drange(-rangeAtThisLevel,rangeAtThisLevel+1,1):

            tmpIntensitiesArray=[]

            #calculate contributionCoef at range
            tmpIntensitiesArray= np.zeros((2*search-1, 2*search-1))
            for side in range(-search+1, search ):#profile WIDTH
                new_p = Point(p.x + side*-norm[1], p.y + side*norm[0])
                #print "side",side
                #cv.WaitKey()



                for t in drange(-search+1 if -search > min_t else min_t+1, \
                               search if search < max_t else max_t-1 , 1):
                       x = int(new_p.x + (t+offsetT)*norm[0])
                       y = int(new_p.y + (t+offsetT)*norm[1])
                       ##print "side=%d,t=%d",side+6,t+6
                       ##print "tmpIntensitiesArray side=%d,t=%d",tmpIntensitiesArray[side+6][t+6]
                       ##print "tmpIntensitiesArray=%r",tmpIntensitiesArray
                       #cv.WaitKey()
                       tmpIntensitiesArray[side+(search-1)][t+(search-1)] = self.g_target_image[scale][y-1, x-1]

                       ##print "t",t
                       #cv.WaitKey()
                       ##print "[%d][%d]"%(side+(search-1),t+(search-1))
                       ##print "(tmpIntensitiesArray)=%r",(tmpIntensitiesArray)
                       #cv.WaitKey()


            ##print "gaussian_filter(tmpIntensitiesArray, 3)=%r",gaussian_filter(tmpIntensitiesArray, 3)
            #cv.WaitKey()




            samplePointsList= []

            side=0
            new_p_centered = Point(p.x + side*-norm[1], p.y + side*norm[0])
            xx = int(new_p_centered.x + (offsetT)*norm[0])
            yy = int(new_p_centered.y + (offsetT)*norm[1])

            # Look 6 pixels to each side too

            for side in range(-search+1, search ):#profile WIDTH

                  new_p_centered = Point(p.x + side*-norm[1], p.y + side*norm[0])

                  for t in drange(-search+1 if -search > min_t else min_t+1, \
                                 search if search < max_t else max_t-1 , 1):

                        #normal for this side (side starts spreads across the window width, controlled by 'search' variable)
                        x = int(new_p.x + (t+offsetT)*norm[0])
                        y = int(new_p.y + (t+offsetT)*norm[1])


                        if SHOW_EVERY_POSSIBLE_POINT==1:
                          cv.Circle(convertedToColouredImg, (x, y), 1, ( (offsetT+rangeAtThisLevel)*15, 255-(offsetT+rangeAtThisLevel)*15, search*(offsetT+10) ) )#individually sampled points along the normal
                          cv.Circle(convertedToColouredImg, (xx, yy), 2, ( 255,255,0 ) )#whisker points
                          cv.NamedWindow("convertedToColouredImgTMP", cv.CV_WINDOW_NORMAL)
                          cv.ShowImage("convertedToColouredImgTMP",convertedToColouredImg)
                          cv.WaitKey(1)


                        #now for each normal point sample again "this many times" to gather information as to that area
                        #samplePointsList= []

                        #for offsetT in drange(-rangeAtThisLevel,rangeAtThisLevel+1,1):
                        #new_p_centered_xx_offset=int(x + offsetT*norm[0])#int(x+offsetT)
                        #new_p_centered_yy_offset=int(y + offsetT*norm[1])#int(y+offsetT)

                        '''
                        for a given point we sample along a profile k pixels either side of
                        the model point in the ith training image. We have 2k + 1 samples which can be
                        put in a vector gi .
                        '''


                        #contributionCoefficient = conv1D [t+(whiskerElements/2)]
                        contributionCoefficient=gaussian_filter(tmpIntensitiesArray, 3)[side+(search-1)][t+(search-1)]
                        ##print "side=%d,t=%d",side+(search-1),t+(search-1)
                        ##print "gaussian_filter(tmpIntensitiesArray, 3)[side+search][t+search]",gaussian_filter(tmpIntensitiesArray, 3)[side+(search-1)][t+(search-1)]
                        ##print "gaussian_filter(tmpIntensitiesArray, 3)=%r",gaussian_filter(tmpIntensitiesArray, 3)
                        #cv.WaitKey()

                        smoothedContribution = contributionCoefficient# * self.g_target_image[scale][y-1, x-1]
                        samplePointsList.append(smoothedContribution)

                        if SHOW_EVERY_POSSIBLE_POINT==1:
                          print "x=%r"%(x)
                          ##print "y=%r"%(y)
                          #print "For the target image of this level for points: x=%r, y=%r"%(x-1,y-1)
                          #print "Inensity=%r"%(self.g_target_image[scale][y-1, x-1])


                        if SHOW_EVERY_POSSIBLE_POINT==1:
                            cv.NamedWindow("convertedToColouredImgTMP", cv.CV_WINDOW_NORMAL)
                            cv.ShowImage("convertedToColouredImgTMP",convertedToColouredImg)
                            cv.WaitKey(1)



            #normalize the sample by dividing by the sum of the absolute element values
            absPointSum=0
            #for i in samplePointsList.pts:
            for sampledIntensity in samplePointsList:

                #print "sampledIntensity=%r"%(sampledIntensity)
                absPointSum+=abs(sampledIntensity)


            if SHOW_EVERY_POSSIBLE_POINT==1:
                print "\nsamplePointsList before normalization=%r"%(samplePointsList)

            for i in range(len(samplePointsList)):
                if SHOW_EVERY_POSSIBLE_STATISTICAL_SAMPLE==1:
                    print i

                if absPointSum!=0:#make sure absPointIsNotZero
                  samplePointsList[i] *= 1/absPointSum


            if SHOW_EVERY_POSSIBLE_STATISTICAL_SAMPLE==1:
                #print "normalized sampledIntensity=%r"%(samplePointsList)
                cv.WaitKey()

            ##print "current whisker normalized point samplePointsList=%r"%(samplePointsList)
            #cv.WaitKey()
            ##print "TrainingMean[%d]=%r"%(p_num,TrainingMean[p_num])
            #cv.WaitKey()
            #calculate Mahalanobis for this point using samplePointsList of the current point
            #..and compared to the training Gmean & Cov matrix of acquire during training
            #tmpMah = scipy.spatial.distance.mahalanobis(np.array(samplePointsList) , TrainingMean[p_num] , np.linalg.inv(np.cov(np.array(samplePointsList),TrainingMean[p_num])))



            #tmpMah = scipy.spatial.distance.mahalanobis(samplePointsList , (TrainingMean[p_num]) , np.linalg.inv(np.cov(samplePointsList,(TrainingMean[p_num]))))
            invcovar=0
            #invcovar=cv2.invert(TrainingCovarianceMatrices[p_num], invcovar, cv2.DECOMP_SVD)#mycovar[0] when used with cv2.calcCovarMatrix #'''OR covar'''
            #invcovar=invcovar[1]

            #invcovar = np.linalg.inv(np.array(TrainingCovarianceMatrices[p_num]))#.reshape((3,3))

            #This worked with 1d profile sampling, but it's hell to slow with 2d profile
            #invcovar=cv2.invert(np.array(TrainingCovarianceMatrices[p_num]), invcovar, cv2.DECOMP_SVD)
            #invcovar=invcovar[1]

            invcovar=np.linalg.pinv(np.array(TrainingCovarianceMatrices[p_num]))

            TrainingMean[p_num] = np.reshape(TrainingMean[p_num],(-1,1))
            samplePointsList = np.reshape(samplePointsList,(-1,1))
            if SHOW_EVERY_POSSIBLE_POINT==1:
              ##print "TrainingCovarianceMatrices[p_num]=%r"%(TrainingCovarianceMatrices[p_num])
              ##print "invcovar=%r"%(invcovar)
              ##print "TrainingMean[%d]=%r"%(p_num,TrainingMean[p_num])
              ##print "normalized samplePointsList=%r"%(samplePointsList)

              ##print "sizeof invcovar=%r"%(invcovar.size)
              ##print "sizeof np.array(samplePointsList).size=%d"%(np.array(samplePointsList).size)
              ##print "sizeof (TrainingMean[p_num]).size=%d"%((TrainingMean[p_num]).size)
              cv.WaitKey(1)

            #print "TrainingCovarianceMatrices[p_num]=%r"%(TrainingCovarianceMatrices[p_num])
            #print "invcovar=%r"%(invcovar)
            #print "TrainingMean[%d]=%r"%(p_num,TrainingMean[p_num])
            #print "normalized samplePointsList=%r"%(samplePointsList)

            #print "sizeof invcovar=%r"%(invcovar.size)
            #print "sizeof np.array(samplePointsList).size=%d"%(np.array(samplePointsList).size)
            #print "sizeof (TrainingMean[p_num]).size=%d"%((TrainingMean[p_num]).size)
            #cv.WaitKey(0)


            #test each offset vector with the training Gmean for this landmark calculated accross set of images
            tmpMah=scipy.spatial.distance.mahalanobis( np.array(samplePointsList), TrainingMean[p_num], invcovar)

            #print "tmpMah=%r"%(tmpMah)
            #cv.WaitKey()
            #cv::Mahalanobis

            if tmpMah<curBest or curBest<0:
                curBest = tmpMah;

                #print "self.g_target_image[scale][yyy-1, xxx-1]=%r"%(self.g_target_image[scale][yyy-1, xxx-1])
                #print "self.g_target_image[scale][yy-1, xx-1]=%r"%(self.g_target_image[scale][yy-1, xx-1])
                #cv.WaitKey()

                #check if the initial position set is somewhow different to one of the possible options, otherwise don't move & used the initial pos
                if (self.g_target_image[scale][yyy-1, xxx-1]) != (self.g_target_image[scale][yy-1, xx-1]):
                  bestPoint = Point(xx, yy)#outter testing index along the whisker
                  cv.Circle(convertedToColouredImg, (bestPoint.x, bestPoint.y), 1, (0,255,255))#individually sampled points along the normal


                  #if once an xx is chosen then prefer this over the xxx, regardless if they have the same intensity
                  hack=1
                else:
                    if hack!=1:#if no other point had been previously selected as bestpoint, then get the original mean/starting position
                      bestPoint = Point(xxx,yyy)#leave at initial pos / don't landmark move anywhere new
                      cv.Circle(convertedToColouredImg, (bestPoint.x, bestPoint.y), 1, (0,0,255))#individually sampled points along the normal



                #bestPoint = Point(xx,yy);#outter testing index along the whisker
                ##bestEP[i] = V[k];
                #print "New best Mahalanobis found"
                #pick the points whose profile g has minimizes the Mahalanobis distance Fit Function
                #cv.Circle(convertedToColouredImg, (bestPoint.x, bestPoint.y), 2, (0,0,255))#individually sampled points along the normal

                #print "bestPoint=%s"%(bestPoint)
                #cv.WaitKey()

                if SHOW_EVERY_POSSIBLE_POINT==1:
                    cv.NamedWindow("convertedToColouredImgTMP", cv.CV_WINDOW_NORMAL)
                    cv.ShowImage("convertedToColouredImgTMP",convertedToColouredImg)
                    cv.WaitKey(1)

      elif USE_1D_PROFILE==1:

              #always 'leave at initial poisition', if no other point
              #is minimizing the mahalanobis cost function
              offsetT=0

              #setting initial position of landmark (WITHOUT ANY OFFSET)
              side=0
              new_p_centered = Point(p.x + side*-norm[1], p.y + side*norm[0])
              xxx = int(new_p_centered.x + (offsetT)*norm[0])
              yyy = int(new_p_centered.y + (offsetT)*norm[1])

              hack=0

              for offsetT in drange(-rangeAtThisLevel,rangeAtThisLevel+1,1):

                  side=0
                  samplePointsList= []
                  #the very center of the profile
                  new_p_centered = Point(p.x + side*-norm[1], p.y + side*norm[0])

                  #the center of the sample offsetted profile
                  new_p_centered_xx_offset = int(new_p_centered.x + (offsetT)*norm[0])
                  new_p_centered_yy_offset = int(new_p_centered.y + (offsetT)*norm[1])

                  #calculate contributionCoef at range
                  tmpIntensitiesArray=[]
                  #for t in drange(-search+1 if -search > min_t else min_t+1, \
                                 #search if search < max_t else max_t-1 , 1):
                  #CHANGED THE ABOVE 2 LINES to THE BELOW 1 LINE (could cause out of bounds exception while accessing the image pixels)
                  for t in range(-search+1, search , 1):
                             x = int(new_p_centered.x + (STEP*t+offsetT)*norm[0])
                             y = int(new_p_centered.y + (STEP*t+offsetT)*norm[1])

                             tmpIntensitiesArray.append(self.g_target_image[scale][y-1, x-1])

                  print "Before tmpIntensitiesArray =%r"%(tmpIntensitiesArray)
                  #cv.WaitKey()
                  print "multiplied with =%r"%(gaussianMatrix[offsetT+(search-1)])
                  #cv.WaitKey()
                  #tmpIntensitiesArray= [gaussianMatrix[offsetT+(search-1)]*el for el in tmpIntensitiesArray]
                  print "After tmpIntensitiesArray =%r"%(tmpIntensitiesArray)
                  #cv.WaitKey()

                  #for t in drange(-search+1 if -search > min_t else min_t+1, \
                                 #search if search < max_t else max_t-1 , 1):
                  #CHANGED THE ABOVE 2 LINES to THE BELOW 1 LINE (could cause out of bounds exception while accessing the image pixels)
                  for t in range(-search+1, search , 1):

                        #normal for this side (side starts spreads across the window width, controlled by 'search' variable)

                        #the each of the points along the 'sample offsetted profile'
                        x = int(new_p_centered.x + (STEP*t+offsetT)*norm[0])
                        y = int(new_p_centered.y + (STEP*t+offsetT)*norm[1])


                        if SHOW_EVERY_POSSIBLE_POINT==1:
                          cv.Circle(convertedToColouredImg, (x, y), 1, ( (offsetT+rangeAtThisLevel)*50, 255-(offsetT+rangeAtThisLevel)*50, search*(offsetT+10) ) )#individually sampled points along the normal

                          #cv.Circle(convertedToColouredImg, (xxx, yyy), 1, ( 255,0,0) )#individually sampled points along the normal
                          cv.Circle(convertedToColouredImg, (new_p_centered_xx_offset, new_p_centered_yy_offset), 2, ( 255,255,0 ) )#whisker points
                          cv.NamedWindow("convertedToColouredImgTMP", cv.CV_WINDOW_NORMAL)
                          cv.ShowImage("convertedToColouredImgTMP",convertedToColouredImg)
                          #cv.WaitKey()


                        #now for each normal point sample again "this many times" to gather information as to that area
                        #samplePointsList= []

                        #for offsetT in drange(-rangeAtThisLevel,rangeAtThisLevel+1,1):
                        #new_p_centered_xx_offset=int(x + offsetT*norm[0])#int(x+offsetT)
                        #new_p_centered_yy_offset=int(y + offsetT*norm[1])#int(y+offsetT)

                        '''
                        for a given point we sample along a profile k pixels either side of
                        the model point in the ith training image. We have 2k + 1 samples which can be
                        put in a vector gi .
                        '''


                        #contributionCoefficient = conv1D [t+(whiskerElements/2)]
                        #contributionCoefficient=gaussianMatrix[t+(search-1)]

                        #CURRENTLY IT LOOKS LIKE THIS IS AN INCONSISTENT COMPARED TO THE contributionCoefficient,
                        #CALCULATED DURING   __createStatisticalProfileModel(...)
                        #contributionCoefficient=gaussianMatrix[t+(search-1)]*tmpIntensitiesArray[t+(search-1)]
                        #contributionCoefficient=gaussian_filter1d(tmpIntensitiesArray, 1)[t+(search-1)]
                        contributionCoefficient=tmpIntensitiesArray[t+(search-1)]



                        ##print "tmpIntensitiesArray=",tmpIntensitiesArray
                        ##print "gaussian_filter1d",gaussian_filter1d(tmpIntensitiesArray, 1)
                        ##print "gaussian_filter1d t",gaussian_filter1d(tmpIntensitiesArray,  3)[t+(search-1)]
                        ##print "t",t
                        #cv.WaitKey()

                        smoothedContribution= contributionCoefficient# * self.g_target_image[scale][y-1, x-1]

                        #samplePointsList.add_point(Point(x,y))
                        #ADD THE INTENSITY ..not the points themselves
                        #samplePointsList.append(self.g_target_image[scale][y-1, x-1])#the image target at this level
                        samplePointsList.append(smoothedContribution)#the image target at this level


                        if SHOW_EVERY_POSSIBLE_POINT==1:
                          print "x=%r"%(x)
                          ##print "y=%r"%(y)
                          #print "For the target image of this level for points: x=%r, y=%r"%(x-1,y-1)
                          #print "Intensity=%r"%(self.g_target_image[scale][y-1, x-1])


                        if SHOW_EVERY_POSSIBLE_POINT==1:
                            #cv.NamedWindow("convertedToColouredImgTMP", cv.CV_WINDOW_NORMAL)
                            #cv.ShowImage("convertedToColouredImgTMP",convertedToColouredImg)
                            cv.WaitKey(1)



                  #normalize the sample by dividing by the sum of the absolute element values
                  absPointSum=0
                  #for i in samplePointsList.pts:
                  for sampledIntensity in samplePointsList:

                      #print "sampledIntensity=%r"%(sampledIntensity)
                      absPointSum+=abs(sampledIntensity)


                  if SHOW_EVERY_POSSIBLE_POINT==1:
                      print "\nsamplePointsList before normalization=%r"%(samplePointsList)

                  for i in range(len(samplePointsList)):
                      if SHOW_EVERY_POSSIBLE_STATISTICAL_SAMPLE==1:
                          print i

                      if absPointSum!=0:#make sure absPointIsNotZero
                        samplePointsList[i] *= 1/absPointSum


                  if SHOW_EVERY_POSSIBLE_STATISTICAL_SAMPLE==1:
                      print "normalized sampledIntensity=%r"%(samplePointsList)
                      #cv.WaitKey()

                  ##print "current whisker normalized point samplePointsList=%r"%(samplePointsList)
                  #cv.WaitKey()
                  ##print "TrainingMean[%d]=%r"%(p_num,TrainingMean[p_num])
                  #cv.WaitKey()
                  #calculate Mahalanobis for this point using samplePointsList of the current point
                  #..and compared to the training Gmean & Cov matrix of acquire during training
                  #tmpMah = scipy.spatial.distance.mahalanobis(np.array(samplePointsList) , TrainingMean[p_num] , np.linalg.inv(np.cov(np.array(samplePointsList),TrainingMean[p_num])))



                  #tmpMah = scipy.spatial.distance.mahalanobis(samplePointsList , (TrainingMean[p_num]) , np.linalg.inv(np.cov(samplePointsList,(TrainingMean[p_num]))))
                  invcovar=0
                  #invcovar=cv2.invert(TrainingCovarianceMatrices[p_num], invcovar, cv2.DECOMP_SVD)#mycovar[0] when used with cv2.calcCovarMatrix #'''OR covar'''
                  #invcovar=invcovar[1]

                  #invcovar = np.linalg.inv(np.array(TrainingCovarianceMatrices[p_num]))#.reshape((3,3))

                  #This worked with 1d profile sampling, but it's hell to slow with 2d profile
                  #invcovar=cv2.invert(np.array(TrainingCovarianceMatrices[p_num]), invcovar, cv2.DECOMP_SVD)
                  #invcovar=invcovar[1]

                  invcovar=np.linalg.pinv(np.array(TrainingCovarianceMatrices[p_num]))

                  TrainingMean[p_num] = np.reshape(TrainingMean[p_num],(-1,1))
                  samplePointsList = np.reshape(samplePointsList,(-1,1))
                  if SHOW_EVERY_POSSIBLE_POINT==1:
                    ##print "TrainingCovarianceMatrices[p_num]=%r"%(TrainingCovarianceMatrices[p_num])
                    ##print "invcovar=%r"%(invcovar)
                    ##print "TrainingMean[%d]=%r"%(p_num,TrainingMean[p_num])
                    ##print "normalized samplePointsList=%r"%(samplePointsList)

                    ##print "sizeof invcovar=%r"%(invcovar.size)
                    ##print "sizeof np.array(samplePointsList).size=%d"%(np.array(samplePointsList).size)
                    ##print "sizeof (TrainingMean[p_num]).size=%d"%((TrainingMean[p_num]).size)
                    cv.WaitKey(1)

                  '''
                  #print "TrainingCovarianceMatrices[p_num]=%r"%(TrainingCovarianceMatrices[p_num])
                  #print "invcovar=%r"%(invcovar)
                  #print "TrainingMean[%d]=%r"%(p_num,TrainingMean[p_num])
                  #print "normalized samplePointsList=%r"%(samplePointsList)

                  #print "sizeof invcovar=%r"%(invcovar.size)
                  #print "sizeof np.array(samplePointsList).size=%d"%(np.array(samplePointsList).size)
                  #print "sizeof (TrainingMean[p_num]).size=%d"%((TrainingMean[p_num]).size)

                  cv.WaitKey()
                  '''


                  #test each offset vector with the training Gmean for this landmark calculated accross set of images
                  tmpMah=scipy.spatial.distance.mahalanobis( np.array(samplePointsList), TrainingMean[p_num], invcovar)

                  #print "tmpMah=%r"%(tmpMah)
                  #cv.WaitKey()
                  #cv::Mahalanobis

                  #if this point is a better match is found OR if the first point tested
                  if tmpMah<curBest or curBest<0:
                      curBest = tmpMah;


                      #check if the initial position set is somewhow different to one of the possible options, otherwise don't move & used the initial pos
                      if (self.g_target_image[scale][yyy-1, xxx-1]) != (self.g_target_image[scale][new_p_centered_yy_offset-1, new_p_centered_xx_offset-1]):
                        bestPoint = Point(new_p_centered_xx_offset, new_p_centered_yy_offset)#outter testing index along the whisker
                        cv.Circle(convertedToColouredImg, (bestPoint.x, bestPoint.y), 1, (0,255,255))#individually sampled points along the normal


                        #if once an new_p_centered_xx_offset is chosen then prefer this over the xxx, regardless if they have the same intensity
                        hack=1
                      else:
                          if hack!=1:#if no other point had been previously selected as bestpoint, then get the original mean/starting position
                            bestPoint = Point(xxx,yyy)#leave at initial pos / don't landmark move anywhere new
                            cv.Circle(convertedToColouredImg, (bestPoint.x, bestPoint.y), 1, (0,0,255))#individually sampled points along the normal




                      #bestEP[i] = V[k];
                      #print "New best Mahalanobis found"
                      #pick the points whose profile g has minimizes the Mahalanobis distance Fit Function

                      #convertedToColouredImg = cv.CreateImage(cv.GetSize( self.g_target_image[scale]), cv.IPL_DEPTH_8U, 3)
                      #cv.CvtColor(img, convertedToColouredImg, cv.CV_GRAY2BGR)

                      #cv.NamedWindow("convertedToColouredImgTMP", cv.CV_WINDOW_NORMAL)
                      #cv.ShowImage("convertedToColouredImgTMP",convertedToColouredImg)




                      #print "bestPoint=%s"%(bestPoint)
                      #cv.WaitKey()

                      if SHOW_EVERY_POSSIBLE_POINT==1:
                          cv.NamedWindow("convertedToColouredImgTMP", cv.CV_WINDOW_NORMAL)
                          cv.ShowImage("convertedToColouredImgTMP",convertedToColouredImg)
                          #cv.WaitKey()




      cv.NamedWindow("convertedToColouredImgTMP", cv.CV_WINDOW_NORMAL)
      cv.ShowImage("convertedToColouredImgTMP",convertedToColouredImg)
      cv.WaitKey(1)


      #CHOOSE POINT on the image with the bestPoint Correspondance
      chosenLandmarkPoint=Point(bestPoint.x, bestPoint.y)

      #last best point chosen - for this side .. for this normal

      if DEBUG_LINES==1:
          cv.Circle(convertedToColouredImg, (chosenLandmarkPoint.x, chosenLandmarkPoint.y), 2, (255,255,255))
          cv.WaitKey(1)




      if STOP_AT_THE_END_OF_EVERY_LANDMARK==1:
          #FINAL best point chosen - for this iteration
          cv.Circle(convertedToColouredImg, (chosenLandmarkPoint.x, chosenLandmarkPoint.y), 1, (255,255,255))
          cv.NamedWindow("convertedToColouredImg", cv.CV_WINDOW_NORMAL)
          cv.ShowImage("convertedToColouredImg",convertedToColouredImg)
          cv.WaitKey(1)
      #print "bestPoint=%r"%(chosenLandmarkPoint)
      #cv.WaitKey()
      return chosenLandmarkPoint




class ActiveShapeModel:
  """
  """
  def __init__(self, shapes = []):
    self.shapes = shapes
    # Make sure the shape list is valid
    self.__check_shapes(shapes)
    # Create weight matrix for points
    #print "Calculating weight matrix..."
    self.w = self.__create_weight_matrix(shapes)
    # Align all shapes
    #print "Aligning shapes with Procrustes analysis..."
    self.shapes = self.__procrustes(shapes)
    #print "Constructing model..."
    # Initialise this in constructor
    (self.evals, self.evecs, self.mean, self.modes) = \
        self.__construct_model(self.shapes)

  def __check_shapes(self, shapes):
    """ Method to check that all shapes have the correct number of
    points """
    if shapes:
      num_pts = shapes[0].num_pts
      for shape in shapes:
        print num_pts
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
    shape_vectors = np.array([s.get_vector() for s in self.shapes])
    mean = np.mean(shape_vectors, axis=0, dtype=np.float64)

    # Move mean to the origin
    # FIXME Clean this up...
    mean = np.reshape(mean, (-1,2))
    min_x = min(mean[:,0])
    min_y = min(mean[:,1])

    #mean = np.array([pt - min(mean[:,i]) for i in [0,1] for pt in mean[:,i]])
    #mean = np.array([pt - min(mean[:,i]) for pt in mean for i in [0,1]])
    mean[:,0] = [x - min_x for x in mean[:,0]]
    mean[:,1] = [y - min_y for y in mean[:,1]]
    #max_x = max(mean[:,0])
    #max_y = max(mean[:,1])
    #mean[:,0] = [x/(2) for x in mean[:,0]]
    #mean[:,1] = [y/(3) for y in mean[:,1]]
    mean = mean.flatten()
    ##print mean

    # Produce covariance matrix
    cov = np.cov(shape_vectors, rowvar=0)
    # Find eigenvalues/vectors of the covariance matrix
    evals, evecs = np.linalg.eig(cov)

    # Find number of modes required to describe the shape accurately
    t = 0
    for i in range(len(evals)):
      if sum(evals[:i]) / sum(evals) < 0.99:
        t = t + 1
      else: break
    #print "Constructed model with %d modes of variation" % t
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
    a = shapes[0]
    trans = np.zeros((4, len(shapes)) )
    converged = False
    current_accuracy = sys.maxint
    while not converged:
      # Now get mean shape
      mean = self.__get_mean_shape(shapes)
      # Align to shape to stop it diverging
      mean = mean.align_to_shape(a, self.w)
      # Now align all shapes to the mean
      for i in range(len(shapes)):
        # Get transformation required for each shape
        trans[:, i] = shapes[i].get_alignment_params(mean, self.w)
        # Apply the transformation
        shapes[i] = shapes[i].apply_params_to_shape(trans[:,i])

      # Test if the average transformation required is very close to the
      # identity transformation and stop iteration if it is
      accuracy = np.mean(np.array([1, 0, 0, 0], dtype=np.float64) - np.mean(trans, axis=1, dtype=np.float64))**2
      # If the accuracy starts to decrease then we have reached limit of precision
      # possible
      if accuracy > current_accuracy: converged = True
      else: current_accuracy = accuracy
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

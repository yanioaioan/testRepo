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
import array

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

  def add_point(self, p):
    self.pts.append(p)
    self.num_pts += 1

  def transform(self, t):
    s = Shape([])
    for p in self.pts:
      s.add_point(p + t)
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
      x = self.pts[p_num+1].x - self.pts[p_num-1].x
      y = self.pts[p_num+1].y - self.pts[p_num-1].y
    
    mag = math.sqrt(x**2 + y**2)
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
    cv.NamedWindow("Shape Model", cv.CV_WINDOW_AUTOSIZE)
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
      #print "pt=%d,%d"%(pt.x,pt.y)
      cv.ShowImage("Shape Model",i)

  @staticmethod
  def show_modes_of_variation(model, mode):
    # Get the limits of the animation
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
      c = cv.WaitKey(10)
      if chr(255&c) == 'q': break

  @staticmethod
  def draw_model_fitter(f):
    cv.NamedWindow("Model Fitter", cv.CV_WINDOW_AUTOSIZE)
    
    #c = cv.WaitKey(10)
    #print "f.shape.pts",f.shape.pts
    #c = cv.WaitKey(10)
    
    # Copy image
    i = cv.CreateImage(cv.GetSize(f.image), f.image.depth, 3)
    cv.Copy(f.image, i)
    for pt_num, pt in enumerate(f.shape.pts):
      # Draw normals
      cv.Circle(i, (int(pt.x), int(pt.y)), 1, (255,255,0), -1)
      print "pt=%d,%d"%(pt.x,pt.y)
    #Draw the original targeted image with cyan landmark points
    cv.ShowImage("Shape Model",i)
    #cv.WaitKey()
    #print 'STAGE - key pressed\n'

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
        #print "FFFFFfirst_line=%s"%first_line
        num_pts = int(first_line)
      for line in fh:
        if not line.startswith("}"):
          pt = line.strip().split()
          #print "line",line
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

class ModelFitter:
  """
  Class to fit a model to an image

  :param asm: A trained active shape model
  :param image: An OpenCV image
  :param t: A transformation to move the shape to a new origin
  """
  def __init__(self, asm, image, t=Point(0.0,0.0)):
    self.image = image
    self.g_image = []
    
    #creates 4 different size grayscale images
    for i in range(0,4):
      self.g_image.append(self.__produce_gradient_image(image, 2**i))
    
    print "G_IMAGE="    
    print self.g_image
    
    cv.WaitKey(1000)
    
        
    self.asm = asm
    # Copy mean shape as starting shape and transform it to origin
    self.shape = Shape.from_vector(asm.mean).transform(t)
    # And resize shape to fit image if required
    if self.__shape_outside_image(self.shape, self.image):
      self.shape = self.__resize_shape_to_fit_image(self.shape, self.image)

    #print "asm.mean",asm.mean
    scale=1
    
    
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
    #Hence we load the ith image, we convert it to grayscale, and permorm sampling along the normal to build a gray-level profile for it
    #We then do that for all of the training images in the set.
    #And so we end up having formed gray-level profiles for all of them..from which we can derive the coresponding covariance matrix for each of the images and plug it to the mahalanobis equation.
    
    #Need to have-create images corresponding to shape1,2,3..... #.pts
    gaussianMatrix=np.array([
							[0.000158,	0.000608,   0.00183,	0.004312,	0.007953,	0.011483,	0.012978,	0.011483,	0.007953,	0.004312,	0.00183,	0.000608,	0.000158],
							[0.000291,	0.001121,	0.003375,	0.007953,	0.014669,	0.021179,	0.023938,	0.021179,	0.014669,	0.007953,	0.003375,	0.001121,	0.000291],
							[0.000421,	0.001618,	0.004873,	0.011483,	0.021179,	0.030579,	0.034561,	0.030579,	0.021179,	0.011483,	0.004873,	0.001618,	0.000421],
							
							[0.000476,	0.001829,	0.005508,	0.012978,	0.023938,	0.034561,	0.039062,	0.034561,	0.023938,	0.012978,	0.005508,	0.001829,	0.000476],
							
							[0.000421,	0.001618,	0.004873,	0.011483,	0.021179,	0.030579,	0.034561,	0.030579,	0.021179,	0.011483,	0.004873,	0.001618,	0.000421],
							[0.000291,	0.001121,	0.003375,	0.007953,	0.014669,	0.021179,	0.023938,	0.021179,	0.014669,	0.007953,	0.003375,	0.001121,	0.000291],
							[0.000158,	0.000608,	0.00183,	0.004312,	0.007953,	0.011483,	0.012978,	0.011483,	0.007953,	0.004312,	0.00183,	0.000608,	0.000158]
							])
	
    print "gaussianMatrix=%s"%(gaussianMatrix)
    
    #iterate the gaussian distribution matrix
    for (x,y), value in np.ndenumerate(gaussianMatrix):
		print "%d, %d =%s"%(x,y,gaussianMatrix[x][y])
		#cv.WaitKey()
        
        
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
    IthImage_NormalizedtmpLandmarkDerivativeIntensityVector=[]


    #for all 3 images' vectors in the training set
    testImageNameCounter=0
    for s in self.asm.shapes:

        #add another list of derivative profile vectors for all landmarks which corresponds to each image
        IthImage_NormalizedtmpLandmarkDerivativeIntensityVector.append([])

        #1)for each image, convert to grayscale, iterate over the  corresponding points, and multiply profile pixel intensity with corresponding kernel gaussian distr. matrix element,
        #..then get the average of them for each tangential search profile and store it in the gith position of the g vector of elements along the whisker.

        #convert this test image to grayscale
        test_image = []
        testImageNameCounter=testImageNameCounter+1
        test_grey_image = cv.LoadImage("grey_image_"+str(testImageNameCounter)+".jpg")
        test_image.append(self.__produce_gradient_image(test_grey_image, 2**0))

        cv.WaitKey()
        print "testing image=%s"%("grey_image_"+str(testImageNameCounter)+".jpg")
        cv.WaitKey()

        #get a vector of all landmark points of each image
        oneTrainingImageVector=Shape.from_vector(s.get_vector())

        for i,p in enumerate(oneTrainingImageVector.pts):

            #this image's current landmark point; for each of these landmarks calculate a 2d windows search profile
            #x=(p.x)
                        #y=(p.y)


            #create a list of Shapes - the g profiles for each landmark point
            currentLandmarkProfiles = []

            #store point
            p = p

            #get normal to the point
            norm = self.shape.get_normal_to_point(i)

            print "\n\n\n\n\n  !!!!!!!!!!!!!!!!!!!!!! New landmark point SEARCH PROFILE calculation !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

            tmpLandmarkDerivativeIntensityVector=[]
            currentLandmarkProfilesNormalizedDerivativesVector=[]

            #along normal (whisker)
            for t in drange(-3, 4, 1):

                # Normal to normal...
                #print "norm",norm

                tmpProfileIntensityVector = []
                tmpProfileDerivativeIntensityVector = []


                # Look 6 pixels to each side along the whisker normal too (seach profile)
                for side in range(-6,7):#tangent width

                    new_p = Point(p.x + side*-norm[1], p.y + side*norm[0])#why ..the other way around? best way to form the search window 12 x 7 = 84 pixels wide window

                    x = int((norm[0]*t + new_p.x))#*math.sin(t*(math.pi/180)))
                    y = int((norm[1]*t + new_p.y))#*math.cos(t*(math.pi/180)))

                    #(equation 11 of the 'Subspace Methods for Pattern Recognition in Intelligent Environment' book)
                    scale=0


                    #add to g (the inensity value of next pixel along the perpendicular to the whisker.. search profile)
                    gradientIntensity=0#test_image[scale][y-1,x-1] * gaussianMatrix[t+3][side+6]

                    #Add weights properly based on gaussian distribution generated matrix
                    print "gaussianMatrix[t+3][side+6]=%s"%(gaussianMatrix[t+3][side+6])

                    #store gray-level profile in tmpProfileIntensityVector ..g vector
                    tmpProfileIntensityVector.append(gradientIntensity)

                print "t=%d along the whisker:\n , tmpProfileIntensityVector:%s"%(t,tmpProfileIntensityVector)
                #cv.WaitKey()


                #Landmark Search Profile with Weights was calculated
                print "Landmark Search Profile with Weights was calculated\n"
                print "tmpProfileIntensityVector:%s\n"%(tmpProfileIntensityVector)

                sumGith=0
                averageGith=0
                #now average each gith and save it to currentLandmarkProfiles vector
                for i in tmpProfileIntensityVector:
                    sumGith=sumGith + i

                averageGith=sumGith/len(tmpProfileIntensityVector)
                print "averageGith=%s"%(averageGith)
                #cv.WaitKey()
                currentLandmarkProfiles.append(averageGith)
                print "currentLandmarkProfiles=%s"%(currentLandmarkProfiles)
                #cv.WaitKey()

            #Now Time for the derivative of this currentLandmarkProfiles vecotr

            #(equation 12 of the 'Subspace Methods for Pattern Recognition in Intelligent Environment' book)
            normalizedBy=0
            for intensity in range(len(currentLandmarkProfiles)-1):

                #calculate derivative profile intensity g_image[i] - g_image[i-1]...and so on.... dg vector
                #....................
                #....................
                difference=currentLandmarkProfiles[intensity+1]-currentLandmarkProfiles[intensity]

                print "%f-%f"%(currentLandmarkProfiles[intensity+1],currentLandmarkProfiles[intensity])

                #store derivative gray-level profile vector
                tmpLandmarkDerivativeIntensityVector.append( difference )

                normalizedBy=normalizedBy+difference

            print "tmpLandmarkDerivativeIntensityVector: %s"%(tmpLandmarkDerivativeIntensityVector)
            print "normalizedBy: %s"%(normalizedBy)
            #cv.WaitKey()

            #(equation 13 of the 'Subspace Methods for Pattern Recognition in Intelligent Environment' book)
            #normalize tmpProfileDerivativeIntensityVector
            NormalizedtmpLandmarkDerivativeIntensityVector=[]
            for x in tmpLandmarkDerivativeIntensityVector:
                if normalizedBy!=0:
                    x = x / normalizedBy
                    NormalizedtmpLandmarkDerivativeIntensityVector.append(x)
                else:
                    x = x / 1
                    NormalizedtmpLandmarkDerivativeIntensityVector.append(x)

            print "NormalizedtmpLandmarkDerivativeIntensityVector %s\n"%(NormalizedtmpLandmarkDerivativeIntensityVector)
            #cv.WaitKey()

            #now store the Normalized tmp Landmark Derivative Intensity Vector for this image for this landmark
            # and move onto caclulating the NormalizedtmpLandmarkDerivativeIntensityVector for this image for the next landmark

            #This vector contains : for each image, all normalized derivative profile of ALL ith's image's landmarks
            print "image=%d\n"%(testImageNameCounter-1)
            IthImage_NormalizedtmpLandmarkDerivativeIntensityVector[testImageNameCounter-1].append(NormalizedtmpLandmarkDerivativeIntensityVector)
            print "Here the following vector is filled  with the  NormalizedtmpLandmarkDerivativeIntensityVector \nor each of the landmarks for this ith image in the training set"
            print "IthImage_NormalizedtmpLandmarkDerivativeIntensityVector %s\n"%(IthImage_NormalizedtmpLandmarkDerivativeIntensityVector[testImageNameCounter-1])
            #cv.WaitKey()





    '''
    #LandMark X Derivatives Vecotr Out of All Images In The Training Set
    LandMarkX_DerivativesVectorSum=nparray()

    imageCounter=0
    landmarkCounter=0
    # Loop over rows. (each tested image)
    for row in IthImage_NormalizedtmpLandmarkDerivativeIntensityVector:
        imageCounter=imageCounter+1
        print "Training Image:%d"%(imageCounter)

        # Loop over columns. (each landmark's normalized derivative profile)
        landmarkCounter=0
        for column in row:
            landmarkCounter=landmarkCounter+1
            print "derivative profile: landmark %d = %s"%(landmarkCounter,column)

            #cv.WaitKey()
        print("\n")
    cv.WaitKey()




    imageCounter=0
    # Loop over rows. (each tested image)
    for row in IthImage_NormalizedtmpLandmarkDerivativeIntensityVector:
        imageCounter=imageCounter+1
        print "Training Image:%d"%(imageCounter)

        # Loop over columns. (each landmark's normalized derivative profile)
        landmarkCounter=0
        for column in row:
            landmarkCounter=landmarkCounter+1
            print "derivative profile: landmark %d = %s"%(landmarkCounter,column)

            #create an element for each landmark added,
            LandMarkX_DerivativesVectorSum[landmarkCounter]=LandMarkX_DerivativesVectorSum[LandMarkX_DerivativesVector] + column
            print

            #cv.WaitKey()
        print("\n")

    '''


#print ("x=%d,y=%d")%(x,y)
#Gives the landmark points of 1 image out of the training set
#print "Gives the landmark points of this-each image out of the training set"
#print ("oneTrainingImageVector=%s")%(oneTrainingImageVector.pts)
#cv.WaitKey()





    
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

		print "\n\n\n\n\n  !!!!!!!!!!!!!!!!!!!!!! New landmark point mahalanobis calculation !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

		currentLandmarkProfilesNormalizedDerivativesVector=[]
		
		#along normal (whisker)
		for t in drange(-3, 3, 1):
			# Normal to normal...
			#print "norm",norm

			
			#print "p",(p.x,p.y)
			#print "new_p",(new_p)
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
			
			print "tmpProfileIntensityVector:%s"%tmpProfileIntensityVector
			
						
			#(equation 12 of the 'Subspace Methods for Pattern Recognition in Intelligent Environment' book)
			normalizedBy=0	
			for intensity in range(len(tmpProfileIntensityVector)-1):
			
				#calculate derivative profile intensity g_image[i] - g_image[i-1]...and so on.... dg vector
				#....................
				#....................	
				difference=tmpProfileIntensityVector[intensity+1]-tmpProfileIntensityVector[intensity]			
				
				print "%d-%d"%(tmpProfileIntensityVector[intensity+1],tmpProfileIntensityVector[intensity])
				
				#store derivative gray-level profile vector
				tmpProfileDerivativeIntensityVector.append( difference )
				
				normalizedBy=normalizedBy+difference 
			
			print "tmpProfileDerivativeIntensityVector: %s"%(tmpProfileDerivativeIntensityVector)
			print "normalizedBy: %s"%(normalizedBy)
			
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
				
			print "NormalizedTmpProfileDerivativeIntensityVector %s\n"%(NormalizedTmpProfileDerivativeIntensityVector)	
			cv.WaitKey(10)	
				
				
			#fill in each normalized derivative profile vector for this landmark and create the currentLandmarkProfilesNormalizedDerivativesVector
			#essentially containing all normalized derivative profiles for current landmark point, of which we should calculate the mean
			
			currentLandmarkProfilesNormalizedDerivativesVector.append(NormalizedTmpProfileDerivativeIntensityVector)
			print "currentLandmarkProfilesNormalizedDerivativesVector total: %s"%(len(currentLandmarkProfilesNormalizedDerivativesVector))
			print "currentLandmarkProfilesNormalizedDerivativesVector: %s"%(currentLandmarkProfilesNormalizedDerivativesVector)
			
			#(equation 14 of the 'Subspace Methods for Pattern Recognition in Intelligent Environment' book)
		
		# now sum all currentLandmarkProfilesNormalizedDerivativesVector elements together and divide by len(currentLandmarkProfilesNormalizedDerivativesVector)
		# which is the total number of dormalized derivative profile vectors for this landmark point
		#/////////////currentLandmarkProfilesNormalizedDerivativesVector/len(currentLandmarkProfilesNormalizedDerivativesVector)
	'''
			
    '''		
			#print "Gmean",Gmean.pts
			currentLandmarkProfiles.append(tmpProfileIntensityVector)
			
		for j in range(len(currentLandmarkProfiles)):
			print "profile %d with %d points: %s"%(j, len(currentLandmarkProfiles[j]), currentLandmarkProfiles[j])
		cv.WaitKey(1000)	
	 '''
	    
	    
	    
	    
		#print "total profiles",len(profiles)
		#cv.WaitKey(1000)
		#now calculate the derivative profile dg
		
    '''
		for i in range(len(profiles)):		    
			print "i",profiles[0].pts
			cv.WaitKey()
    '''
		
		#now calculate the meanG profile for this landmark , as well as the covariance matrix for this landmark
		
		
		
		#print "profile number:",len(profiles)
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
    if min_x > i.width: min_x = 0
    if min_y > i.height: min_y = 0
    ratio_x = (i.width-min_x) / (max_x - min_x)
    ratio_y = (i.height-min_y) / (max_y - min_y)
    new = Shape([])
    for pt in s.pts:
      new.add_point(Point(pt.x*ratio_x if ratio_x < 1 else pt.x, \
                          pt.y*ratio_y if ratio_y < 1 else pt.y))
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
    print "width=%d,height=%d"%(width,height)
    
    #Save out the image and move on to the new cv2 way of doing things so as to gray scale it
    cv.SaveImage("InputToGrayScale.jpg", i) 
    InputToGrayScaleImg=cv2.imread("InputToGrayScale.jpg")
    cv.WaitKey()
        
    grey_image = np.zeros((width,height,1), np.uint8)
    
    print "GREY IMAGE="    
    print grey_image
    cv.WaitKey(1000)
    
    width=width/scale
    height=height/scale
     
    grey_image_small = np.zeros((width,height,1), np.uint8)
    
    grey_image = cv2.cvtColor(InputToGrayScaleImg, cv2.COLOR_BGR2GRAY)
    #cv.Resize(grey_image, grey_image_small)#, interpolation=cv.CV_INTER_NN)
    #cv.Resize(grey_image_small, grey_image)#, interpolation=cv.CV_INTER_NN)
    #grey_image = cv2.GaussianBlur(grey_image,(3,3),0)
  
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
    

  def do_iteration(self, scale):
    """ Does a single iteration of the shape fitting algorithm.
    This is useful when we want to show the algorithm converging on
    an image

    :return shape: The shape in its current orientation
    """

    '''...a shape in the training set can be approximated using the mean shape 
	and a weighted sum of the deviations obtained by x=(mean) + P*b...'''
    

    img = cv.CreateImage(cv.GetSize(self.image), self.g_image[scale].depth, 1)
    cv.Copy(self.g_image[scale], img)

    # Build new shape from max points along normal to current
    # shape
    s = Shape([])

	#for each of the points of the current shape
    for i, pt in enumerate(self.shape.pts):
      max_pt=self.__get_max_along_normal(i, scale)
      s.add_point(max_pt)


      #show max points chosen along normal profile sampled -6..-6 pixels across
      #BEFORE ALIGNMENT
      maxpX=(int)(pt.x)
      maxpY=(int)(pt.y)
      cv.Circle(img, (maxpX,maxpY), 1, (255,255,255))
      cv.NamedWindow("Scale", cv.CV_WINDOW_AUTOSIZE)
      cv.ShowImage("Scale",img)
      
	
    '''align this s shape to mean with a weigted matrix'''

    new_s = s.align_to_shape(Shape.from_vector(self.asm.mean), self.asm.w)
    
    print (self.asm.evals[0])

    #calculate new shape - update the mean based on x=(mean) + P*b
    var = new_s.get_vector() - self.asm.mean
    new = self.asm.mean
    for i in range(len(self.asm.evecs.T)):
      b = np.dot(self.asm.evecs[:,i],var)
      
      #ADDITION of this IF statement CAUSE IT CRASHS IF  self.asm.evals[i] is < 0
      if self.asm.evals[i] > 0:
          max_b = 2*math.sqrt(self.asm.evals[i])
          b = max(min(b, max_b), -max_b)
          new = new + self.asm.evecs[:,i]*b

    
    #align the new shape to the already existing (aligned to the mean) s
    self.shape = Shape.from_vector(new).align_to_shape(s, self.asm.w)
    
    print self.shape.pts


    #show max points chosen along normal profile sampled -6..-6 pixels across
    #AFTER ALIGNMENT
    for i, pt in enumerate(self.shape.pts):
        maxpX=(int)(pt.x)
        maxpY=(int)(pt.y)
        cv.Circle(img, (maxpX,maxpY), 1, (255,255,255))
        cv.NamedWindow("Scale", cv.CV_WINDOW_AUTOSIZE)
        cv.ShowImage("Scale",img)


  def __get_max_along_normal(self, p_num, scale):
    """ Gets the max edge response along the normal to a point

    :param p_num: Is the number of the point-landmark in the shape
    """

	#calculate the normal corresponding to point p_num of the mean shape
    norm = self.shape.get_normal_to_point(p_num)
    print "norm\n",norm
    
    
    #get p, each of the 2d landmark points of the mean shape in the training model
    p = self.shape.pts[p_num]#1st point ..2nd point.. and so on
    
    print "Points of the mean shape of the Trained Set",self.shape.pts

    # Find extremes of normal within the image (for each of the points of the mean)
    # Test x first
    min_t = -p.x / norm[0]
    if p.y + min_t*norm[1] < 0:
      min_t = -p.y / norm[1]
    elif p.y + min_t*norm[1] > self.image.height:
      min_t = (self.image.height - p.y) / norm[1]

    # X first again
    max_t = (self.image.width - p.x) / norm[0]
    if p.y + max_t*norm[1] < 0:
      max_t = -p.y / norm[1]
    elif p.y + max_t*norm[1] > self.image.height:
      max_t = (self.image.height - p.y) / norm[1]

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

    img = cv.CreateImage(cv.GetSize(self.image), self.g_image[scale].depth, 1)
    cv.Copy(self.g_image[scale], img)


    #cv.Circle(img, \
    #    (int(norm[0]*min_t + p.x), int(norm[1]*min_t + p.y)), \
    #    5, (0, 0, 0))
    #cv.Circle(img, \
    #    (int(norm[0]*max_t + p.x), int(norm[1]*max_t + p.y)), \
    #    5, (0, 0, 0))

    # Scan over the whole line
    max_pt = p
    max_edge = 0
    

    # Now check over the vector
    #v = min(max_t, -min_t)
    #for t in drange(min_t, max_t, (max_t-min_t)/l):
    search = 20+scale*10

    counter=-1

    #along the whisker
    for side in range(-3,3):#along normal profile
        # Normal to normal...
        #print "norm",norm
        new_p = Point(p.x + side*-norm[1], p.y + side*norm[0])#why ..the other way around? best way to form the search window 12 x 7 = 84 pixels wide window
        #print "p",(p.x,p.y)
        #print "new_p",(new_p)


        #for t in drange(min_t, max_t, (max_t-min_t)/l):

        # Look 6 pixels to each side too
        for t in drange(-3, 3, 1):#tangent

            horizontalProfileFormed = Shape([])
            counter=1


            #distributed = drange(-search if -search > min_t else min_t, search if search < max_t else max_t , 1)
            #for t in distributed:

            counter=counter+1


            x = int((norm[0]*t + new_p.x))#*math.sin(t*(math.pi/180)))
            y = int((norm[1]*t + new_p.y))#*math.cos(t*(math.pi/180)))

            #print "x=%d, y=%d"%(x,y)
            #cv.WaitKey()

            #a g profile for each point along the normal-whisker is calculated
            horizontalProfileFormed.add_point(Point(x,y))

            if x < 0 or x > self.image.width or y < 0 or y > self.image.height:
                continue
            #cv.Circle(img, (x, y), 1, (100,100,100))

            #print x, y, self.g_image.width, self.g_image.height

            #print "g_image[scale][y-1, x-1]",self.g_image[scale][y-1,x-1]

            print "TEST.... G_IMAGE:",self.g_image[scale][y-1, x-1]

            '''
            if self.g_image[scale][y-1, x-1] > max_edge:#choose the pixel amongst the 12 pixels checked with the greater value(indicating an edge)
                max_edge = self.g_image[scale][y-1, x-1]
                max_pt = Point(new_p.x + t*norm[0], new_p.y + t*norm[1])
            '''

            im_array = np.asarray( self.g_image[scale][:,:] )
            #print im_array
            #cv.WaitKey()

            ###########################################
            ###########################################
            #self.asm.mean

            #y-1 x-1 : keep the pixel inside limits
            if im_array[y-1,x-1] > max_edge:#choose the pixel amongst the 12 pixels checked with the greater value(indicating an edge)
                max_edge = im_array[y-1,x-1]
                max_pt = Point(new_p.x + t*norm[0], new_p.y + t*norm[1])


            #visualization of the each profile separately
            '''
            for point in horizontalProfileFormed.pts:
              cv.Circle(img, (int(point.x), int(point.y)), 1, (255,255,255))
              print "profile number %d"%(counter)
            '''


            #fitting visualization of the whole window of profiles
            #for point in self.shape.pts:
            #  cv.Circle(img, (int(point.x), int(point.y)), 3, (255,255,255))

            #cv.WaitKey()





            cv.NamedWindow("Scale", cv.CV_WINDOW_AUTOSIZE)
            cv.ShowImage("Scale",img)
            #c=cv.WaitKey()
            #print "EXIT when ESC keycode is pressed=%d"%(c)
            #if c == 1048603:
            #    exit()

    return max_pt

class ActiveShapeModel:
  """
  """
  def __init__(self, shapes = []):
    self.shapes = shapes
    # Make sure the shape list is valid
    self.__check_shapes(shapes)
    # Create weight matrix for points
    print "Calculating weight matrix..."
    self.w = self.__create_weight_matrix(shapes)
    
    print 'Shapes BEFORE Weighted Procrustes'
    print self.shapes[0].pts
    print "\n"
    print self.shapes[1].pts
    print "\n"
    print self.shapes[2].pts
    print "\n"
    
    # Align all shapes
    print "Aligning shapes with Procrustes analysis..."
    self.shapes = self.__procrustes(shapes)
    
    print 'Shapes AFTER Weighted Procrustes'
    print self.shapes[0].pts
    print "\n"
    print self.shapes[1].pts
    print "\n"
    print self.shapes[2].pts
    print "\n"
    
    
   
    
    
    print "Constructing model..."
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
    print ("3 training images tested = %s")%(len(shape_vectors))
    print ("3 training images tested = %s")%((shape_vectors))    
    
    
    print ("each image contains %s landmark points")%(len(s.get_vector())/2)#61 sets of coordinates [x,y]    
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
	
	#the mean of the aligned shapes
    mean = np.mean(shape_vectors, axis=0)

    print "mean shape which is the center of the ellipsoidal Allowable Shape Domain - Before reshaping\n",mean
    print "\n"
	
    # Move mean to the origin
    # FIXME Clean this up...
    mean = np.reshape(mean, (-1,2))#turn mean array from 4x5 to 10x2 array
    min_x = min(mean[:,0])#get the min of the 1st row refering to the first X coordinate
    min_y = min(mean[:,1])#get the min of the 2nd row refering to the Second Y coordinate
    
    print "mean After reshaping\n",mean
    print "\n"
    
    
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
    #print mean
    
    
    print "The list of shapes (and their point coordinates:\n",shape_vectors
    print "\n\n"
    
        
    
    # Produce covariance matrix
    
    '''We attempt to model the shape in the Allowable Shape Domain..hence capture the relationships between position of individual landmark points'''
    '''The cloud of landmarks is approximately ellipsoidal, so we need to calculate its center (giving a mean shape and its major axes)'''
    '''Covariance indicates the level to which two variables vary together'''
            
    #shape_vectors=[(1,2),(3,4)]
    cov = np.cov(shape_vectors, rowvar=0)
    print "shape_vectors",shape_vectors
    #cv.WaitKey()
    print "cov\n",cov
    #cv.WaitKey()
    
    # Find eigenvalues/vectors of the covariance matrix
    evals, evecs = np.linalg.eig(cov)
   
    print "evals\n",evals
    print "sum(evals)\n",sum(evals)
    
    
    print "evecs\n",evecs
    
    
   
    #=0
    # Find number of modes required to describe the shape accurately
    t = 0
    for i in range(len(evals)):
	  
          print "sum(evals[:%f])=%f\n"%(sum(evals[:i]),sum(evals[:i]))
          #print "sum(evals[:%f])=%f\n"%(sum(evals),sum(evals))

	  #iterating through the list of evals, as soon as the sum  of evals so far divided by the total sum is >=0.99 then this is the number of modes we need
	  '''Choose the first largest eigenvalues..that represent a wanted percentage of the total variance, a.i. 0.99 or 99%.
	  Defines the proportion of the total variation one wishes to explain
	  (for instance, 0.98 for 98%)'''
		
  	  if sum(evals[:i]) < 1*sum(evals):
		#=c+1
                #print c
		t = t + 1
	  else: break
        
    
    print "Constructed model with %d modes of variation" % t
    
    #evecs = np.array([[1,2,3,4,5,6],[7,8,9,10,11,12]])
    #print evecs[:,:1]
    
    print "evals[:%d]\n%s"%(t,evals[:t])
    
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
    
    # print "TRANS ", trans
    
    converged = False
    current_accuracy = sys.maxint
    
    ml=[]
    lines_counter_written=1#contol flag
    lines_counter=0
    
    while not converged:
      # Now get mean shape
      mean = self.__get_mean_shape(shapes)
      
      #print "Calculated mean:",mean.pts
      
      # Align to shape to stop it diverging
      mean = mean.align_to_shape(a, self.w)#NORMALIZATION
            
      #print "new mean=",mean.pts
      
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
      if accuracy > current_accuracy: converged = True;  #print 'accuracy=',(accuracy);     print 'current_accuracy=',(current_accuracy);
      else: current_accuracy = accuracy; # print 'accuracy=',(accuracy);     print 'current_accuracy=',(current_accuracy);
    
    #target.close()
    print "Final Mean Shape Points=",mean.pts
    print "\n"
    
    #print "ACCURACY=%d, current_accuracy=%d ,  converged=%s"%(accuracy,current_accuracy, converged)
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

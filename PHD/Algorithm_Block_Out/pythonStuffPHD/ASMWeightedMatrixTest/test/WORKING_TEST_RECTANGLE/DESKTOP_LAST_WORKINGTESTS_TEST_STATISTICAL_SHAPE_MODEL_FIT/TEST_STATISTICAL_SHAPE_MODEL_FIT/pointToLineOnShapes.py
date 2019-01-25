import argparse
import cv2
from PyQt4 import QtGui
import sys,math
from scipy.spatial import distance
from testInitASM import PointsReader, Point
import cv
import numpy as np
import pointToLineDistance


asmModeShapeDir = PointsReader.read_directory(sys.argv[1])
theoriticalContourShapeDir = PointsReader.read_directory(sys.argv[2])

def get_normal_to_point(shape, p_num):
    # Normal to first point
    x = 0; y = 0; mag = 0
    if p_num == 0:
      #original
      '''
      x = shape.pts[1].x - shape.pts[0].x
      y = shape.pts[1].y - shape.pts[0].y
      '''
      #landmark position dependent normal calculation

      x = shape.pts[1].x - shape.pts[-1].x
      y = shape.pts[1].y - shape.pts[-1].y

    # Normal to last point
    elif p_num == len(shape.pts)-1:
      #original
      '''
      x = shape.pts[-1].x - shape.pts[-2].x
      y = shape.pts[-1].y - shape.pts[-2].y
      '''

      #landmark position dependent normal calculation

      x = shape.pts[0].x - shape.pts[-2].x
      y = shape.pts[0].y - shape.pts[-2].y

    # Must have two adjacent points, so...
    else:
      x = shape.pts[p_num+1].x - shape.pts[p_num-1].x
      y = shape.pts[p_num+1].y - shape.pts[p_num-1].y
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




#shapesOfDir1 = PointsReader.read_directory(sys.argv[1])
#shapesOfDir2 = PointsReader.read_directory(sys.argv[2])

targetimg = cv.LoadImage("testBackground.png")
sumShapeDistanceOverAllFrames=0
imageLoaded = cv.LoadImage("testBackground.png")

#for each asm model shape and the equivalent theoritical contour shape
sumShapeallPointToLineDistanceALLSHAPES=0
for i in range(len(asmModeShapeDir)):

  shape1 = asmModeShapeDir[i]
  shape2 = theoriticalContourShapeDir[i]
  queriedPoint =0

  #iterate over asm model points,
  #for each asm model point, find best closest point distance
  sumShapeallPointToLineDistance=0
  for i, pt in enumerate(shape1.pts):

      if i==0:
        queriedPoint =  shape1.pts[0]
        previousPoint = shape1.pts[-1]
        nextPoint =     shape1.pts[1]

        theoriticalContourP1 = shape2.pts[-1]
        theoriticalContourP2 = shape2.pts[1]

      elif i == len(shape1.pts)-1:
        queriedPoint =  shape1.pts[-1]
        previousPoint = shape1.pts[-2]
        nextPoint =     shape1.pts[0]

        theoriticalContourP1 = shape2.pts[-2]
        theoriticalContourP2 = shape2.pts[0]

      else:
        queriedPoint =  shape1.pts[i]
        previousPoint = shape1.pts[i-1]
        nextPoint =     shape1.pts[i+1]

        theoriticalContourP1 = shape2.pts[i-1]
        theoriticalContourP2 = shape2.pts[i+1]

      queriedPoint = (queriedPoint.x, queriedPoint.y)
      previousPoint = (previousPoint.x, previousPoint.y)
      nextPoint = (nextPoint.x, nextPoint.y)

      #load image from scratch at every point pair testing
      imageLoaded = cv.LoadImage("testBackground.png")

      #the list of intersected points derived from testing normal line
      #of queried point (defined by prev and & next adjacent points to it)
      #and theoritical contour line segments

      intersectedPoints = []

      #iterate over theoritical contour points      
      #for j in range(len(shape2.pts)-1):
      #load image from scratch at every point pair testing
      #imageLoaded = cv.LoadImage("testBackground.png")

      print 'theoritical point pair testing'
      #theoriticalContourP1 = shape2.pts[i]
      #theoriticalContourP2 = shape2.pts[i+1]

      #For the equivalent 'green' segment of the theoritical contour


      theoriticalContourP1 = (theoriticalContourP1.x, theoriticalContourP1.y)
      theoriticalContourP2 = (theoriticalContourP2.x, theoriticalContourP2.y)

      #test 2 points
      intersectedPoints=pointToLineDistance.test(imageLoaded, intersectedPoints,queriedPoint,previousPoint,nextPoint,theoriticalContourP1,theoriticalContourP2)

      #cv.WaitKey(1)

      print "intersectedpoints"+str(intersectedPoints)
      print "len of intersectedpoints"+str(len(intersectedPoints))
      #cv.WaitKey()

      #Find closest point out of a list of points intersectedPoints
      minDistance=10000000
      bestIndex=-1
      for ind in range(len(intersectedPoints)):

          if intersectedPoints[ind] != None:

            x1=intersectedPoints[ind][0]
            y1=intersectedPoints[ind][1]
            x2=queriedPoint[0]
            y2=queriedPoint[1]

            dst=dist = math.hypot(x2 - x1, y2 - y1)
            if dst < minDistance:
                minDistance = dst
                print "minDistance",minDistance
                print "ind",ind
                bestIndex=ind

      closestIntersectedPointAlongNormal = intersectedPoints[bestIndex]

      print "intersectedPoints ind",bestIndex

      cv.Circle(imageLoaded, (int(closestIntersectedPointAlongNormal[0]),int(closestIntersectedPointAlongNormal[1]) ) ,4  ,(255,255,255),1)
      #cv.Circle(imageLoaded, (int(queriedPoint[0]),int(queriedPoint[1]) ) ,4  ,(0,255,255),1)
      cv.NamedWindow("landmarks On Target Image", cv.CV_WINDOW_NORMAL)
      cv.ShowImage("landmarks On Target Image", imageLoaded)
      #dst=DistancePointLine(10,3, 6,4, 8,6)
      print "minDistance",minDistance
      sumShapeallPointToLineDistance+=minDistance

      print "sumShapeallPointToLineDistance",sumShapeallPointToLineDistance
      print "iter",i


  print "sumShapeallPointToLineDistance",sumShapeallPointToLineDistance
  cv.WaitKey()

  sumShapeDistanceOverAllFrames+=sumShapeallPointToLineDistance

print "sumShapeallPointToLineDistanceALLSHAPES",sumShapeDistanceOverAllFrames
cv.WaitKey()

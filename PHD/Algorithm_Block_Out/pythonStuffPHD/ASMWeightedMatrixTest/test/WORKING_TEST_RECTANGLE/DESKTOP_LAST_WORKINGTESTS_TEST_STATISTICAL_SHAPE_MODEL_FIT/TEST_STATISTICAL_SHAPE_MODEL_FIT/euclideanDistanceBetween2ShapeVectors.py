# import the necessary packages
import argparse
import cv2
from PyQt4 import QtGui
import sys
from scipy.spatial import distance
from testInitASM import PointsReader, Point
import cv
import numpy as np
'''
"need to call script like: python euclideanDistanceBetween2ShapeVectors.py path/to/shape1 path/to/shape2
 and so on.."
'''

#Testing on simple 2d numpy arrays
'''
a=[(207.404550, 130.339398), (203.439719, 137.335059), (202.477555, 147.312265), (205.508385, 158.272758), (208.539216, 169.233251), (211.563005, 178.199006), (211.587648, 185.180586), (210.618443, 193.163055), (209.642197, 199.150786), (214.639601, 202.125290), (220.620292, 201.106799), (226.597462, 199.090939), (233.572001, 197.071559), (240.553581, 197.046916), (247.535161, 197.022273), (255.517630, 197.991478), (262.499209, 197.966835), (271.479047, 198.932520), (279.461515, 199.901725), (287.443984, 200.870930), (297.421190, 201.833094), (298.397436, 195.845362), (296.374536, 187.873455), (296.353413, 181.889244), (294.337553, 175.912073), (290.330477, 170.939312), (288.311097, 163.964773), (287.285565, 155.989345), (285.269705, 150.012175), (283.246805, 142.040268), (282.221273, 134.064840), (283.201039, 129.074477), (283.190478, 126.082371), (277.206267, 126.103493), (271.225576, 127.121985), (265.241364, 127.143107), (257.262416, 127.171271), (249.283468, 127.199434), (243.302777, 128.217925), (238.312414, 127.238159), (231.334354, 128.260170), (225.353664, 129.278662), (217.374715, 129.306825)]
#a=np.array(a).flatten()
print a
exit()

b=[(206.404550, 133.339398), (201.439719, 127.335059)]
a=np.array(a).flatten()
b=np.array(b).flatten()
print a
dst = distance.euclidean(a,b)
print dst
exit()
#cv.WaitKey()
'''



shapesOfDir1 = PointsReader.read_directory(sys.argv[1])
shapesOfDir2 = PointsReader.read_directory(sys.argv[2])

targetimg = cv.LoadImage("testBackground.png")
sumShapeDistanceOverAllFrames=0

for i in range(len(shapesOfDir1)):

    shape1 = shapesOfDir1[i]
    shape2 = shapesOfDir2[i]
    #print 'shape'+str(i)
    #print (shape1.pts)
    a=shape1.get_vector()
    b=shape2.get_vector()
    dst = distance.euclidean(a,b)
    print dst

    sumShapeDistanceOverAllFrames+=dst






    #print len(shape2.pts)

    '''
    for i,p in enumerate(shape.pts):
        tmpP = Point(p.x, p.y)
        cv.Circle(targetimg, ( int(tmpP.x), int(tmpP.y) ), 4, (0,0,255))
        cv.NamedWindow("landmarks On Target Image", cv.CV_WINDOW_NORMAL)
        cv.ShowImage("landmarks On Target Image", targetimg)
        #cv.WaitKey()
    '''
print "sumShapeDistanceOverAllFrames=%r"%(sumShapeDistanceOverAllFrames)
cv.WaitKey()


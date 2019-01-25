import numpy as np
from scipy.spatial import distance

import math 

def distances(xy1, xy2):
   d0 = np.subtract.outer(xy1[:,0], xy2[:,0])
   d1 = np.subtract.outer(xy1[:,1], xy2[:,1])
   
   return np.hypot(d0, d1)
   
   

xy1=np.array([[ 1, 1],[ 1, 2]])

xy2=np.array([[ 2, 2],[ 2, 4]])

print distances(xy1,xy2)

totalVertebraDist=0
PointDist=0
for elemnt in range (len(xy1)):
	msum=0
	for coord in range (len(xy1[0])):
		print "\nxy1[%f][%f]= %f - xy2[%f][%f]= %f"%(elemnt,coord, xy1[elemnt][coord], elemnt, coord, xy2[elemnt][coord])
		tmpSqrtDist=( (xy1[elemnt][coord] - xy2[elemnt][coord])**2 )
		print "tmpSqrtDist=%f"%(tmpSqrtDist)
		msum+=tmpSqrtDist
		print "msum=%f"%(msum)
	
	PointDist+=(msum)	
	print "--------\nPointDist=%f\n--------\n"%(PointDist)



totalVertebraDist=math.sqrt(PointDist)
print "\ntotal euclidean VertebraDist=%f"%(totalVertebraDist)


#alternative way of implementing 2d Euclidean Distance
from math import*
 
xy1=np.array([[ 1, 1],[ 1, 2]])
xy1=xy1.flatten()

xy2=np.array([[ 2, 2],[ 2, 4]])
xy2=xy2.flatten()


def euclidean_distance(x,y):
 
    return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))
 
print "euclidean_distance=%f"%(euclidean_distance(xy1,xy2))

###THE SMALLER THE DISTANCE, THE MORE SIMILAR THE VECTORS


#http://www.codehamster.com/2015/03/09/different-ways-to-calculate-the-euclidean-distance-in-python/

import numpy as np
import sys
import math
from scipy.spatial import distance

 
def euclidean4(vector1, vector2):
    ''' use scipy to calculate the euclidean distance. '''
    dist = distance.euclidean(vector1, vector2)
    return dist
 

print "euclidean3=%f"%( euclidean4(xy1, xy2) )


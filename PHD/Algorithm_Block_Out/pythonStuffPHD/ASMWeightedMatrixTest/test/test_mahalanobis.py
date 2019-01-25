import numpy as np
import scipy.spatial.distance as scidist
from numpy.linalg import inv

import cv

def test_mahalanobis():
<<<<<<< HEAD
        x = np.array([[1, 5], [ 4, 2]])
        y = np.array([[5, 7],  [6, 8]])
=======
        x = np.array([[1, 2], [ 3, 4]])
        y = np.array([[5, 6],  [7, 8]])
>>>>>>> 0db1f6eb0a2dfe00f213e3ae03fbbc096f6c772f
	print x
	print y

	

	#vi = np.array([[2.0, 1.0, 0.0, 2.0, 1.0, 0.0],[1.0, 2.0, 1.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 2.0, 1.0, 0.0], [2.0, 1.0, 0.0, 2.0, 1.0, 0.0],[1.0, 2.0, 1.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 2.0, 1.0, 0.0]])
	#print vi

	X = np.vstack((x,y))
	covMat=np.cov(x,y)
	#covMat=np.array(covMat)

	'''
	maskArray = np.zeros(shape=( 2,2 ), dtype=np.uint8 )
	mask=cv.fromarray( maskArray ) 
	
	vects=[]
	vects.append(cv.fromarray(np.asarray(x)))
	vects.append(cv.fromarray(np.asarray(y)))
	'''
	
	cov = cv.CreateMat(128,128,cv.CV_64FC1)
	avg = cv.CreateMat(1,128,cv.CV_64FC1)	
	
	
	cv.CalcCovarMatrix(cv.fromarray(np.array([1,2])),cov,avg,cv.CV_COVAR_NORMAL)
	
	print "\ncovMat=%s\n"%(covMat)
	invmat = inv(covMat)
	#print invmat


	
	dist=scidist.mahalanobis(x.flatten(), y.flatten(), covMat)#np.linalg.inv(covMat)
	#assert_almost_equal(dist, np.sqrt(6.0))
        print dist

	'''
	a = np.array([[ 1, 2, 3, 4 ],
				  [ 5 ,6 ,7, 8 ],
				  [ 9, 10, 11, 12 ],
				  [ 13, 14, 15, 16 ]])
	'''
				  
	a = np.array(
				[[ 1, 2, 3, 4 ],
				[ 5 ,6 ,7, 8 ],
				[ 9, 10, 11, 12],
				[ 13, 14, 15, 16]])
		  
				  
				  
	print "\na=%s"%(a)
	ainv = inv(a)
	print ainv

       
#test_mahalanobis()



# as column vectors
<<<<<<< HEAD
x = np.array([1, 5,  4, 2])
y = np.array([5, 7,  6, 8])
=======
#x = np.array([[1, 2], [ 3, 4]])
#y = np.array([[2, 1],  [1, 2]])

x = np.array([[1, 1],  [1, 2]])
y = np.array([[2, 3],  [6, 8]])
>>>>>>> 0db1f6eb0a2dfe00f213e3ae03fbbc096f6c772f


def MahalanobisDist(x, y):
    covariance_xy = np.cov(x,y, rowvar=0)
    inv_covariance_xy = np.linalg.inv(covariance_xy)
    xy_mean = np.mean(x),np.mean(y)
    x_diff = np.array([x_i - xy_mean[0] for x_i in x])
    y_diff = np.array([y_i - xy_mean[1] for y_i in y])
    diff_xy = np.transpose([x_diff, y_diff])
    
    md = []
    for i in range(len(diff_xy)):
        md.append(np.sqrt(np.dot(np.dot(np.transpose(diff_xy[i]),inv_covariance_xy),diff_xy[i])))
    return md
    
md=MahalanobisDist(x.flatten(),y.flatten())
print md

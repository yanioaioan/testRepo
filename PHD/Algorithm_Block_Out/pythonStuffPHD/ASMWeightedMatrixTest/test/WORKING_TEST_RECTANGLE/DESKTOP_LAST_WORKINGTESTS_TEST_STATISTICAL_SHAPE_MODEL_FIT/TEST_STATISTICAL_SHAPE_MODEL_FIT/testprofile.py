import numpy as np
import scipy.spatial

p1=np.array([[5,4,5],[6,5,4]])
p2=np.array([[4,5,6],[5,6,5]])
l=[]
l.append(p1)
l.append(p2)
mean=np.mean(np.array(p1), axis=0)

covarOfPointX=np.cov(np.array(p1), rowvar=0)
invcovar=np.linalg.pinv(covarOfPointX)

y1=np.array([[10,9,7]])
y1 = np.reshape(y1,(-1,1))
mean = np.reshape(mean,(-1,1))


print y1.shape
print mean.shape

#euclidean
#print np.linalg.norm(y1-mean)

print "covarOfPointX=%r"%(covarOfPointX)
print "invcovar=%r"%(invcovar)
print "mean=%r"%(mean)
print "y1=%r"%(y1)

print "sizeof invcovar=%r"%(invcovar.size)
print "sizeof y1.size=%d"%(y1.size)
print "sizeof mean.size=%d"%(mean.size)

tmpMah1=scipy.spatial.distance.mahalanobis( y1, mean, invcovar)


y2=np.array([[5,4,6]])
y2 = np.reshape(y2,(-1,1))

#euclidean
#print np.linalg.norm(y2-mean)
tmpMah2=scipy.spatial.distance.mahalanobis( y2, mean, invcovar)

print "mean",mean
print "tmpMah1=",tmpMah1
print "tmpMah2=",tmpMah2

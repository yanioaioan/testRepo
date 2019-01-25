
import cv,array
import numpy as np
import scipy.spatial.distance;

cv.NamedWindow("Shape Model", cv.CV_WINDOW_AUTOSIZE)
g_image = cv.CreateImage((2, 3), 8, 1)
#print "TEST.... G_IMAGE:",g_image[0][0,0]
'''
r = 255#randint(0, 255)
g = 255#randint(0, 255)
b = 0#randint(0, 255)

for i in range(20):
    r = 255#randint(0, 255)
    g = 255#randint(0, 255)
    b = 255#randint(0, 255)
    #cv.Circle(g_image, (i,i), 1, (b,g,r), -1)
'''    
#im_array = np.asarray( g_image )
#print im_array
im_array = np.asarray( g_image[:,:] )

im_array[0,0]=40
im_array[0,1]=50
#print im_array


x=np.random.normal(size=2)
print "x"
y=np.random.normal(size=2)
print "y",y
z = np.vstack((x, y))
print "z",z
c = np.cov(z.T)
print "cov",c
    



'''
s = np.array([[25,3], [133,6], [103,5], [113,2], [121,5]])


    
covar = np.cov(s, rowvar=0);

print "print covar",covar


if(s.shape[1:2]==(1,)):#if a 1 column array
    print covar
    invcovar = np.linalg.inv(covar.reshape(1,1))
    print 1
    print covar
else:
    invcovar = np.linalg.inv(covar)
    print 2
    print "print invcovar",invcovar
    
print scipy.spatial.distance.mahalanobis(s[0],s[1],invcovar)
'''


#cv.ShowImage("Shape Model",g_image)
#cv.WaitKey(0)

#IPLIMAGEimagearray=array.array('B', g_image.tostring())
#print IPLIMAGEimagearray

'''
import cv2.cv as cv

im=cv.LoadImage('imageTarget.png', cv.CV_LOAD_IMAGE_COLOR)

# Laplace on a gray scale picture
gray = cv.CreateImage(cv.GetSize(im), 8, 1)
cv.CvtColor(im, gray, cv.CV_BGR2GRAY)

aperture=3

dst = cv.CreateImage(cv.GetSize(gray), cv.IPL_DEPTH_32F, 1)
cv.Laplace(gray, dst,aperture)

cv.Convert(dst,gray)

thresholded = cv.CloneImage(im)
cv.Threshold(im, thresholded, 50, 255, cv.CV_THRESH_BINARY_INV)

cv.ShowImage('Laplaced grayscale',gray)
'''



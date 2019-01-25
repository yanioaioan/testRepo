import cv2
import numpy as np
#from matplotlib import pyplot as plt

np.set_printoptions(threshold=np.nan)

img = cv2.imread('images/grey_image_1.jpg',0)


# Output dtype = cv2.CV_8U
sobelx8u = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=5)

# Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
sobelx64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
abs_sobel64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobel64f)

ret,thresh1 = cv2.threshold(sobel_8u,30,255,cv2.THRESH_BINARY)


cv2.imshow('thresh1',thresh1)
cv2.waitKey(0)
cv2.destroyAllWindows()


#print "=%s"%(repr(sobel_8u))


'''
plt.subplot(1,3,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(sobelx8u,cmap = 'gray')
plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(sobel_8u,cmap = 'gray')
plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])

plt.show()
'''

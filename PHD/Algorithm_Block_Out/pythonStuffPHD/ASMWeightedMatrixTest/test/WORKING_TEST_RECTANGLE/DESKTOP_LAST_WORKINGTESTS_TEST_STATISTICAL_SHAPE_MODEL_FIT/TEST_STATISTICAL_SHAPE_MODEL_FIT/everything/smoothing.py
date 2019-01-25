import cv2
import numpy as np
#from matplotlib import pyplot as plt

img = cv2.imread('images/grey_image_1.jpg')
cv2.imshow("original", img)

kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(img,-1,kernel)
cv2.imshow("smoothed", dst)

if cv2.waitKey(0) == 27:
	
	cv2.destroyAllWindows()


#plt.subplot(121),plt.imshow(img),plt.title('Original')
#plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
#plt.xticks([]), plt.yticks([])
#plt.show()

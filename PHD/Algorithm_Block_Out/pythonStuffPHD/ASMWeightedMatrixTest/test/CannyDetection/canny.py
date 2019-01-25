

import cv2
import numpy as np
'''
# Load the image
img = cv2.imread("grey_image_1.jpg")

# Split out each channel
blue, green, red = cv2.split(img)

def medianCanny(img, thresh1, thresh2):
    median = numpy.median(img)
    img = cv2.Canny(img, int(thresh1 * median), int(thresh2 * median))
    return img

# Run canny edge detection on each channel
blue_edges = medianCanny(blue, 0.0, 0.1)
green_edges = medianCanny(green, 0.0, 0.1)
red_edges = medianCanny(red, 0.0, 0.1)

# Join edges back into image
edges = blue_edges | green_edges | red_edges

# Find the contours
contours,hierarchy = cv2.findContours(edges, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

hierarchy = hierarchy[0] # get the actual inner list of hierarchy descriptions

# For each contour, find the bounding rectangle and draw it
for component in zip(contours, hierarchy):
    currentContour = component[0]
    currentHierarchy = component[1]
    x,y,w,h = cv2.boundingRect(currentContour)
    if currentHierarchy[2] < 0:
        # these are the innermost child components
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
    elif currentHierarchy[3] < 0:
        # these are the outermost parent components
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)

# Finally show the image
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''




'''
def create_graph(vertex, color):
    for g in range(0, len(vertex)-1):
        for y in range(0, len(vertex[0][0])-1):
            cv2.circle(newimg, (vertex[g][0][y], vertex[g][0][y+1]), 3, (255,255,255), -1)
            cv2.line(newimg, (vertex[g][0][y], vertex[g][0][y+1]), (vertex[g+1][0][y], vertex[g+1][0][y+1]), color, 2)
    cv2.line(newimg, (vertex[len(vertex)-1][0][0], vertex[len(vertex)-1][0][1]), (vertex[0][0][0], vertex[0][0][1]), color, 2)


img = cv2.imread('grey_image_1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Remove of noise, if any
kernel = np.ones((2, 2),np.uint8)
erosion = cv2.erode(gray, kernel, iterations = 1)

#Create a new image of the same size of the starting image
height, width = gray.shape
newimg = np.zeros((height, width, 3), np.uint8)

#Canny edge detector
thresh = 30
edges = cv2.Canny(erosion, thresh, thresh*2)


contours,hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#print contours
for b,cnt in enumerate(contours):
    if hierarchy[0,b,3] == -1:
       approx = cv2.approxPolyDP(cnt,0.015*cv2.arcLength(cnt,True), True)
       clr = (255, 0, 0)
       create_graph(approx, clr) #function for drawing the found contours in the new img
cv2.imwrite('starg.jpg', newimg)
'''





'''
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('grey_image_1.jpg',0)
img2 = img.copy()
template = cv2.imread('template.jpg',0)
w, h = template.shape[::-1]

# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
             'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
 img = img2.copy()
 method = eval(meth)

 # Apply template Matching
 res = cv2.matchTemplate(img,template,method)
 min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
 if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
     top_left = min_loc
 else:
     top_left = max_loc
 bottom_right = (top_left[0] + w, top_left[1] + h)

 cv2.rectangle(img,top_left, bottom_right, 255, 2)

 plt.subplot(121),plt.imshow(res,cmap = 'gray')
 plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
 plt.subplot(122),plt.imshow(img,cmap = 'gray')
 plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
 plt.suptitle(meth)

 plt.show()
'''




'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''






import numpy as np
import cv2


''' file name : canny.py
Description : This sample shows how to find edges using canny edge detection
This is Python version of this tutorial : http://opencv.itseez.com/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html
Level : Beginner
Benefits : Learn to apply canny edge detection to images.
Usage : python canny.py
Written by : Abid K. (abidrahman2@gmail.com) , Visit opencvpython.blogspot.com for more tutorials '''


import cv2
import numpy as np

def CannyThreshold(lowThreshold):
    detected_edges = cv2.GaussianBlur(gray,(3,3),0)
    detected_edges = cv2.Canny(detected_edges,lowThreshold,lowThreshold*ratio,apertureSize = kernel_size)
    dst = cv2.bitwise_and(img,img,mask = detected_edges)  # just add some colours to edges from original image.
    cv2.imshow('canny demo',dst)

lowThreshold = 10
max_lowThreshold = 100
ratio = 5
kernel_size = 3

img = cv2.imread('grey_image_1.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.namedWindow('canny demo')

cv2.createTrackbar('Min threshold','canny demo',lowThreshold, max_lowThreshold, CannyThreshold)

CannyThreshold(0)  # initialization
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()


'''
img = cv2.imread("grey_image_1.jpg")
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV);
lower_hand = np.array([20,20,20])
upper_hand = np.array([40,40,40])

mask = cv2.inRange(hsv, lower_hand, upper_hand)

res = cv2.bitwise_and(img, img, mask=mask)

#"derp" wasn't needed in my code tho..
contours,hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    hull = cv2.convexHull(cnt)
    cv2.drawContours(img,[cnt],0,(0,255,0),2)
    cv2.drawContours(img,[hull],0,(0,0,255),2)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''


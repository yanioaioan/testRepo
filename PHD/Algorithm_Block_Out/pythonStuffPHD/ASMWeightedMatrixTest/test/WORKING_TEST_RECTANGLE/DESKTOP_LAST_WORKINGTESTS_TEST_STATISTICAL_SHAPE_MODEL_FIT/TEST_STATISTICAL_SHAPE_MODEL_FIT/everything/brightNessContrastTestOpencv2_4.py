import cv2.cv as cv

import random

im = cv.LoadImage("everything/images/fluoroTEST/test/semi-autoLandMarked/images/grey_image_32.jpg") #or LoadImage and access pixel with Get2D/Set2D

size = cv.GetSize(im)
grey_image = cv.CreateImage(size, 8, 1)

cv.CvtColor(im, grey_image, cv.CV_RGB2GRAY)

#decrease brightness of the image, increase the contrast of the image
for i in range(grey_image.height):
    for j in range(grey_image.width):
        #print 'decreasing brightness'
        grey_image[i,j] = grey_image[i,j] - 50
        #print 'increasing contrast'
        grey_image[i,j] = grey_image[i,j] * 1.5

cv.ShowImage("Normal", im)
cv.ShowImage("Brightnessed - Contrasted", grey_image)
cv.WaitKey(0)

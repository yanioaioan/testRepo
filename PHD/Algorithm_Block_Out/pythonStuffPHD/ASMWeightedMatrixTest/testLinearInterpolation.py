import cv2
import numpy as np
import scipy.ndimage

img = cv2.imread('myimage.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
height, width, channels = img.shape

equ = cv2.equalizeHist(gray)    # Remember histogram equalization works only for grayscale images

cv2.imshow('src',gray)
#cv2.imshow('equ',equ)

print (img.shape[0])


cv2.waitKey(0)

'''
x = np.arange(width*height)

print 'Original array:'
print x

print 'Resampled by a factor of 2 with nearest interpolation:'
print scipy.ndimage.zoom(x, 2, order=0)


print 'Resampled by a factor of 2 with bilinear interpolation:'
print scipy.ndimage.zoom(x, 2, order=1)


print 'Resampled by a factor of 2 with cubic interpolation:'
print scipy.ndimage.zoom(x, 2, order=3)
'''

#bilinear_interpolate(img,x


cv2.destroyAllWindows()


def bilinear_interpolate(im, img[1,:], img[1,:]):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1);
    x1 = np.clip(x1, 0, im.shape[1]-1);
    y0 = np.clip(y0, 0, im.shape[0]-1);
    y1 = np.clip(y1, 0, im.shape[0]-1);

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id

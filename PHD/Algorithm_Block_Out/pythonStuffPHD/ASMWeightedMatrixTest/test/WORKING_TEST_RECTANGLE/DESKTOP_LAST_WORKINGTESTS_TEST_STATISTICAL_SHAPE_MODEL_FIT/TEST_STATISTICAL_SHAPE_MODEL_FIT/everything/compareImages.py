import cv2
import numpy as np
import sys

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err
	
if len(sys.argv) == 3:
	original = cv2.imread(sys.argv[1])
	comparewith = cv2.imread(sys.argv[2])
else:
	original = cv2.imread("1_to_compare.jpg")
	comparewith = cv2.imread("2_to_compare.jpg")

err=mse(original,comparewith)
print "err between the 2 images is=%f"%(err)
print "So, images are dissimilar by %f "%(err)



#import math
#print "dist_euclidean"
#dist_euclidean = math.sqrt(sum((original - comparewith)^2)) / original.size


#Not currently used, taken from https://www.cs.hmc.edu/~jlevin/ImageCompare.py
def PixelCompare(im1, im2, mode = "pct", alpha = .01):
    if im1.size == im2.size and im1.mode == im2.mode:
        randPix = im1.getpixel((0,0))
        maxSum = []
        diff = []
        for channel in range(len(randPix)):
            diff += [0.0]
            maxSum += [0.0]
        width = im1.size[0]
        height = im1.size[1]
        for i in range(width):
            for j in range(height):
                pixel1 = im1.getpixel((i,j))
                pixel2 = im2.getpixel((i,j))
                for channel in range(len(randPix)):
                    maxSum[channel] += 255
                    diff[channel] += abs(pixel1[channel] - pixel2[channel])
        if mode == "pct":
            ret = ()
            for channel in range(len(randPix)):
                ret += (diff[channel]/maxSum[channel],)
            return ret
        for channel in range(len(randPix)):
            if diff[channel] > alpha*maxSum[channel]:
                return False
        return True
    return False

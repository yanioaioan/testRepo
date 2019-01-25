# import the necessary packages
import argparse
import cv2
from PyQt4 import QtGui
import sys

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False


import cv2
import numpy as np
import sys

hbins = 180
sbins = 255
hrange = [0,180]
srange = [0,256]
ranges = hrange+srange


def hist_lines(im,x1,x2,y1,y2):
    h = np.zeros((300,256,3))
    if len(im.shape)!=2:
        print "hist_lines applicable only for grayscale images"
        #print "so converting image to grayscale for representation"
        im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    im = im[y1:y2,x1:x2].copy()

    hist_item = cv2.calcHist([im],[0],None,[256],[0,256])
    cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
    hist=np.int32(np.around(hist_item))
    for x,y in enumerate(hist):
        cv2.line(h,(x,0),(x,y),(255,255,255))
    y = np.flipud(h)
    cv2.namedWindow('histogram_target[%d:%d,%d:%d]'%(y1,y2,x1,x2),cv2.WINDOW_NORMAL)
    cv2.imshow('histogram_target[%d:%d,%d:%d]'%(y1,y2,x1,x2),y)


    return hist_item



def click_and_crop(event, x, y, flags, param):
        # grab references to the global variables
        global refPt, cropping

        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
                #refPt = [(x, y)]
                cropping = True

        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
                # record the ending (x, y) coordinates and indicate that
                # the cropping operation is finished
                refPt.append((x, y))
                cropping = False

                # draw a rectangle around the region of interest
                #cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)

        for i in refPt:
            cv2.circle(image,(i[0],i[1]), 1, (0,0,255), -1)


        cv2.imshow("image", image)


app = QtGui.QApplication(sys.argv)
app.setStyle("fusion") #Changing the style

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image, clone it, and setup the mouse callback function
image = cv2.imread(args["image"])
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)


# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        image = clone.copy()

    # if the 'c' key is pressed, break from the loop
    elif key == ord("c"):
        break

    elif key == ord("e"):
        '''
        print "Opening the file..."
        target = open("1.pts", 'w')

        for i in refPt:
            #cv2.circle(image,(i[0],i[1]), 1, (0,0,255), -1)

            target.write(str(i[0]))
            target.write(" ")
            target.write(str(i[1]))
            target.write("\n")

        print "And finally, we close it."
        target.close()
        '''
        break



x1=refPt[0][0]
x2=refPt[1][0]
y1=refPt[0][1]
y2=refPt[1][1]

print "x1,y1,x2,y2=%d,%d,%d,%d"%(x1,y1,x2,y2)
cv2.waitKey(0)

# if there are two reference points, then crop the region of interest
# from teh image and display it
if len(refPt) == 2:
        roi = clone[y1:y2, x1:x2]
        cv2.imshow("ROI", roi)
        cv2.waitKey(0)


#TARGET IMAGE ROI
targetImage = cv2.imread(sys.argv[2])#iso_colour_bars.png')

targetRoi = hist_lines(targetImage, x1+10,x2+10,y1,y2)


more_similar = hist_lines(targetImage, x1,x2,y1,y2)

less_similar = hist_lines(targetImage, x1+100,x2+100,y1,y2)



for i in xrange(1):

        a_testImage_hsv_Roi_1_hist_DIFF = cv2.compareHist(targetRoi,more_similar,cv2.cv.CV_COMP_CHISQR)
        b_testImage_hsv_Roi_2_hist_DIFF = cv2.compareHist(targetRoi,less_similar,cv2.cv.CV_COMP_CHISQR)
        print "Method: {0} -- a: {1} , b: {2}".format(i, a_testImage_hsv_Roi_1_hist_DIFF, b_testImage_hsv_Roi_2_hist_DIFF)

#if method 0 has been used in compareHist then->: the highest the value, the more accurate the similarity
if a_testImage_hsv_Roi_1_hist_DIFF <= b_testImage_hsv_Roi_2_hist_DIFF:
        print "more_similar is more similar to target image roi of interest"
else:
        print "less_similar is more similar to target image roi of interest"



cv2.waitKey(0)



# close all open windows
cv2.destroyAllWindows()

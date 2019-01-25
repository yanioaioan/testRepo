# import the necessary packages
import argparse
import cv2
from PyQt4 import QtGui
import sys
 
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False
 
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
        
        print "Opening the file..."
        target = open("testmylandmark3.pts", 'w')

        for i in refPt:
            #cv2.circle(image,(i[0],i[1]), 1, (0,0,255), -1)
            
            target.write(str(i[0]))
            target.write(" ")
            target.write(str(i[1]))
            target.write("\n")
            
        print "And finally, we close it."
        target.close()
            
        break
        
 
# if there are two reference points, then crop the region of interest
# from teh image and display it
if len(refPt) == 2:
	roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
	cv2.imshow("ROI", roi)
	cv2.waitKey(0)
 
# close all open windows
cv2.destroyAllWindows()

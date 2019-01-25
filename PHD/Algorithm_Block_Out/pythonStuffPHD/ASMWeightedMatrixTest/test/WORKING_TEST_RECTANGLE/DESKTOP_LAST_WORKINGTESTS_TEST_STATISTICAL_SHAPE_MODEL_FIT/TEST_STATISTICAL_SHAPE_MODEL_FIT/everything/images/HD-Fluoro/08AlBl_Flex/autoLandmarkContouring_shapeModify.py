# import the necessary packages
import argparse
import cv2,cv
from PyQt4 import QtGui
import sys

'''
"need to call script like: python autoLandmarkContouring.py -p autoLandMarked/pts/ -i autoLandMarked/images/grey_image_6.png -n 6 -m 2
 and so on.."
'''


# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
mouseDrawnList = []
cropping = False


drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1

 # mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.rectangle(image,(ix,iy),(x,y),(0,255,0),-1)
            else:
                #cv2.circle(image,(x,y),5,(0,0,255),-1)

                if (x,y) not in mouseDrawnList:
                    mouseDrawnList.append((x, y))
                ###else:
                    ###print 'In already'



    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        #if mode == True:
            #cv2.rectangle(image,(ix,iy),(x,y),(0,255,0),-1)
            ###print 'drew rectangle'
        #else:
            #cv2.circle(image,(x,y),5,(0,0,255),-1)
            ###print 'drew circle'


    for i in mouseDrawnList:
        ###print len(mouseDrawnList)
        cv2.circle(image,(i[0],i[1]), 1, (0,0,255), -1)


def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
    #global mouseDrawnList, cropping

	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
                #mouseDrawnList = [(x, y)]
		cropping = True
                if (x,y) not in mouseDrawnList:
                    mouseDrawnList.append((x, y))
                #else:
                    ###print 'In already'

	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
                mouseDrawnList.append((x, y))
		cropping = False

		# draw a rectangle around the region of interest
                #cv2.rectangle(image, mouseDrawnList[0], mouseDrawnList[1], (0, 255, 0), 2)

        for i in mouseDrawnList:
	    cv2.circle(image,(i[0],i[1]), 1, (0,0,255), -1)


	cv2.imshow("image", image)

#click_and_drag landmark
#if one the pixel clikced is contained in the list of the landmarks drawn the detect which one and modify it and update the list of the landmarks drawn
def click_and_drag(event, x, y, flags, param):
    # grab references to the global variables
    #    global mouseDrawnList, cropping

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed

    if event == cv2.EVENT_LBUTTONDOWN:
                #mouseDrawnList = [(x, y)]
                ####print meanshapeVectorNew
                tmpMeanVec=Shape.from_vector(meanshapeVectorNew)

                sumx=0
                sumy=0
                ###print tmpMeanVec.pts
                #for i, pt in enumerate(tmpMeanVec.pts):
                    ###print "point",pt.x,pt.y
                    #cv2.waitKey(0)


                #if (x,y) in meanshapeVectorNew:
                    #mouseDrawnList.append((x, y))
                    ###print "landmark clicked"
                #else:
                    ####print "landmark clicked"
                #    pass




app = QtGui.QApplication(sys.argv)
app.setStyle("fusion") #Changing the style

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-n", "--landmarkspathnumber", required=True, help="name of #.pts file corresponding to an image, ..if grey_image_1 ..then 1.pts, and so on..")
ap.add_argument("-p", "--pointsPath", required=True, help="path to the 1.pts or 2.pts landmarks text file number, so as to calculate the mean shape so far")
ap.add_argument("-m", "--pickMode", required=True, help="mode of landmarks picking")#1 --> manual picking, 2 --> predefined shape points to write out,


args = vars(ap.parse_args())

# load the image, clone it, and setup the mouse callback function
image = cv2.imread(args["image"])
clone = image.copy()
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
#cv2.setMouseCallback("image", click_and_crop)
cv2.setMouseCallback("image", click_and_drag)
mode=False#draw circle
#cv2.setMouseCallback("image", draw_circle,mode)

landmarkspathnumber = args["landmarkspathnumber"]


#calculate "mean shape" from already existing shape vectors
sys.path.append('../../../../')#to be able to findtestInitASM.py
from testInitASM import PointsReader
import numpy as np, sys

pointsPath = args["pointsPath"]
pickMode = int(args["pickMode"])


shapesSoFar = PointsReader.read_directory(pointsPath)
if not shapesSoFar:
    raise Exception(" \n There needs to be at least one previously manually landmarked shape vector to start with \n \
so please create 1 first & then re run this program!\n")
shape_vectorsSoFar = np.array([s.get_vector() for s in shapesSoFar])
meanshapeVector = np.mean(shape_vectorsSoFar, axis=0) #[(50,50),(100,50),(50,100),(100,100)]


meanshapeVectorNew=[]
meanshapeVectorNew = meanshapeVector

#for s in shapesSoFar:
#    ###print len(s.pts)

from testInitASM import Shape

meanshape = Shape.from_vector(meanshapeVector)#.transform(t)

###print "meanshapeVector -->=",meanshapeVector

#drawing from mean shape
def drawMeanShape():
    global meanshape
    for pt_num, pt in enumerate(meanshape.pts):

        nextPoint=-1
        prevPoint=-1

        # Draw normals
        #print 'pt.x',pt.x
        #print 'pt.y',pt.y

        cv2.circle(image, (int(round(pt.x)), int(round(pt.y))), 1, (0,0,255), -1)
        nextPoint = (int(round(pt.x)), int(round(pt.y)))
        '''
        if prevPoint != -1:#draw line connecting every other point with the next one , except for the last to the first segment
            cv.Line(i, prevPoint , nextPoint ,(0,0,255),1)

        else:#draw line segment connecting the last to the first landmark
            prevPoint = (int(round(f.shape.pts[-1].x)), int(round(f.shape.pts[-1].y)))
            cv.Line(i, prevPoint , nextPoint ,(0,0,255),1)
        '''

        prevPoint = nextPoint
        #cv2.imshow("image", image)
        ####print "drawing mean point..%d..%s"%(pt_num,pt)
        #cv2.waitKey(100)


dx=0
dy=0
degreesOfRotation=0
new_degrees=0

meanshapeVec=Shape([])

def recalculateTransform(new_dx, new_dy, new_degrees, direction):

    #keep reseting the image and avoid having multiple shape instances drawn on it
    global image
    image = cv2.imread(args["image"])

    print 'Transforming %s..'%(direction)

    #Transformation recalculation
    global meanshapeVector
    global meanshapeVectorNew

    meanshapeVec=Shape.from_vector(meanshapeVectorNew)

    print "meanshapeVec.pts=",meanshapeVec.pts
    print "meanshapeVectorNew=",meanshapeVectorNew

    sumx=0
    sumy=0
    for i, pt in enumerate(meanshapeVec.pts):
        sumx+=pt.x
        sumy+=pt.y
    #Round the fuck out the nearest integer of sumx&sumy otherwise it introduces floating point errors.
    sumx=int(round((sumx)))/len(meanshapeVec.pts)

    print 'len(meanshapeVec.pts=',len(meanshapeVec.pts)
    print 'sumy=',int(round((sumy)))
    sumy=int(round((sumy)))/len(meanshapeVec.pts)

    from testInitASM import Point
    CenterOfMass = Point(sumx,sumy)
    CenterOfMass = Point(int(sumx),int(sumy))

    print "int(sumx),int(sumy)====",int(sumx),int(sumy)

    global degreesOfRotation
    degreesOfRotation=new_degrees#-15

    #CenterOfMass=Point(0,0)
    global dx
    global dy

    dx=int(round(new_dx))
    dy=int(round(new_dy))
    ###print dx,dy
    t=Point(dx,dy)
    if CenterOfMass.__eq__(Point(0,0)):
        t= Point(0,0)

    global meanshape

    ###print "current CenterOfMass = ",CenterOfMass
    #print 'dx=',dx
    #print 'dy=',dy

    meanshape = Shape.from_vector(meanshapeVector).transform(t, int(round((degreesOfRotation%360))), (int(sumx),int(sumy)))
    #print "meanshapeVec.pts=",meanshapeVec.pts

    #update meanshapeVector
    meanshapeVectorNew = meanshape.get_vector()
    ###print "meanshape.get_vector()=",meanshape.get_vector()

l_r=0
u_d=0
rot=0

# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    global l_r, u_d, rot

    #TRANSLATE
    # if the 'up' key is pressed, transform up
    if key == (82):
        u_d-=10
        recalculateTransform(l_r, u_d, rot, "Up")

    # if the 'down' key is pressed, transform up
    if key == (84):
        u_d+=10
        recalculateTransform(l_r, u_d, rot, "Down")

    # if the 'left' key is pressed, transform up
    if key == (81):
        l_r-=10
        recalculateTransform(l_r, u_d, rot, "Left")

    # if the 'right' key is pressed, transform up
    if key == (83):
        l_r+=10
        recalculateTransform(l_r, u_d, rot, "Right")

    #ROTATE
    # if the 'a' key is pressed, rotate left
    if key == (97):
        rot-=10
        recalculateTransform(l_r, u_d, rot, "_rotating")
    # if the 'd' key is pressed, rotate right
    if key == (100):
        rot+=10
        recalculateTransform(l_r, u_d, rot, "_rotating")



    #draw the shape again on the target image
    drawMeanShape()

    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
      image = clone.copy()

    # if the 'c' key is pressed, break from the loop
    elif key == ord("c"):
      break

    elif key == ord("e"):


        #Write out all mouse hand drawng contour points
        if pickMode==1:#1 stands for manual pick number of points

            if landmarkspathnumber!="":

                ###print "Opening the file..."
                stepper=5
                ###print pointsPath

                target = open(pointsPath+str(landmarkspathnumber)+".pts", 'w')#sys.argv[2] represents the # number of the landmarks text file: in ex. #.pts

                target.write("version: 1\n")
                target.write("n_points: %d\n"%(len(mouseDrawnList)/stepper))
                target.write("{\n")

                for i in range(len(mouseDrawnList)):
                    if i%stepper==0:#split distance divisor
                        #cv2.circle(image,(i[0],i[1]), 1, (0,0,255), -1)

                        target.write(str(mouseDrawnList[i][0]))
                        target.write(" ")
                        target.write(str(mouseDrawnList[i][1]))
                        target.write("\n")

                ###print "And finally, we close it."
                target.write("}\n")
                target.close()

            #else:
                  ###print "need to call script like: python landmark.py -i grey_image_2.png -n 2 or\n python landmark.py -i grey_image_3.png -n 3 and so on.."

            break

        elif pickMode==2:#2 stands for  predefined number of points
            #Write out all the contour landmarks points x,y
            ###print landmarkspathnumber

            if landmarkspathnumber!="":

                ###print "Opening the file..."
                ###print "meanshapeVectorNew=",meanshapeVectorNew

                ###print pointsPath
                target = open(pointsPath+str(landmarkspathnumber)+".pts", 'w')#sys.argv[2] represents the # number of the landmarks text file: in ex. #.pts

                target.write("version: 1\n")

                target.write("n_points: %d\n"%( len(meanshapeVectorNew)/2 ) )
                target.write("{\n")

                for i in range (len(meanshapeVectorNew)):

                        if i%2==0:
                            ###print meanshapeVectorNew[i]

                            target.write(str(meanshapeVectorNew[i]))
                            target.write(" ")
                        else:
                            target.write(str(meanshapeVectorNew[i]))
                            target.write("\n")

                ###print "And finally, we close it."
                target.write("}\n")
                target.close()

           #else:
                  ###print "need to call script like: python landmark.py -i grey_image_2.png -n 2 or\n python landmark.py -i grey_image_3.png -n 3 and so on.."

            break


# if there are two reference points, then crop the region of interest
# from the image and display it
if len(mouseDrawnList) == 2:
        roi = clone[mouseDrawnList[0][1]:mouseDrawnList[1][1], mouseDrawnList[0][0]:mouseDrawnList[1][0]]
        #cv2.imshow("ROI", roi)
        cv2.waitKey(0)

# close all open windows
cv2.destroyAllWindows()

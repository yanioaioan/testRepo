# import the necessary packages
import argparse
import cv2,cv
from PyQt4 import QtGui
import sys,copy

from math import sin, cos, radians

def rotate_point(point, angle, center_point=(0, 0)):
    """Rotates a point around center_point(origin by default)
    Angle is in degrees.
    Rotation is counter-clockwise
    """
    angle_rad = radians(angle % 360)
    # Shift the point so that center_point becomes the origin
    new_point = (point[0] - center_point[0], point[1] - center_point[1])
    new_point = (new_point[0] * cos(angle_rad) - new_point[1] * sin(angle_rad),
                 new_point[0] * sin(angle_rad) + new_point[1] * cos(angle_rad))
    # Reverse the shifting we have done
    new_point = (new_point[0] + center_point[0], new_point[1] + center_point[1])
    return new_point


def rotate_polygon(polygon, angle, center_point=(0, 0)):

    sumx=0
    sumy=0
    for i in range(len(polygon)):
        sumx += int(round(polygon[i][0]))
        sumy += int(round(polygon[i][1]))

    sumx=int(round((sumx)/len(polygon)))
    sumy=int(round((sumy)/len(polygon)))

    """Rotates the given polygon which consists of corners represented as (x,y)
    around center_point (origin by default)
    Rotation is counter-clockwise
    Angle is in degrees
    """
    center_point=(sumx,sumy)
    print "Center of mass of poly=",center_point

    rotated_polygon = []
    for corner in polygon:
        rotated_corner = rotate_point(corner, angle, center_point)
        rotated_polygon.append(rotated_corner)
    return rotated_polygon



def translate_polygon(polygon,new_degrees, t):
    translated_polygon = []

    sumx=0
    sumy=0
    for i in range(len(polygon)):
        sumx += int(round(polygon[i][0]))
        sumy += int(round(polygon[i][1]))

        #translated_polygon.append(( int(round(polygon[i][0])), int(round(polygon[i][1])) ))

    sumx=int(round((sumx)/len(polygon)))
    sumy=int(round((sumy)/len(polygon)))

    """Rotates the given polygon which consists of corners represented as (x,y)
    around center_point (origin by default)
    Rotation is counter-clockwise
    Angle is in degrees
    """
    center_point=(sumx,sumy)
    print "Center of mass of poly=",center_point

    angle_rad = radians(new_degrees % 360)


    for point in polygon:
        new_point = (point[0] + t[0], point[1] + t[1])


        # Shift the point so that center_point becomes the origin
        #new_point = ( (point[0] -center_point[0])+ t[0], (point[1] -center_point[1])+ t[1] )

        translated_polygon.append(new_point)
    return translated_polygon





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
                #tmpMeanVec=Shape.from_vector(meanshapeVectorNew)

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
cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
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


shapesSoFar,filesReadOrder = PointsReader.read_directory(pointsPath)
if not shapesSoFar:
    raise Exception(" \n There needs to be at least one previously manually landmarked shape vector to start with \n \
so please create 1 first & then re run this program!\n")
shape_vectorsSoFar = np.array([s.get_vector() for s in shapesSoFar])
meanshapeVector = np.mean(shape_vectorsSoFar, axis=0) #[(50,50),(100,50),(50,100),(100,100)]
s=[]
for i,j in np.reshape(meanshapeVector, (-1,2)):
  s.append((i, j))
print s
meanshapeVector = s
#for i in meanshapeVector:


def translate(tx,ty):
    return np.matrix([[1, 0, tx],
                      [0, 1, ty],
                      [0, 0, 1]])

def rotate(theta):
    return np.matrix([[np.cos(theta), -np.sin(theta), 0],
               [np.sin(theta), np.cos(theta),  0],
               [0,                0,           1]])


print meanshapeVector

#meanshapeVector = [ (415, 615), (407, 600),(404, 587),(402, 571),(397, 554),(381 ,550),(364, 547),(351 ,546),(338, 545),(328, 555),(325,573),(315,560)]
#meanshapeVector = [ (200,200),(400,200),(400,400),(200,400)]


from testInitASM import Shape

#original intact shape to transform
#meanshape = Shape.from_vector(meanshapeVector)#.transform(t)

meanshapeVec=Shape([])

###print "meanshapeVector -->=",meanshapeVector

#drawing from mean shape
def drawMeanShape(m, centerofmass):
    #global meanshapeVector
    #for pt_num, pt in enumerate(m.pts):
    for i in range(len(m)):

        nextPoint=-1
        prevPoint=-1

        # Draw normals
        #print 'pt.x',pt.x
        #print 'pt.y',pt.y

        if i==0:
            col =(0,0,255)
        if i==1:
            col =(0,255,0)
        if i==2:
            col =(255,0,0)
        if i==3:
            col =(0,0,0)

        #first of all draw center of mass, caus I want to see it.
        #cv2.circle(image, (int(centerofmass[0]), int(centerofmass[1])), 1, col, -1)

        cv2.circle(image, (int(m[i][0]),int(m[i][1])), 1, col, -1)#m[i][0] m[i][1]
        nextPoint = (int(m[i][0]), int(m[i][1]))


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





def recalculateTransform(new_dx, new_dy, new_degrees, newShapeWithNedCoM, direction):

    #keep reseting the image and avoid having multiple shape instances drawn on it
    global image
    image = copy.copy(clone)

    print 'Transforming %s..'%(direction)

    #Transformation recalculation
    #global meanshapeVector
    #global meanshapeVectorNew

    #meanshapeVec=Shape.from_vector(meanshapeVectorNew)

    #print "meanshapeVec.pts=",meanshapeVec.pts
    #print "meanshapeVectorNew=",meanshapeVectorNew

    #grab the initial untransformed vector
    #WORKS!
    #_meanShape = Shape.from_vector(meanshapeVector)

    #grab the transformed vector everytime
    #_meanShape = copy.deepcopy(newShapeWithNedCoM)

    _meanShape = newShapeWithNedCoM[:]


    #global meanshape
    #_meanShape=meanshape


    sumx=0
    sumy=0

    #for i, pt in enumerate(_meanshape.pts):
    #    sumx+=int(round(pt.x))
    #    sumy+=int(round(pt.y))
    #    print "sumx,sumy=",sumx,sumy

    #global meanshapeVector
    #calculate it's center of mass
    '''
    for i in range(len(_meanShape.pts)):
        sumx += int(_meanShape.pts[i].x)
        sumy += int(_meanShape.pts[i].y)
    '''
    for i in range(len(_meanShape)):
        sumx += round(_meanShape[i][0])
        sumy += round(_meanShape[i][1])

    #Round the fuck out the nearest integer of sumx&sumy otherwise it introduces floating point errors.
    sumx=round(sumx/len(_meanShape))

    #print 'len(meanshapeVec.pts=',len(meanshapeVec.pts)
    #print 'sumy=',int(round((sumy)))
    sumy=round(sumy/len(_meanShape))
    com = ((sumx),(sumy))


    from testInitASM import Point
    #CenterOfMass = Point(sumx,sumy)
    #CenterOfMass = Point(int(sumx),int(sumy))

    #print "int(sumx),int(sumy)====",int(sumx),int(sumy)



    #CenterOfMass=Point(0,0)


    dx=new_dx
    dy=new_dy
    ###print dx,dy
    t=Point(dx,dy)
    #if CenterOfMass.__eq__(Point(0,0)):
    #    t= Point(0,0)

    #global meanshape

    ###print "current CenterOfMass = ",CenterOfMass
    #print 'dx=',dx
    #print 'dy=',dy

    #create and return a shape which is the original untransformed vector by deg degree
    #new_ss = Shape.from_vector(meanshapeVector).transform(t, deg%360, (int(sumx),int(sumy)))
    #new_ss = Shape.from_vector(_meanShape.get_vector()).transform(t, deg%360, (int(sumx),int(sumy)))

    new_ss=rotate_polygon(_meanShape, new_degrees)
    new_ss=translate_polygon(new_ss,  new_degrees, (dx,dy))

    #MATRIX WAY OF TRANSFORMING (please comment the above 2 lines, and uncomment below)
    '''
    trOrig = translate(-com[0],-com[1])

    trBackFromOrig = translate(com[0],com[1])

    #transform Shape's vertices
    tr = translate(dx, dy)

    theta=new_degrees*(np.pi/180.0)
    ro = rotate(theta)

    transformed_polygon=[]

    #currentVector=[_meanShape[0][0],_meanShape[0][0],1]
    #pointVector=np.matrix([1,1,1]).reshape(3,-1)
    counter=0
    for i in _meanShape:

        #this causes it to slow down as it creates a np.matrix all the time.However, itemset() below doesn't seem to work as well.
        currentVector=[i[0],i[1],1]#the 3rd element is always 1, to represent affine transformations
        pointVector=np.matrix(currentVector).reshape(3,-1)#reshape to match rows of the 1st matrix & columns of the 2nd matrix

        if counter==0:
            pointVector=np.matrix([1,1,1]).reshape(3,-1)
        counter+=1

        #pointVector.itemset((0,0), i[0])
        #pointVector.itemset((1,0), i[1])
        print i
        print pointVector

        m=(tr*trBackFromOrig*ro*trOrig)
        t=m*pointVector
        #print t.item(0,0)
        #print t.item(1,0)
        #print t.item(2,0)

        transformed_polygon.append( t)#(t.item(0,0), t.item(1,0))

    new_ss=transformed_polygon
    '''


    #meanshapeVector = new_ss.get_vector()

    #update meanshapeVector
    #meanshapeVectorNew = meanshape.get_vector()
    ###print "meanshape.get_vector()=",meanshape.get_vector()


    return new_ss, com



if __name__ == "__main__":

    # keep looping until the 'q' key is pressed

    l_r=0
    u_d=0
    rot=0
    dx=0
    dy=0
    degreesOfRotation=0
    new_degrees=0

    '''
    my_polygon = [ (415, 615), (407, 600),(404, 587),(402, 571),(397, 554),(381 ,550),(364, 547),(351 ,546),(338, 545),(328, 555),(925,573),(315,560)]
    while rot<3600:

        new_poly=rotate_polygon(my_polygon, rot)
        rot+=1

    exit()
    '''
    m = meanshapeVector#copy.deepcopy(meanshapeVector)
    com=(0,0)
    new = meanshapeVector
    #new = [ (200,200),(400,200),(400,400),(200,400)]

    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        global l_r, u_d, rot
        #shape to draw
        #print 'MM=',m

        #TRANSLATE
        # if the 'up' key is pressed, transform up
        if key == (82):
            l_r=0
            u_d=-1
            rot=0
            dx=0
            dy=0

            m,com = recalculateTransform(l_r, u_d, rot, new ,"Up")
            new = list(m[:])


        # if the 'down' key is pressed, transform up
        if key == (84):
            l_r=0
            u_d=1
            rot=0
            dx=0
            dy=0

            m,com = recalculateTransform(l_r, u_d, rot, new ,"Down")
            new = list(m[:])


        # if the 'left' key is pressed, transform up
        if key == (81):
            l_r=-1
            u_d=0
            rot=0
            dx=0
            dy=0

            m,com = recalculateTransform(l_r, u_d, rot, new ,"Left")
            new = list(m[:])


        # if the 'right' key is pressed, transform up
        if key == (83):
            l_r=1
            u_d=0
            rot=0
            dx=0
            dy=0

            m,com = recalculateTransform(l_r, u_d, rot, new ,"Right")
            new = list(m[:])

        #ROTATE
        # if the 'a' key is pressed, rotate left
        if key == (97):
            l_r=0
            u_d=0
            rot=-1
            dx=0
            dy=0

            m,com = recalculateTransform(l_r, u_d, rot, new ,"_rotating")
            new = list(m[:])


        # if the 'd' key is pressed, rotate right
        if key == (100):
            l_r=0
            u_d=0
            rot=1
            dx=0
            dy=0

            m,com = recalculateTransform(l_r, u_d, rot, new ,"_rotating")
            #new = Shape(list(m.pts[:]))
            new = list(m[:])




        #draw the shape again on the target image
        drawMeanShape(m,com)

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
                print pointsPath+str(landmarkspathnumber)+".pts",
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

                    target.write("n_points: %d\n"%( len(meanshapeVector) ) )#/2
                    target.write("{\n")

                    for i in range (len(meanshapeVector)):
                                print 'writing out',str(m[i][0])+" "+str(m[i][1])

                                target.write(str(m[i][0])+" "+str(m[i][1]))
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

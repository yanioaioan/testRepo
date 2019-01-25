from __future__ import division 
import math, cv,pontLineSideTest

def lineMagnitude (x1, y1, x2, y2):
    lineMagnitude = math.sqrt(math.pow((x2 - x1), 2)+ math.pow((y2 - y1), 2))
    return lineMagnitude
 
#Calc minimum distance from a point and a line segment (i.e. consecutive vertices in a polyline).
def DistancePointLine (px, py, x1, y1, x2, y2):
    #http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
    LineMag = lineMagnitude(x1, y1, x2, y2)
 
    if LineMag < 0.00000001:
        DistancePointLine = 9999
        return DistancePointLine
 
    u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
    u = u1 / (LineMag * LineMag)
 
    if (u < 0.00001) or (u > 1):
        #// closest point does not fall within the line segment, take the shorter distance
        #// to an endpoint
        ix = lineMagnitude(px, py, x1, y1)
        iy = lineMagnitude(px, py, x2, y2)
        if ix > iy:
            DistancePointLine = iy
        else:
            DistancePointLine = ix
    else:
        # Intersecting point is on the line, use the formula
        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        DistancePointLine = lineMagnitude(px, py, ix, iy)
 
    return DistancePointLine


#dst=DistancePointLine(10,3, 6,4, 8,6)
#print dst


'''
def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False
'''
def slope(P1, P2):
    # dy/dx
    # (y2 - y1) / (x2 - x1)
    
    div = (P2[0] - P1[0])
    if div == 0:
		div=1
    return (P2[1] - P1[1]) / div

def y_intercept(P1, slope):
    # y = mx + b
    # b = y - mx
    # b = P1[1] - slope * P1[0]
    return P1[1] - slope * P1[0]

def line_intersect(m1, b1, m2, b2):
    if m1 == m2:
        print ("These lines are parallel!!!")
        return None
    # y = mx + b
    # Set both lines equal to find the intersection point in the x direction
    # m1 * x + b1 = m2 * x + b2
    # m1 * x - m2 * x = b2 - b1
    # x * (m1 - m2) = b2 - b1
    # x = (b2 - b1) / (m1 - m2)
    x = (b2 - b1) / (m1 - m2)
    # Now solve for y -- use either line, because they are equal here
    # y = mx + b
    y = m1 * x + b1
    return x,y

def oppositeSigns(x, y):
    print "x",x
    print "y",y

    #if abs(x < 10) or abs(y < 10):#account for pixel error
    #    return False #same side

    return ((x ^ y) < 0);

def test(imageLoaded, intersectedPoints, queriedPoint,previousPoint,nextPoint, theoriticalContourP1, theoriticalContourP2):
    i = imageLoaded


    #Start point
    p=(100,40)#middle point defined by adjacent points below
    p=queriedPoint
    #cv.Circle(i, (int(p[0]),int(p[1]) ) ,4  ,(0,100,255),1)
    #cv.NamedWindow("landmarks On Target Image", cv.CV_WINDOW_NORMAL)
    #cv.ShowImage("landmarks On Target Image", i)
    #cv.WaitKey(0)

    #iterate over all points in pairs
    p1=(110,60)
    p1=(int(previousPoint[0]),int(previousPoint[1]))

    p2=(90,20)
    p2=(int(nextPoint[0]),int(nextPoint[1]))


    #cv.Circle(i, (int(p1[0]),int(p1[1]) ) ,4  ,(255,0,0),1)
    #cv.Circle(i, (int(p2[0]),int(p2[1]) ) ,4  ,(255,0,0),1)
    #cv.NamedWindow("landmarks On Target Image", cv.CV_WINDOW_NORMAL)
    #cv.ShowImage("landmarks On Target Image", i)
    #cv.WaitKey(0)


    x=p1[0]-p2[0]
    y=p1[1]-p2[1]
    mag = math.sqrt(x**2 + y**2)
    print "p1="+str(p1)
    print "p2="+str(p2)
    #normalDefineByTwoPoints p1,p2 (p is considerend to be equidistant between p1&p2)
    if mag!=0:
        norm=(-y/mag, x/mag)
    else:
        mag=1
        norm=(-y/mag, x/mag)

    cv.Line(i, p1 , p2 ,(0,255,255),1)#yellow line segment connecting pren & next points of the p point in question
    #cv.NamedWindow("landmarks On Target Image", cv.CV_WINDOW_NORMAL)
    #cv.ShowImage("landmarks On Target Image", i)
    #cv.WaitKey(0)


    #start of line along normal to p
    new_p = (p[0] + 1*norm[0], p[1] + 1*norm[0])
    t=0
    perp1= (int(new_p[0] + t*norm[0]) , int(new_p[1] + t*norm[1]) )

    t=-50
    perp2= (int(new_p[0] + t*norm[0]) ,int(new_p[1] + t*norm[1]) )


    normalLine= (perp1, perp2)
    print "normalLine",normalLine

    cv.Line(i, normalLine[0] , normalLine[1] ,(0,0,255),1)

    #cv.NamedWindow("landmarks On Target Image", cv.CV_WINDOW_NORMAL)
    #cv.ShowImage("landmarks On Target Image", i)
    #cv.WaitKey(0)


    lp1=normalLine[0]
    lp2=normalLine[1]

    lp1X= lp1[0]
    lp1Y= lp1[1]
    lp2X= lp2[0]
    lp2Y= lp2[1]

    #intersectedPoints=[]

    #test with segment 1
    testP1=(40,20)
    testP1=(int(theoriticalContourP1[0]),int(theoriticalContourP1[1]))
    testP2=(30,40)
    testP2=(int(theoriticalContourP2[0]),int(theoriticalContourP2[1]))

    cv.Line(i, testP1 , testP2 ,(0,255,0),1)
    cv.NamedWindow("landmarks On Target Image", cv.CV_WINDOW_NORMAL)
    cv.ShowImage("landmarks On Target Image", i)
    #cv.WaitKey()



    #find lines' intersection
    A1 = [lp1X, lp1Y]
    A2 = [lp2X, lp2Y]
    B1 = [testP1[0],testP1[1]]
    B2 = [testP2[0],testP2[1]]
    slope_A = slope(A1, A2)
    slope_B = slope(B1, B2)
    y_int_A = y_intercept(A1, slope_A)
    y_int_B = y_intercept(B1, slope_B)
    R=line_intersect(slope_A, y_int_A, slope_B, y_int_B)


    A=p1
    cv.Circle(i, (int(A[0]),int(A[1]) ) ,4  ,(255,255,0),1)
    #cv.NamedWindow("landmarks On Target Image", cv.CV_WINDOW_NORMAL)
    #cv.ShowImage("landmarks On Target Image", i)
    print "A",A
    cv.WaitKey(1)

    B=p2
    cv.Circle(i, (int(B[0]),int(B[1]) ) ,4  ,(255,255,0),1)
    #cv.NamedWindow("landmarks On Target Image", cv.CV_WINDOW_NORMAL)
    #cv.ShowImage("landmarks On Target Image", i)
    print "B",B
    cv.WaitKey(1)

    P=R
    #cv.Circle(i, (int(P[0]),int(P[1]) ) ,4  ,(0,255,255),1)
    #cv.NamedWindow("landmarks On Target Image", cv.CV_WINDOW_NORMAL)
    #cv.ShowImage("landmarks On Target Image", i)
    print "P",P
    cv.WaitKey(1)

    Rp=normalLine[1]
    cv.Circle(i, (int(Rp[0]),int(Rp[1]) ) ,4  ,(255,255,0),1)
    cv.NamedWindow("landmarks On Target Image", cv.CV_WINDOW_NORMAL)
    cv.ShowImage("landmarks On Target Image", i)
    print "Rp",Rp
    cv.WaitKey(1)


    #P_LINE_TEST = pontLineSideTest.test(A,B,P,Rp)
    #(Bx - Ax) * (Cy - Ay) - (By - Ay) * (Cx - Ax)
    if P != None:
      P_LINE_TEST1 = (B[0] - A[0]) * (P[1] - A[1]) - (B[1] - A[1]) * (P[0] - A[0])
      P_LINE_TEST2 = (B[0] - A[0]) * (Rp[1] - A[1]) - (B[1] - A[1]) * (Rp[0] - A[0])

      print "P_LINE_TEST1",P_LINE_TEST1
      print "P_LINE_TEST2",P_LINE_TEST2

      cv.WaitKey(1)
      signs = oppositeSigns(int(P_LINE_TEST1), int(P_LINE_TEST2))
      print "oppositeSigns",signs

      #print 'P on same side as R'
      #if not signs:
      #print 'P on same side as R'
      cv.WaitKey(1)

      if R != None:
          intersectedPoints.append(R)
          print "R",R#(line_intersect(slope_A, y_int_A, slope_B, y_int_B))
          cv.Circle(i, (int(R[0]),int(R[1]) ) ,4  ,(0,255,255),1)
          cv.NamedWindow("landmarks On Target Image", cv.CV_WINDOW_NORMAL)
          cv.ShowImage("landmarks On Target Image", i)
          #cv.WaitKey()




    '''
    #test with segment 2
    testP1=(30,40)
    testP2=(40,80)

    cv.Line(i, testP1 , testP2 ,(0,255,0),1)
    #cv.NamedWindow("landmarks On Target Image", cv.CV_WINDOW_NORMAL)
    #cv.ShowImage("landmarks On Target Image", i)

    #find lines' intersection
    A1 = [lp1X, lp1Y]
    A2 = [lp2X, lp2Y]
    B1 = [testP1[0],testP1[1]]
    B2 = [testP2[0],testP2[1]]
    slope_A = slope(A1, A2)
    slope_B = slope(B1, B2)
    y_int_A = y_intercept(A1, slope_A)
    y_int_B = y_intercept(B1, slope_B)
    R=line_intersect(slope_A, y_int_A, slope_B, y_int_B)
    intersectedPoints.append(R)
    print(line_intersect(slope_A, y_int_A, slope_B, y_int_B))

    #cv.Circle(i, (int(R[0]),int(R[1]) ) ,4  ,(255,255,0),1)
    #cv.NamedWindow("landmarks On Target Image", cv.CV_WINDOW_NORMAL)
    #cv.ShowImage("landmarks On Target Image", i)




    #test with segment 3
    testP1=(40,80)
    testP2=(70,100)

    cv.Line(i, testP1 , testP2 ,(0,255,0),1)
    #cv.NamedWindow("landmarks On Target Image", cv.CV_WINDOW_NORMAL)
    #cv.ShowImage("landmarks On Target Image", i)

    #find lines' intersection
    A1 = [lp1X, lp1Y]
    A2 = [lp2X, lp2Y]
    B1 = [testP1[0],testP1[1]]
    B2 = [testP2[0],testP2[1]]
    slope_A = slope(A1, A2)
    slope_B = slope(B1, B2)
    y_int_A = y_intercept(A1, slope_A)
    y_int_B = y_intercept(B1, slope_B)
    R=line_intersect(slope_A, y_int_A, slope_B, y_int_B)
    intersectedPoints.append(R)
    print(line_intersect(slope_A, y_int_A, slope_B, y_int_B))

    #cv.Circle(i, (int(R[0]),int(R[1]) ) ,4  ,(255,255,0),1)
    #cv.NamedWindow("landmarks On Target Image", cv.CV_WINDOW_NORMAL)
    #cv.ShowImage("landmarks On Target Image", i)
    '''


    #Find closest point out of a list of points intersectedPoints
    '''
    minDistance=10000000
    bestIndex=-1
    for ind in range(len(intersectedPoints)):
        x1=intersectedPoints[ind][0]
        y1=intersectedPoints[ind][1]
        x2=p[0]
        y2=p[1]

        dst=dist = math.hypot(x2 - x1, y2 - y1)
        if dst < minDistance:
            minDistance = dst
            bestIndex=ind

    closestIntersectedPointAlongNormal = intersectedPoints[bestIndex]

    #cv.Circle(i, (int(closestIntersectedPointAlongNormal[0]),int(closestIntersectedPointAlongNormal[1]) ) ,4  ,(255,255,255),1)
    #cv.NamedWindow("landmarks On Target Image", cv.CV_WINDOW_NORMAL)
    #cv.ShowImage("landmarks On Target Image", i)
    #dst=DistancePointLine(10,3, 6,4, 8,6)
    #print dst

    cv.WaitKey()
    '''

    return intersectedPoints


if __name__ == "__main__":

  test()

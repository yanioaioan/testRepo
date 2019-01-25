import math as math
import cv
import sys

class vec2Point ( object ):
  """a point in 2d cartesian point"""
  def __init__(self, x, y):
    self.x = x
    self.y = y



def get_normal_to_point(p1, p2):

    '''
    in general..
    Given a 2d vector (a,b), the normal is (x,y):
    x = b
    y = -a

    So basically, flip a and b, and negate the y. Two assignments and one negation. Can't be cheaper!
    '''

    x = p2.x - p1.x
    y = p2.y - p1.y

    mag = math.sqrt(x**2 + y**2)#normalize as well
    return vec2Point(-y/mag, x/mag)

if __name__=="__main__":

    i = cv.LoadImage(sys.argv[1])
    img = cv.CreateImage(cv.GetSize(i), i.depth, 3)
    cv.Copy(i, img)


    p1 = vec2Point(100.0,100.0)
    cv.Circle(img, ( int(p1.x), int(p1.y) ), 2, (0,0,200))
    p2 = vec2Point(200.0,200.0)
    cv.Circle(img, ( int(p2.x), int(p2.y) ), 2, (0,0,200))

    normal=get_normal_to_point(p1,p2)

    print 'normal between (%f,%f) & (%f,%f) = (%f,%f)'%( p1.x, p1.y, p2.x, p2.y, normal.x, normal.y)


    for whisk in range(-20,20):#along normal profile
        # Normal to normal...

        #blue whisker
        new_p = vec2Point((normal.x+100) + whisk*(normal.x), (normal.y+200) + whisk*(normal.y))
        cv.Circle(img, ( int(new_p.x), int(new_p.y) ), 2, (200,0,0))

        #green tangent
        for tan in range(-50,50):#along normal profile
            # normal to normal...

            #x = int((normal.x*tan + new_p.x))#*math.sin(t*(math.pi/180)))
            #y = int((normal.y*tan + new_p.y))#*math.cos(t*(math.pi/180)))

            tanP = vec2Point((new_p.x) + tan*-(normal.y), (new_p.y) + tan*(normal.x))

            #new_p = vec2Point((normal.x+100) + tan*(normal.x), (normal.y+200) + tan*(normal.y))
            cv.Circle(img, ( int(tanP.x), int(tanP.y) ), 2, (0,200,0))
            cv.ShowImage("test normal between 2 points",img)
            c=cv.WaitKey(1)

            if c==1048603 :#whichever integer key code makes the app exit
                exit()

        #cv.Circle(img, ( int(new_p.x), int(new_p.y) ), 2, (200,0,0))





    cv.NamedWindow("test normal between 2 points", cv.CV_WINDOW_AUTOSIZE)
    cv.ShowImage("test normal between 2 points",img)
    cv.WaitKey()

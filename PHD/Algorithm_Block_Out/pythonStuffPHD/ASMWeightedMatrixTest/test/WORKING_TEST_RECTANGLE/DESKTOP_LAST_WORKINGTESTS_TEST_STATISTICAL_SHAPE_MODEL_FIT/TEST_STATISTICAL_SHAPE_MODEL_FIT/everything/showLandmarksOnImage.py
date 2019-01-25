import cv,glob,sys,os
class Point ( object ):
  """ Class to represent a point in 2d cartesian space """
  def __init__(self, x, y):
    self.x = x
    self.y = y




  def __add__(self, p):
    """ Return a new point which is equal to this point added to p
    :param p: The other point
    """
    return Point(self.x + p.x, self.y + p.y)

  def __div__(self, i):#division by a constant
    return Point(self.x/i, self.y/i)

  def __eq__(self, other):
    return self.x == other.x and self.y == other.y

  def __ne__(self, other):
    return not self.__eq__(other)

  def __repr__(self):
    """return a string representation of this point. """
    return '(%f, %f)' % (self.x, self.y)

  def dist(self, p):
    """ Return the distance of this point to another point

    :param p: The other point
    """
    return math.sqrt((p.x - self.x)**2 + (p.y - self.y)**2)

  def __abs__(self):
    """ Return a new point which is the absolute of p
    :param p: The other point
    """
    return Point( abs(self.x), abs(self.y) )

  '''
  def __idiv__(self, p):#division by another point
    """ Return a new point which is the division of bothe x and y correspondingly with p
    :param p: The other point
    """
    return Point( self.x/float(p.x), self.y/float(p.y) )
  '''

class Shape ( object ):
  """ Class to represent a shape.  This is essentially a list of Point
  objects
  """
  def __init__(self, pts = []):
    self.pts = pts
    self.num_pts = len(pts)

  def __add__(self, other):
    """ Operator overloading so that we can add one shape to another
    """
    s = Shape([])
    for i,p in enumerate(self.pts):
      s.add_point(p + other.pts[i])
    return s

  def __div__(self, i):
    """ Division by a constant.
    Each point gets divided by i
    """
    s = Shape([])
    for p in self.pts:
      s.add_point(p/i)
    return s

  def __eq__(self, other):
    for i in range(len(self.pts)):
      if self.pts[i] != other.pts[i]:
        return False
    return True

  def __ne__(self, other):
    return not self.__eq__(other)
  '''
  def add_point(self, p):
    self.pts.append(p)
    self.num_pts += 1


  def transform(self, t):
    s = Shape([])
    for p in self.pts:
      s.add_point(p + t)
    return s
  '''

  def add_point(self, p, degreesOfRotation=0, CenterOfMass=Point(0,0)):


    if CenterOfMass.x != 0 and CenterOfMass.y != 0 and degreesOfRotation!=0:
        #degreesOfRotation=-15
        f=degreesOfRotation*(math.pi/180)
        #p.x = p.x * math.cos(f) - p.y *math.sin(f)
        #p.y = p.y * math.cos(f) + p.x *math.sin(f)
        p.x = CenterOfMass.x + (p.x-CenterOfMass.x)*math.cos(f)-(p.y-CenterOfMass.y)*math.sin(f)
        p.y = CenterOfMass.y + (p.x-CenterOfMass.x)*math.sin(f)+(p.y-CenterOfMass.y)*math.cos(f)

    self.pts.append(p)
    self.num_pts += 1

  def transform(self, t, degreesOfRotation=0, CenterOfMass=Point(0,0) ):
    s = Shape([])
    for p in self.pts:
      s.add_point(p + t, degreesOfRotation, CenterOfMass)
    return s

  """ Helper methods for shape alignment """
  def __get_X(self, w):
    return sum([w[i]*self.pts[i].x for i in range(len(self.pts))])
  def __get_Y(self, w):
    return sum([w[i]*self.pts[i].y for i in range(len(self.pts))])
  def __get_Z(self, w):
    return sum([w[i]*(self.pts[i].x**2+self.pts[i].y**2) for i in range(len(self.pts))])
  def __get_C1(self, w, s):
    return sum([w[i]*(s.pts[i].x*self.pts[i].x + s.pts[i].y*self.pts[i].y) \
        for i in range(len(self.pts))])
  def __get_C2(self, w, s):
    return sum([w[i]*(s.pts[i].y*self.pts[i].x - s.pts[i].x*self.pts[i].y) \
        for i in range(len(self.pts))])

  def get_alignment_params(self, s, w):
    """ Gets the parameters required to align the shape to the given shape
    using the weight matrix w.  This applies a scaling, transformation and
    rotation to each point in the shape to align it as closely as possible
    to the shape.

    This relies on some linear algebra which we use numpy to solve.

    [ X2 -Y2   W   0][ax]   [X1]
    [ Y2  X2   0   W][ay] = [Y1]
    [ Z    0  X2  Y2][tx]   [C1]
    [ 0    Z -Y2  X2][ty]   [C2]

    We want to solve this to find ax, ay, tx, and ty

    :param shape: The shape to align to
    :param w: The weight matrix
    :return x: [ax, ay, tx, ty]
    """

    X1 = s.__get_X(w)
    X2 = self.__get_X(w)
    Y1 = s.__get_Y(w)
    Y2 = self.__get_Y(w)
    Z = self.__get_Z(w)
    W = sum(w)
    C1 = self.__get_C1(w, s)
    C2 = self.__get_C2(w, s)

    a = np.array([[ X2, -Y2,   W,  0],
                  [ Y2,  X2,   0,  W],
                  [  Z,   0,  X2, Y2],
                  [  0,   Z, -Y2, X2]])

    b = np.array([X1, Y1, C1, C2])
    # Solve equations
    # result is [ax, ay, tx, ty]
    return np.linalg.solve(a, b)

  def apply_params_to_shape(self, p):
    new = Shape([])
    # For each point in current shape
    for pt in self.pts:
      new_x = (p[0]*pt.x - p[1]*pt.y) + p[2]
      new_y = (p[1]*pt.x + p[0]*pt.y) + p[3]
      new.add_point(Point(new_x, new_y))
    return new

  def align_to_shape(self, s, w):
    p = self.get_alignment_params(s, w)
    return self.apply_params_to_shape(p)

  def get_vector(self):
    vec = np.zeros((self.num_pts, 2))
    for i in range(len(self.pts)):
      vec[i,:] = [self.pts[i].x, self.pts[i].y]
    return vec.flatten()

  def get_normal_to_point(self, p_num):
    # Normal to first point
    x = 0; y = 0; mag = 0
    if p_num == 0:
      #original
      '''
      x = self.pts[1].x - self.pts[0].x
      y = self.pts[1].y - self.pts[0].y
      '''
      #landmark position dependent normal calculation

      x = self.pts[1].x - self.pts[-1].x
      y = self.pts[1].y - self.pts[-1].y

    # Normal to last point
    elif p_num == len(self.pts)-1:
      #original
      '''
      x = self.pts[-1].x - self.pts[-2].x
      y = self.pts[-1].y - self.pts[-2].y
      '''

      #landmark position dependent normal calculation

      x = self.pts[0].x - self.pts[-2].x
      y = self.pts[0].y - self.pts[-2].y

    # Must have two adjacent points, so...
    else:
      x = self.pts[p_num+1].x - self.pts[p_num-1].x
      y = self.pts[p_num+1].y - self.pts[p_num-1].y
    mag = math.sqrt(x**2 + y**2)

    #HACKED TO AVOID DIVISION BY zero when /norm[0] or when /norm[1]

    if y==0:
        y=1
    if x==0:
        x=1

    if mag!=0:
      return (-y/mag, x/mag)
    else:
        mag=1
        return (-y/mag, x/mag)

  @staticmethod
  def from_vector(vec):
    s = Shape([])
    for i,j in np.reshape(vec, (-1,2)):
      s.add_point(Point(i, j))
    return s

class ShapeViewer ( object ):
  """ Provides functionality to display a shape in a window
  """
  @staticmethod
  def show_shapes(shapes):
    """ Function to show all of the shapes which are passed to it
    """
    cv.NamedWindow("Shape Model",  cv.CV_WINDOW_NORMAL)#cv.CV_WINDOW_AUTOSIZE
    # Get size for the window
    max_x = int(max([pt.x for shape in shapes for pt in shape.pts]))
    max_y = int(max([pt.y for shape in shapes for pt in shape.pts]))
    min_x = int(min([pt.x for shape in shapes for pt in shape.pts]))
    min_y = int(min([pt.y for shape in shapes for pt in shape.pts]))

    i = cv.CreateImage((max_x-min_x+20, max_y-min_y+20), cv.IPL_DEPTH_8U, 3)
    cv.Set(i, (0, 0, 0))
    for shape in shapes:
      r = randint(0, 255)
      g = randint(0, 255)
      b = randint(0, 255)
      #r = 0
      #g = 0
      #b = 0

      nextPoint=-1
      prevPoint=-1

      for pt_num, pt in enumerate(shape.pts):
        # Draw normals
        #norm = shape.get_normal_to_point(pt_num)
        #cv.Line(i,(pt.x-min_x,pt.y-min_y), \
        #    (norm[0]*10 + pt.x-min_x, norm[1]*10 + pt.y-min_y), (r, g, b))



        #cv.Circle(i, (int(pt.x-min_x), int(pt.y-min_y)), 2, (r, g, b), -1)
        nextPoint = (int(pt.x-min_x), int(pt.y-min_y))
        if prevPoint != -1:
            cv.Line(i, prevPoint , nextPoint ,(0,255,255),1)

        prevPoint = nextPoint

    cv.ShowImage("Shape Model",i)

  @staticmethod
  def show_modes_of_variation(model, mode):
    # Get the limits of the animation
    start = -2*math.sqrt(model.evals[mode])
    stop = -start
    step = (stop - start) / 100

    b_all = np.zeros(model.modes)
    b = start
    while True:
      b_all[mode] = b
      s = model.generate_example(b_all)
      ShapeViewer.show_shapes([s])
      # Reverse direction when we get to the end to keep it running
      if (b < start and step < 0) or (b > stop and step > 0):
        step = -step
      b += step
      c = cv.WaitKey(10)
      if chr(255&c) == 'q': break



  @staticmethod
  def draw_model_fitter(f,scale):
    cv.NamedWindow("Model Fitter", cv.CV_WINDOW_NORMAL)
    # Copy image

    i = cv.CreateImage(cv.GetSize(f.target), f.target.depth, 3)

    #Shows original target
    cv.Copy(f.target, i)

    ##Shows derivatives of target image
    #convertedToColouredImg = cv.CreateImage(cv.GetSize( f.g_target_image[scale]), cv.IPL_DEPTH_8U, 3)
    #cv.CvtColor(f.g_target_image[scale], convertedToColouredImg, cv.CV_GRAY2BGR)
    #cv.Copy(convertedToColouredImg, i)



    nextPoint=-1
    prevPoint=-1

    for pt_num, pt in enumerate(f.shape.pts):
      # Draw normals
      cv.Circle(i, (int(pt.x), int(pt.y)), 2, (0,0,255), -1)
      #nextPoint = (int(pt.x), int(pt.y))
      #if prevPoint != -1:
      #    cv.Line(i, prevPoint , nextPoint ,(0,0,255),1)

      prevPoint = nextPoint
      cv.NamedWindow("Model Fitter", cv.CV_WINDOW_NORMAL)
      cv.ShowImage("Model Fitter",i)


class PointsReader ( object ):
  """ Class to read from files provided on Tim Cootes's website."""
  @staticmethod
  def read_points_file(filename):
    """ Read a .pts file, and returns a Shape object """
    s = Shape([])
    num_pts = 0
    with open(filename) as fh:
      # Get expected number of points from file
      first_line = fh.readline()
      if first_line.startswith("version"):
        # Then it is a newer type of file...
        num_pts = int(fh.readline().split()[1])
        # Drop the {
        fh.readline()
      else:
        # It is an older file...
        num_pts = int(first_line)
      for line in fh:
        if not line.startswith("}"):
          pt = line.strip().split()
          s.add_point(Point(float(pt[0]), float(pt[1])))
    if s.num_pts != num_pts:
      #print "Unexpected number of points in file.  "\
      "Expecting %d, got %d" % (num_pts, s.num_pts)
    return s

  @staticmethod
  def read_directory(dirname):
    """ Reads an entire directory of .pts files and returns
    them as a list of shapes
    """
    pts = []
    for file in glob.glob(os.path.join(dirname, "*.pts")):
      pts.append(PointsReader.read_points_file(file))
    return pts


imageCounterColor=0;

targetimg = cv.LoadImage(sys.argv[2])
shapes = PointsReader.read_directory(sys.argv[1])
for shape in shapes:
    imageCounterColor+=1


    for i,p in enumerate(shape.pts):
        tmpP = Point(p.x, p.y)
        cv.Circle(targetimg, ( int(tmpP.x), int(tmpP.y) ), 1, (imageCounterColor*50,imageCounterColor/50,0))
        cv.NamedWindow("landmarks On Target Image", cv.CV_WINDOW_NORMAL)
        cv.ShowImage("landmarks On Target Image", targetimg)
cv.WaitKey()

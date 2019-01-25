import numpy as np

#A startline point
#B endline point
#P test point
#R reference side point
def test(A,B,P,R):

  ABx=B[0] - A[0]
  ABy=B[1] - A[1];
  AB=(ABx,ABy)

  ARx=R[0] - A[0]
  ARy=R[1] - A[1];
  AR=(ARx,ARy)

  APx=P[0] - A[0]
  APy=P[1] - A[1];
  AP=(APx,APy)

  #ABxAR and ABxAP

  #You can still use cross products.
  #If A and B are the points defining the line,
  #R is your reference point, and P is your test point,
  #then form the two cross products ABxAR and ABxAP.
  #The dot product of these two vectors will be positive
  #if the points lie on the same side of the line
  #and negative if they do not.

  print AB, AR

  res1=np.cross(AB, AR)
  res2=np.cross(AB, AP)
  #res1=cross(AB,AR)
  #res2=cross(AB,AP)

  print res1,res2

  finalres=np.dot(res1,res1)
  print finalres

  if finalres >0:
      print 'P on same side as R'

  return finalres


if __name__ == "__main__":

    A=(1,1)
    B=(2,3)
    P=(-9,3)
    R=(-10,5)
    P=R

    A=(217, 105)
    B=(205, 117)
    P=(210.6, 112.6)
    R=(172, 74)

    A = (217, 105)
    B = (205, 117)
    P = (220.6, 122.6)
    R = (172, 74)
    P=R

    #(Bx - Ax) * (Cy - Ay) - (By - Ay) * (Cx - Ax)
    print (B[0] - A[0]) * (P[1] - A[1]) - (B[1] - A[1]) * (P[0] - A[0])

    exit()


    print test(A,B,P,R)


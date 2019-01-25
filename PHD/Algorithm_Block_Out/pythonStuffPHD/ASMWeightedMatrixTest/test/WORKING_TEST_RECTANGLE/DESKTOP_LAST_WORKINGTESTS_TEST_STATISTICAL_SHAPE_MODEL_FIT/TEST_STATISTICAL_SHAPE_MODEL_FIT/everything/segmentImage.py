#!/usr/bin/env python

from PyQt4 import QtGui
import numbers
from ASM_WORKING_HOME_MODIFIED_WORKING_optimizing import *



def ConvertFromCv2ToCvImage(i, imagepath):
    size=cv.GetSize(i)
    grey_imageFor = cv.CreateImage(size, 8, 1)
    grey_imageFormattedToOldCvVersion = cv.LoadImage(imagepath)
    cv.CvtColor(grey_imageFormattedToOldCvVersion, grey_imageFor, cv2.COLOR_BGR2GRAY)
    return grey_imageFor

def createCannyEdgeImage(image):

    lowThreshold = 7
    max_lowThreshold = 100
    ratio = 3
    kernel_size = 3

    img = cv2.imread(sys.argv[2])

    #smoothed before passed to canny edge detector
    kernel = np.ones((3,3),np.float32)/9
    img = cv2.filter2D(img,-1,kernel)
    #cv2.imshow("smoothed", dst)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    cv2.namedWindow('canny %s'%(sys.argv[2]))


    detected_edges = cv2.GaussianBlur(gray,(3,3),0)
    detected_edges = cv2.Canny(detected_edges,lowThreshold,lowThreshold*ratio,apertureSize = kernel_size)
    dst = cv2.bitwise_and(img,img,mask = detected_edges)  # just add some colours to edges from original image.
    cv2.imshow('Cannied: %s'%(sys.argv[2]),dst)
    cv2.imwrite("canny_target_image.jpg",dst)
    return dst







def main():

  #read in the marked shapes
  shapes = PointsReader.read_directory(sys.argv[1])

  targetimg = cv.LoadImage(sys.argv[2])

  imageCounterColor=0;
  for shape in shapes:
      imageCounterColor+=1
      for i,p in enumerate(shape.pts):
          tmpP = Point(p.x, p.y)
          cv.Circle(targetimg, ( int(tmpP.x), int(tmpP.y) ), 4, (imageCounterColor*50,imageCounterColor/50,0))
          cv.NamedWindow("landmarks On Target Image", cv.CV_WINDOW_NORMAL)
          cv.ShowImage("landmarks On Target Image", targetimg)
          #cv.WaitKey()


  #create a save original shapes as marked on original images
  '''
  for s in (shapes):
      for i,pt in enumerate(s.pts):
          print "%s"%(i)
  '''

  #deep copy the original marked shapes to save them for mahalanobis later use
  #cause they are going to fed to the ASM model to for a PDM and they are going to be changed
  #align to the mean shape and later on transformed to the target image shape
  originalShapes=copy.deepcopy(shapes)
  print len(originalShapes)
  cv.WaitKey()


  #create a Model for the shapes read
  a = ActiveShapeModel(shapes)
  
  # load the image target
  print "loaded image : %s\n"%(sys.argv[2])
  i = cv.LoadImage(sys.argv[2])


  #multiresolutionImage = cv2.imread(sys.argv[2])
  #lower_reso = cv2.pyrDown(multiresolutionImage)
  #cv2.imshow('multiresolutionImage',lower_reso)



  #create Canny edge image
  canniedTarget=createCannyEdgeImage(sys.argv[2])


  keyexit=cv.WaitKey()
  if keyexit == 1048603:
      exit()


  #convert from cv2 to cv image
  #i=ConvertFromCv2ToCvImage(i, "canny_target_image.jpg")



  #i = cv.LoadImage("canny_target_image.jpg")
  #m = FitModel(a, i, "canny_target_image.jpg", originalShapes) #i, sys.argv[2]
  m = FitModel(a, i, sys.argv[2], originalShapes)
  ShapeViewer.draw_model_fitter(m)
  
  #print "m.shape\,",m.shape.pts
  
  #need to add max iterations per multires-level
  #need to add The desired proportion of points withing ns/2 (4 pixels=4/2=2) of the current position
  
  GTrainingMeansCalculated = 0

  print "len of variation vecs= %d"%(len(a.evecs.T))
  cv.WaitKey()

  #for all directions of modes
  exitCheckingModesOfVariation=False
  while not exitCheckingModesOfVariation:
      for i in range(len(a.evecs.T)):
          ShapeViewer.show_modes_of_variation(a, i)
          print 'press enter to move to next movement OR ESC to exit the app'

          c=cv.WaitKey()
	  print "keypressed=%d"%(c)

          #exit the app
          if c == 1048603:
              exit()
          #exitCheckingModesOfVariation
	  if c == 1048677 or c == 1048586:# 'e' Key press for exitCheckingModesOfVariation:# 'e' Key press for exitCheckingModesOfVariation
	      exitCheckingModesOfVariation=True
	      print "exitCheckingModesOfVariation=True"
	      break





  #cv.WaitKey()
  totalNumberOfIterations=500
  for iterationIndex in range(totalNumberOfIterations):

    if iterationIndex>0:
        GTrainingMeansCalculated=1

    m.do_iteration(0 ,iterationIndex, GTrainingMeansCalculated)#performs 100 iterations based on the first image resolution


    #cv.WaitKey()
    print 'iteration %d'%(iterationIndex)
    #c=cv.WaitKey()
    #if chr(255&c) == 'q': exit()
    ShapeViewer.draw_model_fitter(m)
    cv.WaitKey(10)
  c=cv.WaitKey()
  if isinstance(c, int) :#whichever integer key code makes the app exit
      exit()


  
  
  #print "m.shape after iteration\,",m.shape.pts
  
  
  '''  
  #print 'STAGE 1 DONE'
  		
  for i in range(100):
    m.do_iteration(3)
    ShapeViewer.draw_model_fitter(m)
  
  #print 'STAGE 2 DONE'
  
  '''
  
  '''
  for i in range(10):
    m.do_iteration(3)
    ShapeViewer.draw_model_fitter(m)
    
  
  #print 'STAGE 3 DONE'
  
  for j in range(100):
    m.do_iteration(0)
    ShapeViewer.draw_model_fitter(m)
 
  #print 'STAGE 4 DONE'
  '''
  
    

if __name__ == "__main__":

    app = QtGui.QApplication(sys.argv)
    app.setStyle("fusion") #Changing the style

    main()
    sys.exit(app.exec_())
  


#!/usr/bin/env python

from PyQt4 import QtGui
import numbers
#from ASM import *
from testInitASM import *

import copy
import cv2

def exportSegmentationPoints(model, landmarkspathnumber, path=None):
    print "exporting Segmentation Points"

    print model.shape.pts
    cv.WaitKey()



    print "Opening the file..."

    target=0
    if path==None:
        target = open(str(sys.argv[1])+"/edgeSegmentedPoints/"+str(landmarkspathnumber)+".pts", 'w')#sys.argv[2] represents the # number of the landmarks text file: in ex. #.pts
    else:
        print 'Path specified to export statically segmented images...'
        cv.WaitKey()
        print path
        cv.WaitKey()
        target = open(path, 'w')#sys.argv[2] represents the # number of the landmarks text file: in ex. #.pts

    target.write("version: 1\n")
    target.write("n_points: %d\n"%( len(model.shape.pts) ) )
    target.write("{\n")

    for i in range (len(model.shape.pts)):
            #print int(model.shape.pts[i].x)
            #print len(model.shape.pts)
            #cv.WaitKey()

            target.write(str(int(model.shape.pts[i].x)))
            target.write(" ")
            target.write(str(int(model.shape.pts[i].y)))
            target.write("\n")

    print "And finally, we close it."
    target.write("}\n")
    target.close()



def ConvertFromCv2ToCvImage(i, imagepath):
    size=cv.GetSize(i)
    grey_imageFor = cv.CreateImage(size, 8, 1)
    grey_imageFormattedToOldCvVersion = cv.LoadImage(imagepath)
    cv.CvtColor(grey_imageFormattedToOldCvVersion, grey_imageFor, cv2.COLOR_BGR2GRAY)
    return grey_imageFor

def createCannyEdgeImage(image):

    lowThreshold = 17
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
  #print np.array([s.get_vector() for s in shapes])
  #cv.WaitKey()


  targetimg = cv.LoadImage(sys.argv[2])


  images = []
  imageNames = []
  for i in range (len(shapes)):
      testImageNameCounter=i+1
      path=sys.argv[2]
      pathSplit = path.split("/")
      #print path
      #print pathSplit

      finalpath=''
      for pathSubPart in range(len(pathSplit)-1):
          finalpath+=pathSplit[pathSubPart]+'/'

      dirname=finalpath
      #print dirname
      #cv.WaitKey()


      #strip from the 2nd cmd argument the grey_image_2.png for example "images/fluoroTest"
      """ Reads an entire directory of either .jpg or .png files"""
      currentImageName = glob.glob(os.path.join(dirname, "grey_image_%d.jpg"%(testImageNameCounter) ))
      #or search for png images
      if not currentImageName:
          currentImageName = glob.glob(os.path.join(dirname, "grey_image_%d.png"%(testImageNameCounter) ))

      #print currentImageName
      #currentImageName="all5000images"+ "\\" +"franck_"+"%05d"%(testImageNameCounter)+".jpg"#grey_image_
      print currentImageName[0]
      test_grey_image = cv.LoadImage(currentImageName[0])
      print "LALALA=%s"%(currentImageName[0])
      images.append(test_grey_image)
      imageNames.append(currentImageName[0])



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
  target = cv.LoadImage(sys.argv[2])


  #multiresolutionImage = cv2.imread(sys.argv[2])
  #lower_reso = cv2.pyrDown(multiresolutionImage)
  #cv2.imshow('multiresolutionImage',lower_reso)



  #create Canny edge image
  canniedTarget=createCannyEdgeImage(sys.argv[2])


  keyexit=cv.WaitKey()
  if keyexit == 1048603:
      exit()


  #convert from cv2 to cv image
  #i=ConvertFromCv2ToCvImage(target, "canny_target_image.jpg")



  #i = cv.LoadImage("canny_target_image.jpg")
  #m = FitModel(a, i, "canny_target_image.jpg", originalShapes) #i, sys.argv[2]
  m = FitModel(a, images, imageNames, target, originalShapes)# sys.argv[2], originalShapes
  scale=2
  ShapeViewer.draw_model_fitter(m,scale)
  
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
        print c

        #exit the app
        if c == 1048603:
            exit()
        #exitCheckingModesOfVariation
        if  chr(255&c) == 'e': #c == 1048677:# 'e' Key press for exitCheckingModesOfVariation
            print "exitCheckingModesOfVariation"
            global exitCheckingModesOfVariation
            exitCheckingModesOfVariation=True
            break
      if exitCheckingModesOfVariation == True:
          break



  '''
  for i in range(1):
    m.do_iteration(3)
    ShapeViewer.draw_model_fitter(m)
  for i in range(1):
    m.do_iteration(2)
    ShapeViewer.draw_model_fitter(m)
  for i in range(10):
    m.do_iteration(3)
    ShapeViewer.draw_model_fitter(m)
  for j in range(100):
    m.do_iteration(0)
    ShapeViewer.draw_model_fitter(m)
  '''


  #Reset Gmean&Covariances once before each pyramid Level
  resetTrainingGmeaAndCovariances=1;
  for iterationIndex in range(10):
    scale=2
    skipToNextLevel=m.do_iteration(scale, resetTrainingGmeaAndCovariances)#,iterationIndex, GTrainingMeansCalculated#performs 100 iterations based on the first image resolution
    if skipToNextLevel == True :
        break
    resetTrainingGmeaAndCovariances=0

    print 'iteration %d'%(iterationIndex)

    ShapeViewer.draw_model_fitter(m,scale)
    cv.WaitKey(1)

  cv.WaitKey()

  #write the segmentation of this frame after 30 iterations
  path=sys.argv[2]
  pathSplit = path.split("_")
  #print pathSplit
  splitExt = pathSplit[-1].split(".")
  #print splitExt
  #print splitExt[0]
  image_landmarkNumber = splitExt[0]
  whereToExport = (str(sys.argv[1])+"/exportedPointsForEachStaticFrame/"+"_frames__scale_2_frame_"+str(image_landmarkNumber)+".pts")
  exportSegmentationPoints(m, images, whereToExport)

  #Reset Gmean&Covariances once before each pyramid Level
  resetTrainingGmeaAndCovariances=1;
  for iterationIndex in range(30):

    scale=1
    skipToNextLevel=m.do_iteration(scale, resetTrainingGmeaAndCovariances )#,iterationIndex, GTrainingMeansCalculated#performs 100 iterations based on the first image resolution
    if skipToNextLevel == True :
        break
    resetTrainingGmeaAndCovariances=0

    print 'iteration %d'%(iterationIndex)

    ShapeViewer.draw_model_fitter(m,scale)
    cv.WaitKey(1)

  cv.WaitKey()


  #write the segmentation of this frame after 30 iterations
  path=sys.argv[2]
  pathSplit = path.split("_")
  #print pathSplit
  splitExt = pathSplit[-1].split(".")
  #print splitExt
  #print splitExt[0]
  image_landmarkNumber = splitExt[0]
  whereToExport = (str(sys.argv[1])+"/exportedPointsForEachStaticFrame/"+"_frames__scale_1_frame_"+str(image_landmarkNumber)+".pts")
  exportSegmentationPoints(m, images, whereToExport)

  #Reset Gmean&Covariances once before each pyramid Level
  resetTrainingGmeaAndCovariances=1;
  for iterationIndex in range(200):

    scale=0
    skipToNextLevel=m.do_iteration(scale, resetTrainingGmeaAndCovariances )#,iterationIndex, GTrainingMeansCalculated#performs 100 iterations based on the first image resolution
    if skipToNextLevel == True :
        break
    resetTrainingGmeaAndCovariances=0

    print 'iteration %d'%(iterationIndex)

    ShapeViewer.draw_model_fitter(m,scale)
    cv.WaitKey(1)

  if WRITE_OUT_SEGMENTED_RESULT_AS_SET_OF_POINTS==1:

      path=sys.argv[2]
      pathSplit = path.split("_")
      #print pathSplit
      splitExt = pathSplit[-1].split(".")
      #print splitExt
      #print splitExt[0]
      image_landmarkNumber = splitExt[0]

      exportSegmentationPoints(m, image_landmarkNumber)

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
  





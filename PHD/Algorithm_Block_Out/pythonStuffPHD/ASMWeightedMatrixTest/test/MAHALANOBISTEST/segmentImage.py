#!/usr/bin/env python

from PyQt4 import QtGui

from ASM import *

def main():
  #read in the marked shapes
  shapes = PointsReader.read_directory(sys.argv[1])

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


  #create a Model for the shapes read
  a = ActiveShapeModel(shapes)
  
  # load the image target
  i = cv.LoadImage(sys.argv[2])


  m = ModelFitter(a, i, originalShapes)
  ShapeViewer.draw_model_fitter(m)
  
  #print "m.shape\,",m.shape.pts
  
  #need to add max iterations per multieres-level
  #need to add The desired proportion of points withing ns/2 (4 pixels=4/2=2) of the current position
  
  cv.WaitKey()
  for i in range(100):
    m.do_iteration(0 ,i)#performs 100 iterations based on the first image resolution
    #cv.WaitKey()
    print 'iteration %d'%(i)
    #c=cv.WaitKey()
    #if chr(255&c) == 'q': exit()
    ShapeViewer.draw_model_fitter(m)
    cv.WaitKey(1)
  c=cv.WaitKey()
  
  
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
  


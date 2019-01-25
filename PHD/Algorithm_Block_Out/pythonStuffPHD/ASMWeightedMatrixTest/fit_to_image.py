#!/usr/bin/env python

from active_shape_models import *

def main():
  shapes = PointsReader.read_directory(sys.argv[1])
  a = ActiveShapeModel(shapes)
  
  # load the image
  i = cv.LoadImage(sys.argv[2])
  
  m = ModelFitter(a, i)
  ShapeViewer.draw_model_fitter(m)
  
  #print "m.shape\,",m.shape.pts
  
  #need to add max iterations per multieres-level
  #need to add The desired proportion of points withing ns/2 (4 pixels=4/2=2) of the current position
  
  cv.WaitKey()
  for i in range(100):
    m.do_iteration(0)#performs 10 iterations based on the first image resolution
    #cv.WaitKey()
    #print 'iteration %d'%(i)
    #c=cv.WaitKey()
    #if chr(255&c) == 'q': exit()
    ShapeViewer.draw_model_fitter(m)
    cv.WaitKey(100)
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
  main()
  


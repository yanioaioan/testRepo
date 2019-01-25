#!/usr/bin/env python
# -*- coding: utf-8 -*-

#changes made

import math,cv
import numpy as np
import scipy
#from testInitASM import Point


#gaussianMatrix = np.array([0.002406  ,0.009255,      0.027867,       0.065666        ,0.121117,      0.174868,       0.197641        ,0.174868       ,0.121117       ,0.065666       ,0.027867       ,0.009255       ,0.002406])

#17 elements
gaussianMatrix17 = np.array([0.000078 ,0.000489 ,0.002403 ,0.009245 ,0.027835 ,0.065592 ,0.12098 ,0.17467 ,0.197417 ,0.17467 ,0.12098 ,0.065592 ,0.027835 ,0.009245 ,0.002403 ,0.000489 ,0.000078])
gaussianMatrix5 = np.array([0.153388 ,0.221461 ,0.250301 ,0.221461 ,0.153388])




def drange(start, stop, step):
  r = start
  while r < stop:
    yield r
    r += step

def myprint(arg):
    print "arg=%s"%(arg)

class Sampling:


    def __init__(self, TOTALLEVELS=4,#only 3 used currently
                DEBUG_LINES=0,
                STOP_AT_THE_END_OF_EVERY_LANDMARK=0,
                SHOW_EVERY_POSSIBLE_POINT=0,
                SHOW_EVERY_POSSIBLE_STATISTICAL_SAMPLE=0,
                USE_1D_PROFILE=1,
                USE_2D_PROFILE=0,
                USE_MAX_EDGE=0,
                STEP=2,#sample every "STEP" pixels along the whisker
                PIXELS_SEARCH=4, #6+1
                JUST_SMOOTH_NO_SOBEL=1,

                ):

                    print "Sampling Class init"

                    self.DEBUG_LINES=DEBUG_LINES
                    self.STOP_AT_THE_END_OF_EVERY_LANDMARK=STOP_AT_THE_END_OF_EVERY_LANDMARK
                    self.SHOW_EVERY_POSSIBLE_POINT=SHOW_EVERY_POSSIBLE_POINT
                    self.SHOW_EVERY_POSSIBLE_STATISTICAL_SAMPLE=SHOW_EVERY_POSSIBLE_STATISTICAL_SAMPLE
                    self.USE_1D_PROFILE=USE_1D_PROFILE
                    self.USE_2D_PROFILE=USE_2D_PROFILE
                    self.USE_MAX_EDGE=USE_MAX_EDGE
                    self.STEP=STEP#sample every "STEP" pixels along the whisker
                    self.PIXELS_SEARCH=PIXELS_SEARCH #6+1
                    self.JUST_SMOOTH_NO_SOBEL=JUST_SMOOTH_NO_SOBEL





    #created Once when trainingSetLevel_#.txt is (BEWARE-rename properly to avoid issues confusing trainingSetLevel_0.txt of one training set with another)

    def createStatisticalProfileModel(self, p_num, scale , image, shape, mat):
        """ Gets the max edge response along the normal to a point

        :param p_num: Is the number of the point in the shape
        """


        norm = shape.get_normal_to_point(p_num)
        #print norm
        p = shape.pts[p_num]

        # Find extremes of normal within the image
        # Test x first
        min_t = -p.x / norm[0]
        if p.y + min_t*norm[1] < 0:
          min_t = -p.y / norm[1]
        elif p.y + min_t*norm[1] > image.height:
          min_t = (image.height - p.y) / norm[1]

        # X first again
        max_t = (image.width - p.x) / norm[0]
        if p.y + max_t*norm[1] < 0:
          max_t = -p.y / norm[1]
        elif p.y + max_t*norm[1] > image.height:
          max_t = (image.height - p.y) / norm[1]

        # Swap round if max is actually larger...
        tmp = max_t
        max_t = max(min_t, max_t)
        min_t = min(min_t, tmp)

        # Get length of the normal within the image
        x1 = min(p.x+max_t*norm[0], p.x+min_t*norm[0])
        x2 = max(p.x+max_t*norm[0], p.x+min_t*norm[0])
        y1 = min(p.y+max_t*norm[1], p.y+min_t*norm[1])
        y2 = max(p.y+max_t*norm[1], p.y+min_t*norm[1])
        l = math.sqrt((x2-x1)**2 + (y2-y1)**2)


        #cv.NamedWindow("Statistical model sampling", cv.CV_WINDOW_NORMAL)
        #cv.ShowImage("Statistical model sampling", image)
        #cv.WaitKey()

        img = cv.CreateImage(cv.GetSize(image), image.depth, 1)
        cv.Copy(image, img)

        convertedToColouredImg = cv.CreateImage(cv.GetSize(image), cv.IPL_DEPTH_8U, 3)
        cv.CvtColor(img, convertedToColouredImg, cv.CV_GRAY2BGR)

        # Scan over the whole line
        max_pt = p
        max_edge = 0
        max_edge_contr = 0

        #search = 2*scale+1

        #Example starting from -5...to...4
        search = self.PIXELS_SEARCH#how many pixels to search on each direction excluding the center element-->say 4..so in total will be 5.


        #Assigning Gaussian Weights to gradient profile elements
        #Precalculate the total number of whisker elements so as to .. remove noise based on gausian kernel contribution (4.2   H. Lu and F. Yang    Width of Search Profile)
        whiskerElements=0
        #for t in drange(-search+1 if -search > min_t else min_t+1, \
        # , ,     search if search < max_t else max_t , 1):

        if -search+1==0:#make sure -search+1 doesn't become 0, by hacking/setting search to 2
            search=2
        for t in range(-search, search+1,1):
            whiskerElements+=1
        #print "\nwhiskerElements=%d"%(whiskerElements)

        if self.DEBUG_LINES==1:
            cv.WaitKey()
        #create 1D convolution filter for each whisker line
        #x = np.arange(-whiskerElements/2+1, (whiskerElements/2)+1, 1)
        x = np.arange(-search, (search)+1, 1)


        #myprint(x)
        stdDev=2
        #convolution array
        conv1D = 1 / np.sqrt(2 * np.pi) * stdDev*np.exp(-x ** 2 / (2.*stdDev**2))
        #conv1D = 1 / stdDev * np.sqrt(2 * np.pi) * stdDev*np.exp( (-1/2.0) * (x/stdDev)**2)
        #myprint(conv1D)

        # Look 6 pixels to each side too
        '''
        for side in range(-2*(scale+1), 2*(scale+1) ):#profile WIDTH

          # Normal to normal...
          ##print "side=%s"%(side)
          #cv.WaitKey()

          #profile LENGTH
          new_p = Point(p.x + side*-norm[1], p.y + side*norm[0])
        '''
        side=0
        #samplePointsList=Shape([])
        #sampled intensity on the derivative image
        samplePointsList= []

        if self.USE_1D_PROFILE==1:

                    smoothedContribution=0

                    #print "creating Statistical Shape 1D Profile"

                    side=0

                    import testInitASM#only import the molule and used like "testInitASM.method()". Don't import 'from' the module, otherwise circular dependency becomes apparent
                    new_p = testInitASM.Point(p.x + side*-norm[1], p.y + side*norm[0])

                    #calculate contributionCoef at range
                    tmpIntensitiesArray=[]
                    #for t in drange(-search+1 if -search > min_t else min_t+1, \
                                   #search if search < max_t else max_t-1 , 1):
                    #CHANGED THE ABOVE 2 LINES to THE BELOW 1 LINE (could cause out of bounds exception while accessing the image pixels)

                    t=(-search-1)#-5..to..4
                    prev_x = int(new_p.x + ( t )*norm[0])#self.STEP*
                    prev_y = int(new_p.y + ( t)*norm[1])#self.STEP*

                    for t in range(-search, search+1 , 1):
                               x = int(new_p.x + (t)*norm[0])#self.STEP*
                               y = int(new_p.y + (t)*norm[1])#self.STEP*

                               #print y-1, x-1
                               #print prev_y-1, prev_x-1

                               #page 94, http://www.milbo.org/stasm-files/phd-milborrow.pdf
                               const=2;
                               w=5
                               h=5
                               #USE IT BELOW
                               gaussianEulerDistFromCenterWeight=np.exp( -const * (x**2+y**2)/ (((w/2)**2) +((h/2)**2)) )

                               print "conv1D--------=%r"%(conv1D)
                               #cv.WaitKey(0)


                               #tmpIntensitiesArray.append(   ( image[y, x] - image[prev_y, prev_x] ) )#calculate signed gradient

                               #Denoise: 4.2 H. Lu and F. Yang  Width of Search Profile
                               #By expanding the width of search profile, the effect of noises can be reduced in
                               #some degree. Thus, it can be used to improve the robustness of classical ASM.
                               #Subspace Methods for Pattern Recognition in Intelligent Environment
                               tmpIntensitiesArray.append( conv1D[t+search]*  ( (0.25*image[y-1, x-1] + 0.5*image[y, x] + 0.25*image[y+1, x+1]) - (0.25*image[prev_y-1, prev_x-1]+ 0.5*image[prev_y, prev_x] + 0.25*image[prev_y+1, prev_x+1]) ) )#calculate signed gradient

                               convertedToColouredImg = cv.CreateImage(cv.GetSize( image), cv.IPL_DEPTH_8U, 3)
                               cv.CvtColor(img, convertedToColouredImg, cv.CV_GRAY2BGR)
                               #explicit gradient sampling of what's actually beign store in each element of 'tmpIntensitiesArray' above
                               #cv.Circle(convertedToColouredImg, (x, y), 1, (0,255,0))#individually sampled points along the normal
                               cv.Circle(convertedToColouredImg, (x-2, y-2), 1, (0, 100, 0))#individually sampled points along the normal
                               cv.Circle(convertedToColouredImg, (x-1, y-1), 1, (0, 255, 0))#individually sampled points along the normal
                               cv.Circle(convertedToColouredImg, (x, y),     1, (0, 100, 0))#individually sampled points along the normal
                               #explicit gradient sampling
                               cv.Circle(convertedToColouredImg, (prev_x-2, prev_y-2), 1, (0,100,0))#individually sampled points along the normal
                               cv.Circle(convertedToColouredImg, (prev_x-1, prev_y-1), 1, (0,255,0))#individually sampled points along the normal
                               cv.Circle(convertedToColouredImg, (prev_x, prev_y),     1, (0,100,0))#individually sampled points along the normal
                               #cv.NamedWindow("convertedToColouredImgTMP", cv.CV_WINDOW_NORMAL)
                               #cv.ShowImage("convertedToColouredImgTMP",convertedToColouredImg)
                               print 'CREATING TRAINING SET SAMPLING..'
                               #cv.WaitKey()

                               prev_x = x
                               prev_y = y


                    #print " tmpIntensitiesArray =%r"%(tmpIntensitiesArray)
                    #cv.WaitKey()
                    ##print "multiplied with =%r"%(gaussianMatrix[offsetT+(search-1)])
                    #cv.WaitKey()
                    #tmpIntensitiesArray= [0.034561*el for el in tmpIntensitiesArray]
                    #print "After tmpIntensitiesArray =%r"%(tmpIntensitiesArray)
                    #cv.WaitKey()



                    #for t in drange(-search+1 if -search > min_t else min_t+1, \
                                   #search if search < max_t else max_t-1 , 1):
                    #CHANGED THE ABOVE 2 LINES to THE BELOW 1 LINE (could cause out of bounds exception while accessing the image pixels)
                    for t in range(-search, search+1 , 1):

                          #normal for this side (side starts spreads across the window width, controlled by 'search' variable)
                          x = int(new_p.x + (t)*norm[0])#self.STEP*
                          y = int(new_p.y + (t)*norm[1])#self.STEP*

                          '''
                          for a given point we sample along a profile k pixels either side of
                          the model point in the ith training image. We have 2k + 1 samples which can be
                          put in a vector gi .
                          '''


                          #samplePointsList.add_point(Point(x,y))
                          #ADD THE INTENSITY ..not the points themselves
                          #use the sobeled image
                          #samplePointsList.append(image[y, x])

                          #use this if no gaussian distribution is taken into account
                          #use the sobeled image read from 'SAMPLING_TEST.jpg' back in again
                          #samplePointsList.append(mat[y-1, x-1])

                          #Here we add to out current sampled intensity vector,
                          #the gaussian contribution instead of the actual intensity
                          #contributionCoefficient = conv1D [t+(whiskerElements/2)]
                          #contributionCoefficient=gaussianMatrix[t+(search-1)]


                          #contributionCoefficient=gaussianMatrix[t+(search-1)]*tmpIntensitiesArray[t+(search-1)]

                          #Sigma 2,   Kernel Size 17 # http://dev.theomader.com/gaussian-kernel-calculator/
                          #contributionCoefficient=gaussianMatrix5[t+(search)]*tmpIntensitiesArray[t+(search)]

                          #contributionCoefficient=scipy.ndimage.filters.gaussian_filter1d(tmpIntensitiesArray, 1)[t+(search-1)]
                          contributionCoefficient=tmpIntensitiesArray[t-(search+1)]

                          ##print "tmpIntensitiesArray=",tmpIntensitiesArray
                          ##print "gaussian_filter1d",gaussian_filter1d(tmpIntensitiesArray, 1)
                          ##print "gaussian_filter1d t",gaussian_filter1d(tmpIntensitiesArray,  3)[t+(search-1)]
                          ##print "t",t
                          #cv.WaitKey()


                          ##print "contributionCoefficient=",contributionCoefficient
                          ##print "gaussianMatrix=",gaussianMatrix
                          ##print "t=",t
                          #cv.WaitKey()

                          smoothedContribution = contributionCoefficient# * image[y, x]

                          samplePointsList.append(smoothedContribution)


                          '''
                          if not smoothedContribution:

                              print "tmpIntensitiesArray=",tmpIntensitiesArray
                              print "smoothedContribution=",smoothedContribution

                              cv.WaitKey()
                          '''


                          #print "x=%r, y=%r"%(x-1,y-1)

                          #Was
                          ##print "%r..intensity value added to sampleIntensity"%(image[y, x])
                          #print "%r..intensity value added to sampleIntensity"%(image[y, x])

                          if self.SHOW_EVERY_POSSIBLE_POINT==1:
                              cv.NamedWindow("convertedToColouredImgTMP", cv.CV_WINDOW_NORMAL)
                              cv.ShowImage("convertedToColouredImgTMP",convertedToColouredImg)
                              cv.WaitKey()


                    #normalize the sample by dividing by the sum of the absolute element values
                    #absPointSum=Point(0,0)
                    absPointSum=0

                    #for i in samplePointsList.pts:
                    for sampledIntensity in samplePointsList:

                        #print "sampledIntensity=%r"%(sampledIntensity)
                        absPointSum+=abs(sampledIntensity)
                        ##print abs(i)
                        ##print absPointSum
                        #cv.WaitKey()

                    #print "\nsamplePointsList before normalization=%r"%(samplePointsList)
                    for i in range(len(samplePointsList)):
                        if self.SHOW_EVERY_POSSIBLE_STATISTICAL_SAMPLE==1:
                            print samplePointsList
                            ##print absPointSum
                            #print i
                            #cv.WaitKey()
                        #normalize point
                        #samplePointsList.pts[i] *= 1/absPointSum
                        if absPointSum!=0:
                            samplePointsList[i] *= 1/absPointSum


                    if self.SHOW_EVERY_POSSIBLE_STATISTICAL_SAMPLE==1:
                        print samplePointsList
                        #cv.WaitKey()




                    if not samplePointsList:
                        print "samplePointsList=",samplePointsList
                        cv.WaitKey()

                    '''
                    if len(samplePointsList)<13:
                        print 'STOP'
                        cv.WaitKey()
                    print len(samplePointsList)
                    '''


                    return samplePointsList






    def get_MAHALANOBIS(self, shape, g_target_image, TrainingCovarianceMatrices, TrainingMean, p_num, scale, createImageOncePerIteration):#, curBest
        """ Gets the optimum fit based on the Mahalanobis for this particular point on this scale/level

        :param p_num: Is the number of the point in the shape
        """

        norm = shape.get_normal_to_point(p_num)
        p = shape.pts[p_num]

        # Find extremes of normal within the image
        # Test x first


        min_t = -p.x / norm[0]
        if p.y + min_t*norm[1] < 0:
          min_t = -p.y / norm[1]
        elif p.y + min_t*norm[1] >  g_target_image[scale].height:
          min_t = ( g_target_image[scale].height - p.y) / norm[1]

        # X first again
        max_t = ( g_target_image[scale].width - p.x) / norm[0]
        if p.y + max_t*norm[1] < 0:
          max_t = -p.y / norm[1]
        elif p.y + max_t*norm[1] >  g_target_image[scale].height:
          max_t = ( g_target_image[scale].height - p.y) / norm[1]
        '''

        min_t = -p.x / norm[0]
        if p.y + min_t*norm[1] < 0:
          min_t = -p.y / norm[1]
        elif p.y + min_t*norm[1] > self.target.height:
          min_t = (self.target.height - p.y) / norm[1]

        # X first again
        max_t = (self.target.width - p.x) / norm[0]
        if p.y + max_t*norm[1] < 0:
          max_t = -p.y / norm[1]
        elif p.y + max_t*norm[1] > self.target.height:
          max_t = (self.target.height - p.y) / norm[1]
        '''


        # Swap round if max is actually larger...
        tmp = max_t
        max_t = max(min_t, max_t)
        min_t = min(min_t, tmp)

        # Get length of the normal within the image
        x1 = min(p.x+max_t*norm[0], p.x+min_t*norm[0])
        x2 = max(p.x+max_t*norm[0], p.x+min_t*norm[0])
        y1 = min(p.y+max_t*norm[1], p.y+min_t*norm[1])
        y2 = max(p.y+max_t*norm[1], p.y+min_t*norm[1])
        l = math.sqrt((x2-x1)**2 + (y2-y1)**2)


        cv.NamedWindow("targetImageToWorkAgainst-grey_image", cv.CV_WINDOW_NORMAL)
        cv.ShowImage("targetImageToWorkAgainst-grey_image",g_target_image[scale])
        #cv.WaitKey()

        img = cv.CreateImage(cv.GetSize( g_target_image[scale]), g_target_image[scale].depth, 1)
        cv.Copy(g_target_image[scale], img)


        if createImageOncePerIteration:
          convertedToColouredImg = cv.CreateImage(cv.GetSize( g_target_image[scale]), cv.IPL_DEPTH_8U, 3)
          cv.CvtColor(img, convertedToColouredImg, cv.CV_GRAY2BGR)
          self.convertedToColouredImg = convertedToColouredImg

        convertedToColouredImg = self.convertedToColouredImg


        # Scan over the whole line
        max_pt = p
        max_edge = 0
        max_edge_contr = 0

        # Now check over the vector

        #search = 2*scale+1
        #Example starting from -5...to...and including 4, +- 2 on each side-->
        #-->so further down we''make it offset.. from -7...to...and including 6
        search = self.PIXELS_SEARCH #how many pixels to search on each direction excluding the center element-->say 4..so in total will be 5.



        #Assigning Gaussian Weights to gradient profile elements
        #Precalculate the total number of whisker elements so as to .. remove noise based on gausian kernel contribution (4.2   H. Lu and F. Yang    Width of Search Profile)
        whiskerElements=0
        #for t in drange(-search+1 if -search > min_t else min_t+1, \
        # , ,     search if search < max_t else max_t , 1):

        if -search+1==0:#make sure -search+1 doesn't become 0, by hacking/setting search to 2
            search=2
        profile_res=1
        for t in range(-(search+self.STEP), (search+self.STEP)+1,profile_res):
            whiskerElements+=1
        #print "\nwhiskerElements=%d"%(whiskerElements)

        if self.DEBUG_LINES==1:
            cv.WaitKey()
        #create 1D convolution filter for each whisker line
        print "whiskerElements"
        print whiskerElements
        #x = np.arange(-whiskerElements/2+1, (whiskerElements/2)+1, 1)
        x = np.arange(-search, (search)+1, 1)

        #myprint(x)
        stdDev=2
        #convolution array
        conv1D = 1 / np.sqrt(2 * np.pi)*stdDev*  np.exp(-x ** 2 / (2.*stdDev**2))
        #conv1D = 1 / stdDev * np.sqrt(2 * np.pi) * stdDev*np.exp( (-1/2.0) * (x/stdDev)**2)
        #myprint(conv1D)


        # Look 6 pixels to each side too
        '''
        for side in range(-2*(scale+1), 2*(scale+1) ):#profile WIDTH

          # Normal to normal...
          ##print "side=%s"%(side)
          #cv.WaitKey()

          #profile LENGTH
          new_p = Point(p.x + side*-norm[1], p.y + side*norm[0])
        '''


        #sampled intensity on the derivative image
        curBest=-1
        bestPoint=0
        savedOffset=0
        import testInitASM#only import the molule and used like "testInitASM.method()". Don't import 'from' the module, otherwise circular dependency becomes apparent
        prevBestPoint=testInitASM.Point(-1,-1)

        #at level 2-> runs from -6 to 6, then at level 1 runs from -3 to 3, and then at level 0 from -1 to 1
        rangeAtThisLevel=0
        rangeAtThisLevel=-search
        #print "rangeAtThisLevel=%d"%(rangeAtThisLevel)
        rangeAtThisLevel=abs(rangeAtThisLevel)


        if self.USE_1D_PROFILE==1:

                #always 'leave at initial poisition', if no other point
                #is minimizing the mahalanobis cost function
                offsetT=0

                #setting initial position of landmark (WITHOUT ANY OFFSET)
                side=0
                #import testInitASM#only import the molule and used like "testInitASM.method()". Don't import 'from' the module, otherwise circular dependency becomes apparent
                new_p_centered = testInitASM.Point(p.x + side*-norm[1], p.y + side*norm[0])
                xxx = int(new_p_centered.x + (offsetT)*norm[0])
                yyy = int(new_p_centered.y + (offsetT)*norm[1])

                hack=0
                ######so..calculate from -6 to and 6 with step 1 :  -6..-5..-4......0......4..5..6
                profile_res=1#COULD BE 2
                for offsetT in drange(-self.STEP, self.STEP+1, profile_res):## -2 0 2 offset

                    side=0
                    samplePointsList= []
                    #the very center of the profile
                    new_p_centered = testInitASM.Point(p.x + side*-norm[1], p.y + side*norm[0])

                    #the center of the sample offsetted profile
                    new_p_centered_xx_offset = int(new_p_centered.x + (offsetT)*norm[0])
                    new_p_centered_yy_offset = int(new_p_centered.y + (offsetT)*norm[1])

                    #calculate contributionCoef at range
                    tmpIntensitiesArray=[]
                    #for t in drange(-search+1 if -search > min_t else min_t+1, \
                                   #search if search < max_t else max_t-1 , 1):
                    #CHANGED THE ABOVE 2 LINES to THE BELOW 1 LINE (could cause out of bounds exception while accessing the image pixels)

                    t=(-search-1)#-7..to..6
                    prev_x = int(new_p_centered.x + (t+offsetT)*norm[0])#self.STEP*
                    prev_y = int(new_p_centered.y + (t+offsetT)*norm[1])#self.STEP*

                    for t in range(-search, search+1 , 1):
                               x = int(new_p_centered.x + (t+offsetT)*norm[0])#self.STEP*
                               y = int(new_p_centered.y + (t+offsetT)*norm[1])#self.STEP*

                               #print y-1, x-1
                               #print prev_y-1, prev_x-1

                               #page 94, http://www.milbo.org/stasm-files/phd-milborrow.pdf
                               const=2;
                               w=5
                               h=5
                               #USE IT BELOW
                               gaussianEulerDistFromCenterWeight=np.exp( -const * (x**2+y**2)/ (((w/2)**2) +((h/2)**2)) )

                               print "conv1D=%r"%(conv1D)
                               print "conv1D[%d]=%r"%(t+search,conv1D[t+search])
                               #cv.WaitKey(0)


                               #tmpIntensitiesArray.append(   ( g_target_image[scale][y, x] - g_target_image[scale][prev_y, prev_x] ) )
                               tmpIntensitiesArray.append( conv1D[t+search]* ( (0.25*g_target_image[scale][y-1, x-1] + 0.5*g_target_image[scale][y, x] + 0.25*g_target_image[scale][y+1, x+1]) - (0.25*g_target_image[scale][prev_y-1, prev_x-1]+ 0.5*g_target_image[scale][prev_y, prev_x] + 0.25*g_target_image[scale][prev_y+1, prev_x+1]) ) )

                               #convertedToColouredImg = cv.CreateImage(cv.GetSize( g_target_image[scale]), cv.IPL_DEPTH_8U, 3)
                               #cv.CvtColor(img, convertedToColouredImg, cv.CV_GRAY2BGR)
                               #explicit gradient sampling of what's actually beign store in each element of 'tmpIntensitiesArray' above
                               #cv.Circle(convertedToColouredImg, (x, y), 1, (0,255,0))#individually sampled points along the normal
                               ###cv.Circle(convertedToColouredImg, (x-2, y-2), 1, (0,20,0))#individually sampled points along the normal
                               #cv.NamedWindow("convertedToColouredImgTMP", cv.CV_WINDOW_NORMAL)
                               #cv.ShowImage("convertedToColouredImgTMP",convertedToColouredImg)
                               #cv.WaitKey()
                               ###cv.Circle(convertedToColouredImg, (x-1, y-1), 1, (0,50,0))#individually sampled points along the normal
                               #cv.NamedWindow("convertedToColouredImgTMP", cv.CV_WINDOW_NORMAL)
                               #cv.ShowImage("convertedToColouredImgTMP",convertedToColouredImg)
                               #cv.WaitKey()
                               ###cv.Circle(convertedToColouredImg, (x, y),     1, (0,20,0))#individually sampled points along the normal
                               #cv.NamedWindow("convertedToColouredImgTMP", cv.CV_WINDOW_NORMAL)
                               #cv.ShowImage("convertedToColouredImgTMP",convertedToColouredImg)
                               #cv.WaitKey()
                               #explicit gradient sampling
                               ###cv.Circle(convertedToColouredImg, (prev_x-2, prev_y-2), 1, (0,20,0))#individually sampled points along the normal
                               #cv.NamedWindow("convertedToColouredImgTMP", cv.CV_WINDOW_NORMAL)
                               #cv.ShowImage("convertedToColouredImgTMP",convertedToColouredImg)
                               #cv.WaitKey()
                               ###cv.Circle(convertedToColouredImg, (prev_x-1, prev_y-1), 1, (0,50,0))#individually sampled points along the normal
                               #cv.NamedWindow("convertedToColouredImgTMP", cv.CV_WINDOW_NORMAL)
                               #cv.ShowImage("convertedToColouredImgTMP",convertedToColouredImg)
                               #cv.WaitKey()
                               ###cv.Circle(convertedToColouredImg, (prev_x, prev_y),     1, (0,20,0))#individually sampled points along the normal
                               #cv.NamedWindow("convertedToColouredImgTMP", cv.CV_WINDOW_NORMAL)
                               #cv.ShowImage("convertedToColouredImgTMP",convertedToColouredImg)
                               print 'SEARCHING'
                               #cv.WaitKey()



                               prev_x = x
                               prev_y = y

                    #print "Before tmpIntensitiesArray =%r"%(tmpIntensitiesArray)
                    #print "sum(tmpIntensitiesArray) =%r"%(sum(tmpIntensitiesArray))
                    #cv.WaitKey()
                    #print "multiplied with =%r"%(gaussianMatrix[offsetT+(search-1)])
                    #cv.WaitKey()
                    #tmpIntensitiesArray= [gaussianMatrix[offsetT+(search-1)]*el for el in tmpIntensitiesArray]
                    #print "After tmpIntensitiesArray =%r"%(tmpIntensitiesArray)
                    #cv.WaitKey()


                    #for t in drange(-search+1 if -search > min_t else min_t+1, \
                                   #search if search < max_t else max_t-1 , 1):
                    #CHANGED THE ABOVE 2 LINES to THE BELOW 1 LINE (could cause out of bounds exception while accessing the image pixels)
                    for t in range(-search, search+1 , 1):

                          color = (255,255,255)
                          if   offsetT == -10:
                              color = (0,0,255)
                          elif offsetT == -8:
                              color = (0,255,255)
                          elif offsetT == -6:
                              color = (255,0,255)
                          elif offsetT == -4:
                              color = (255,255,0)
                          elif offsetT == -2:
                              color = (255,0,0)
                          elif offsetT == 0:
                              color = (138,43,226)
                          elif   offsetT == 2:
                              color = (30,144,255)
                          elif offsetT == 4:
                              color = (139,139,139)
                          elif offsetT == 6:
                              color = (255,193,193)
                          elif offsetT == 8:
                              color = (255,127,0)
                          elif offsetT == 10:
                              color = (255,174,185)

                          #convertedToColouredImg = cv.CreateImage(cv.GetSize( g_target_image[scale]), cv.IPL_DEPTH_8U, 3)
                          #cv.CvtColor(img, convertedToColouredImg, cv.CV_GRAY2BGR)


                          #normal for this side (side starts spreads across the window width, controlled by 'search' variable)

                          #the each of the points along the 'sample offsetted profile'
                          x = int(new_p_centered.x + (t+offsetT)*norm[0])#self.STEP*
                          y = int(new_p_centered.y + (t+offsetT)*norm[1])#self.STEP*


                          if self.SHOW_EVERY_POSSIBLE_POINT==1:

                            #cv.Circle(convertedToColouredImg, (x, y), 1, ( (offsetT+rangeAtThisLevel)*50, 255-(offsetT+rangeAtThisLevel)*50, search*(offsetT+10) ) )#individually sampled points along the normal
                            #cv.Circle(convertedToColouredImg, (x, y), 1, (color[0],color[1],color[2]) )#individually sampled points along the normal


                            #cv.Circle(convertedToColouredImg, (xxx, yyy), 1, ( 255,0,0) )#individually sampled points along the normal
                            #sample profile center point
                            #cv.Circle(convertedToColouredImg, (new_p_centered_xx_offset, new_p_centered_yy_offset), 2, ( 255,255,0 ) )
                            cv.NamedWindow("convertedToColouredImgTMP", cv.CV_WINDOW_NORMAL)
                            cv.ShowImage("convertedToColouredImgTMP",convertedToColouredImg)
                            #cv.WaitKey()


                          #now for each normal point sample again "this many times" to gather information as to that area
                          #samplePointsList= []

                          #for offsetT in drange(-rangeAtThisLevel,rangeAtThisLevel+1,1):
                          #new_p_centered_xx_offset=int(x + offsetT*norm[0])#int(x+offsetT)
                          #new_p_centered_yy_offset=int(y + offsetT*norm[1])#int(y+offsetT)

                          '''
                          for a given point we sample along a profile k pixels either side of
                          the model point in the ith training image. We have 2k + 1 samples which can be
                          put in a vector gi .
                          '''

                          print "t=%r"%(t)
                          print "offsetT=%r"%(offsetT)
                          print "t+(search)=%r"%(t+(search))
                          print "t+(search)+offsetT+self.STEP=%r"%(t+(search)+offsetT+self.STEP)
                          #cv.WaitKey()

                          #contributionCoefficient = conv1D [t+(whiskerElements/2)]
                          #contributionCoefficient=gaussianMatrix[t+(search-1)]

                          #CURRENTLY IT LOOKS LIKE THIS IS AN INCONSISTENT COMPARED TO THE contributionCoefficient,
                          #CALCULATED DURING __createStatisticalProfileModel(...)

                          #contributionCoefficient=gaussianMatrix[t+(search-1)]*tmpIntensitiesArray[t+(search-1)]

                          #Sigma 2,   Kernel Size 17 # http://dev.theomader.com/gaussian-kernel-calculator/
                          #contributionCoefficient=gaussianMatrix17[t+(search)+offsetT+self.STEP]*tmpIntensitiesArray[t+(search)]

                          #contributionCoefficient=scipy.ndimage.filters.gaussian_filter1d(tmpIntensitiesArray, 1)[t+(search-1)]
                          contributionCoefficient=tmpIntensitiesArray[t+(search)]


                          ##print "tmpIntensitiesArray=",tmpIntensitiesArray
                          ##print "gaussian_filter1d",gaussian_filter1d(tmpIntensitiesArray, 1)
                          ##print "gaussian_filter1d t",gaussian_filter1d(tmpIntensitiesArray,  3)[t+(search-1)]
                          ##print "t",t
                          #cv.WaitKey()

                          smoothedContribution= contributionCoefficient# * g_target_image[scale][y, x]

                          #samplePointsList.add_point(Point(x,y))
                          #ADD THE INTENSITY ..not the points themselves
                          #samplePointsList.append(g_target_image[scale][y, x])#the image target at this level
                          samplePointsList.append(smoothedContribution)#the image target at this level


                          if self.SHOW_EVERY_POSSIBLE_POINT==1:
                            print "x=%r"%(x)
                            ##print "y=%r"%(y)
                            #print "For the target image of this level for points: x=%r, y=%r"%(x-1,y-1)
                            #print "Intensity=%r"%(g_target_image[scale][y, x])


                          if self.SHOW_EVERY_POSSIBLE_POINT==1:
                              cv.NamedWindow("convertedToColouredImgTMP", cv.CV_WINDOW_NORMAL)
                              cv.ShowImage("convertedToColouredImgTMP",convertedToColouredImg)
                              cv.WaitKey()


                    #print "Before samplePointsList =%r"%(samplePointsList)
                    #cv.WaitKey()


                    #normalize the sample by dividing by the sum of the absolute element values
                    absPointSum=0
                    #for i in samplePointsList.pts:
                    for sampledIntensity in samplePointsList:

                        #print "sampledIntensity=%r"%(sampledIntensity)
                        absPointSum+=abs(sampledIntensity)


                    if self.SHOW_EVERY_POSSIBLE_POINT==1:
                        print "\nsamplePointsList before normalization=%r"%(samplePointsList)

                    for i in range(len(samplePointsList)):
                        if self.SHOW_EVERY_POSSIBLE_STATISTICAL_SAMPLE==1:
                            print i

                        if absPointSum!=0:#make sure absPointIsNotZero
                          samplePointsList[i] *= 1/absPointSum


                    if self.SHOW_EVERY_POSSIBLE_STATISTICAL_SAMPLE==1:
                        print "normalized sampledIntensity=%r"%(samplePointsList)
                        #cv.WaitKey()

                    ##print "current whisker normalized point samplePointsList=%r"%(samplePointsList)
                    #cv.WaitKey()
                    ##print "TrainingMean[%d]=%r"%(p_num,TrainingMean[p_num])
                    #cv.WaitKey()
                    #calculate Mahalanobis for this point using samplePointsList of the current point
                    #..and compared to the training Gmean & Cov matrix of acquire during training
                    #tmpMah = scipy.spatial.distance.mahalanobis(np.array(samplePointsList) , TrainingMean[p_num] , np.linalg.inv(np.cov(np.array(samplePointsList),TrainingMean[p_num])))



                    #tmpMah = scipy.spatial.distance.mahalanobis(samplePointsList , (TrainingMean[p_num]) , np.linalg.inv(np.cov(samplePointsList,(TrainingMean[p_num]))))
                    invcovar=0
                    #invcovar=cv2.invert(TrainingCovarianceMatrices[p_num], invcovar, cv2.DECOMP_SVD)#mycovar[0] when used with cv2.calcCovarMatrix #'''OR covar'''
                    #invcovar=invcovar[1]

                    #invcovar = np.linalg.inv(np.array(TrainingCovarianceMatrices[p_num]))#.reshape((3,3))

                    #This worked with 1d profile sampling, but it's hell to slow with 2d profile
                    #invcovar=cv2.invert(np.array(TrainingCovarianceMatrices[p_num]), invcovar, cv2.DECOMP_SVD)
                    #invcovar=invcovar[1]

                    invcovar=np.linalg.pinv(np.array(TrainingCovarianceMatrices[p_num]))

                    TrainingMean[p_num] = np.reshape(TrainingMean[p_num],(-1,1))
                    samplePointsList = np.reshape(samplePointsList,(-1,1))
                    if self.SHOW_EVERY_POSSIBLE_POINT==1:
                      ##print "TrainingCovarianceMatrices[p_num]=%r"%(TrainingCovarianceMatrices[p_num])
                      ##print "invcovar=%r"%(invcovar)
                      ##print "TrainingMean[%d]=%r"%(p_num,TrainingMean[p_num])
                      ##print "normalized samplePointsList=%r"%(samplePointsList)

                      ##print "sizeof invcovar=%r"%(invcovar.size)
                      ##print "sizeof np.array(samplePointsList).size=%d"%(np.array(samplePointsList).size)
                      ##print "sizeof (TrainingMean[p_num]).size=%d"%((TrainingMean[p_num]).size)
                      cv.WaitKey(1)

                    '''
                    #print "TrainingCovarianceMatrices[p_num]=%r"%(TrainingCovarianceMatrices[p_num])
                    #print "invcovar=%r"%(invcovar)
                    #print "TrainingMean[%d]=%r"%(p_num,TrainingMean[p_num])
                    #print "normalized samplePointsList=%r"%(samplePointsList)

                    #print "sizeof invcovar=%r"%(invcovar.size)
                    #print "sizeof np.array(samplePointsList).size=%d"%(np.array(samplePointsList).size)
                    #print "sizeof (TrainingMean[p_num]).size=%d"%((TrainingMean[p_num]).size)

                    cv.WaitKey()
                    '''
                    print "sum(samplePointsList) =%r"%(sum(samplePointsList))
                    #cv.WaitKey()
                    print "sum(TrainingMean[p_num]) =%r"%(sum(TrainingMean[p_num]))
                    #cv.WaitKey()

                    print "(samplePointsList) =%r"%((samplePointsList))
                    #cv.WaitKey()
                    print "(TrainingMean[p_num]) =%r"%((TrainingMean[p_num]))
                    #cv.WaitKey()



                    #test each offset vector with the training Gmean for this landmark calculated accross set of images
                    tmpMah=scipy.spatial.distance.mahalanobis( np.array(samplePointsList), TrainingMean[p_num], invcovar)

                    #print "tmpMah=%r"%(tmpMah)
                    #cv.WaitKey()
                    #cv::Mahalanobis

                    #if this point is a better match is found OR if the first point tested
                    if tmpMah<curBest or curBest<0:
                        curBest = tmpMah;


                        #check if the initial position set is somewhow different to one of the possible options, otherwise don't move & used the initial pos
                        if (g_target_image[scale][yyy, xxx]) != (g_target_image[scale][new_p_centered_yy_offset, new_p_centered_xx_offset]):
                          bestPoint = testInitASM.Point(new_p_centered_xx_offset, new_p_centered_yy_offset)#outter testing index along the whisker
                          savedOffset = offsetT

                          #set previous bestPoint to BLACK, so as to be clearly identifiable
                          if prevBestPoint != testInitASM.Point(-1,-1):
                              cv.Circle(convertedToColouredImg, (prevBestPoint.x, prevBestPoint.y), 1, (0,0,0))#individually sampled points along the normal
                          #set current bestPoint to YELLOW, so as to be clearly identifiable
                          cv.Circle(convertedToColouredImg, (bestPoint.x, bestPoint.y), 1, (0,255,255))#individually sampled points along the normal
                          cv.NamedWindow("convertedToColouredImgTMP", cv.CV_WINDOW_NORMAL)
                          cv.ShowImage("convertedToColouredImgTMP",convertedToColouredImg)
                          #cv.WaitKey()

                          prevBestPoint=bestPoint


                          #if once an new_p_centered_xx_offset is chosen then prefer this over the xxx, regardless if they have the same intensity
                          hack=1
                        else:
                            #The Red painted pixel here means that regardless of the fact that this pixel's is a better fitting one, -compared to the bestindex=-1 we're starting with
                            #However as it has the same value as the center pixels of this profile sample,
                            #so why bother changing to this new one, we can as well stay at the original centered pixel
                            if hack!=1:#if no other point had been previously selected as bestpoint, then get the original mean/starting position
                              bestPoint = testInitASM.Point(xxx,yyy)#leave at initial pos / don't landmark move anywhere new
                              savedOffset = offsetT
                              cv.Circle(convertedToColouredImg, (bestPoint.x, bestPoint.y), 1, (0,0,255))#individually sampled points along the normal




                        #bestEP[i] = V[k];
                        #print "New best Mahalanobis found"
                        #pick the points whose profile g has minimizes the Mahalanobis distance Fit Function

                        #convertedToColouredImg = cv.CreateImage(cv.GetSize( g_target_image[scale]), cv.IPL_DEPTH_8U, 3)
                        #cv.CvtColor(img, convertedToColouredImg, cv.CV_GRAY2BGR)

                        #cv.NamedWindow("convertedToColouredImgTMP", cv.CV_WINDOW_NORMAL)
                        #cv.ShowImage("convertedToColouredImgTMP",convertedToColouredImg)




                        #print "bestPoint=%s"%(bestPoint)
                        #cv.WaitKey()

                        if self.SHOW_EVERY_POSSIBLE_POINT==1:
                            cv.NamedWindow("convertedToColouredImgTMP", cv.CV_WINDOW_NORMAL)
                            cv.ShowImage("convertedToColouredImgTMP",convertedToColouredImg)
                            cv.WaitKey()



        cv.NamedWindow("convertedToColouredImgTMP", cv.CV_WINDOW_NORMAL)
        cv.ShowImage("convertedToColouredImgTMP",convertedToColouredImg)
        cv.WaitKey(1)


        #CHOOSE POINT on the image with the bestPoint Correspondance
        chosenLandmarkPoint=testInitASM.Point(bestPoint.x, bestPoint.y)

        #last best point chosen - for this side .. for this normal

        if self.DEBUG_LINES==1:
            cv.Circle(convertedToColouredImg, (chosenLandmarkPoint.x, chosenLandmarkPoint.y), 5, (255,255,255))
            cv.WaitKey(1)




        if self.STOP_AT_THE_END_OF_EVERY_LANDMARK==1:
            #FINAL best point chosen - for this iteration
            cv.Circle(convertedToColouredImg, (chosenLandmarkPoint.x, chosenLandmarkPoint.y), 1, (255,255,255))
            cv.NamedWindow("convertedToColouredImg", cv.CV_WINDOW_NORMAL)
            cv.ShowImage("convertedToColouredImg",convertedToColouredImg)
            cv.WaitKey(1)
        #print "bestPoint=%r"%(chosenLandmarkPoint)
        #cv.WaitKey()
        return chosenLandmarkPoint, savedOffset


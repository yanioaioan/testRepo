
import numpy, procrupy
import operator
import math
import glob,os
import sys





if(len(sys.argv)<2):
	print "enter 0 to write out precreated sample shapes Before Procrustes Alignment Takes Place OR 1 to write out the aligned shapes After Procrusted have taken place "
	exit()


#shapes = numpy.array([[[1, 1], [5, 5], [10, 3]],	[[2, 2], [7, 7], [15, 5]],	[[-5, -5], [-1, -1], [-10, -3]]])

#mylist=[[-5, 0], [-5, 5], [5, 0], [5, 5]]
#ylist2=[[-10, -8], [-10, 8], [10, -8], [10, 10]]
#mylist3=[[-13, -4], [-13, 8], [13, -6], [13, 6]]


mylist=[]
for i in range(0,100,1):
	a=i*2+math.sin(i)*4.5++math.sin(i)*20
	b=-2*i+math.cos(i)
	
	mylist.append([a,b])
	
print mylist

mylist2=[]
for i in range(0,100,1):
	a=i*1.5+math.sin(i)*25+math.sin(i)*20+100
	b=-2*i+math.cos(i)+10
	
	mylist2.append([a,b])

print mylist2

mylist3=[]
for i in range(0,100,1):
	a=i+math.sin(i)*1.5+math.sin(i)*20+10-100
	b=-2*i+math.cos(i)-10
	
	mylist3.append([a,b])

print mylist3

	
	
shapes = numpy.array([mylist,mylist2,mylist3])

mean_shape = procrupy.generalized_procrustes_analysis(shapes, threshold=0.1)
print mean_shape


from PIL import Image, ImageDraw

im = Image.open("image.png")


datas =im.getdata()
newData = []
for item in datas:
	newData.append((0, 0, 0, 0))
	
im.putdata(newData)

draw = ImageDraw.Draw(im)	


#mylist=[1, 1, 5, 5, 10, 3]
'''
mlist=[]
for i in mylist:
    mlist.append(i[0])
    mlist.append(i[1])
'''
mlist=[]
#This is just to test And be removed
mlistBeforeAligning=[8.000000, 8.000000, 8.000000, 16.000000, 22.000000, 8.000000, 22.000000, 16.000000, 10.000000, 12.000000, 9.000000, 18.000000, 27.000000, 10.000000, 21.000000, 13.000000, 22.000000, 13.000000, 25.000000, 16.000000]#before Weighted GPA
mlistAfterAligning=[8.703079, 8.399187, 8.703079, 15.774956, 21.610675, 8.399187, 21.610675, 15.774956, 10.547021, 12.087072, 9.625050, 17.618899, 26.220531, 10.243129, 20.688704, 13.009043, 21.610675, 13.009043, 24.376588, 15.774956]#after Weighted GPA
#This is just to test And be removed

if (sys.argv[1]=='0'):
	mlist=mlistBeforeAligning
elif (sys.argv[1]=='1'):
	mlist=mlistAfterAligning
	

print mlist

for i in range(0, len(mlist) ,2):
	mlist[i] = mlist[i] + (640/2)
	mlist[i+1] = mlist[i+1] + (480/2)

draw.point((mlist) , fill=(255,0,0))


#mylist2=[2, 2, 7, 7, 15, 5]
'''
mlist2=[]
for i in mylist2:
    mlist2.append(i[0])
    mlist2.append(i[1])

''' 

mlist2=[]
#This is just to test And be removed
mlist2BeforeAligning=[6.000000, 6.000000, 6.000000, 12.000000, 24.000000, 6.000000, 24.000000, 12.000000, 8.000000, 7.000000, 9.000000, 13.000000, 21.000000, 8.000000, 28.000000, 15.000000, 24.000000, 11.000000, 32.000000, 18.000000]#before Weighted GPA
mlist2AfterAligning=[8.107311, 9.743575, 8.178456, 14.469667, 22.285586, 9.530137, 22.356732, 14.256229, 9.694532, 10.507542, 10.553360, 15.221776, 19.946256, 11.141074, 25.543033, 16.571845, 22.344874, 13.468547, 28.729334, 18.887460]#after Weighted GPA

#This is just to test And be removed

if (sys.argv[1]=='0'):
	mlist2=mlist2BeforeAligning
elif (sys.argv[1]=='1'):
	mlist2=mlist2AfterAligning

print mlist2

for i in range(0, len(mlist2) ,2):
	mlist2[i] = mlist2[i] + (640/2)
	mlist2[i+1] = mlist2[i+1] + (480/2)



draw.point((mlist2) , fill=(0,255,0))


#mylist3=[-5, -5, -1, -1, -10, -3]

'''
mlist3=[]
for i in mylist3:
    mlist3.append(i[0])
    mlist3.append(i[1])
'''

mlist3=[]
#This is just to test And be removed
mlist3BeforeAligning=[10.000000, 10.000000, 10.000000, 20.000000, 20.000000, 10.000000, 20.000000, 20.000000, 12.000000, 12.000000, 11.000000, 22.000000, 24.000000, 13.000000, 23.000000, 25.000000, 21.000000, 26.000000, 22.000000, 27.000000]#before Weighted GPA
mlist3AfterAligning=[9.150569, 7.554430, 11.290750, 16.413896, 18.010035, 5.414250, 20.150215, 14.273715, 11.350498, 8.898287, 12.604732, 17.971771, 22.195875, 7.216017, 23.878145, 18.061394, 22.320270, 19.375377, 23.420235, 20.047305]#after Weighted GPA

#This is just to test And be removed

if (sys.argv[1]=='0'):
	mlist3=mlist3BeforeAligning
elif (sys.argv[1]=='1'):
	mlist3=mlist3AfterAligning

print mlist3

for i in range(0, len(mlist3) ,2):
	mlist3[i] = mlist3[i] + (640/2)
	mlist3[i+1] = mlist3[i+1] + (480/2)


draw.point((mlist3) , fill=(0,0,255))

'''
#This is just to test And be removed
shapes = numpy.array([[mlist],[mlist2],[mlist3]])
mean_shape = procrupy.generalized_procrustes_analysis(shapes, threshold=0.1)
print mean_shape
#This is just to test And be removed

myFinalmean_shape=[]
for i in mean_shape:
    myFinalmean_shape.append(i[0])
    myFinalmean_shape.append(i[1])

print myFinalmean_shape

for i in range(0, len(myFinalmean_shape) ,2):
	myFinalmean_shape[i] = myFinalmean_shape[i] + (640/2)
	myFinalmean_shape[i+1] = myFinalmean_shape[i+1] + (480/2)

#draw.point((myFinalmean_shape) , fill=(255,255,0))
'''

#Last computed mean after aligning with Procrustes analysis...
finalMeanShapeMotionPoints=[7.888905, 8.149339, 8.693468, 15.775849, 20.967155, 7.293005, 21.771718, 14.919515, 9.937706, 10.258033, 10.371069, 17.287201, 23.316221, 9.205568, 23.951925, 16.133778, 22.556950, 15.482759, 26.286404, 18.705175]
#Last computed mean after afetr entering __construct_model () ... mean = np.mean(shape_vectors, axis=0)
#finalMeanShapeMotionPoints=[  8.65365279,   8.56573077 ,  9.39076165 , 15.55283968 , 20.63543191,   7.78119134 , 21.37254078 , 14.76830026 , 10.53068385, 10.49763352,  10.92771414,  16.93748182,  22.78755376,   9.53340687 , 23.36996072,  15.88076046 , 22.0919399 ,  15.28432225,  25.50871905 , 18.23657381]
				
for i in range(0, len(finalMeanShapeMotionPoints) ,2):
	finalMeanShapeMotionPoints[i] = finalMeanShapeMotionPoints[i] + (640/2)
	finalMeanShapeMotionPoints[i+1] = finalMeanShapeMotionPoints[i+1] + (480/2)
draw.point((finalMeanShapeMotionPoints) , fill=(255,255,255))

'''
counter=0
#draw mean shape variation whilst aligning
ml=[]
print "meanShapeMotionShape=",meanShapeMotionShape
for i in (meanShapeMotionShape):#only one, for is just for parsing convenience
	for i,p in enumerate(i.pts):		  		  
		  ml[i]=p.x
		  ml[i+1]=p.y
		  



for i in range(0, len(ml),1):
	ml[i] = ml[i] #+ (640/2)
	print "ml[i]=",ml[i]
	
draw.point((ml) , fill=(255,255,255))
'''



del draw

# write to stdout
if (sys.argv[1]=='0'):
	im.save("imageBEFORE.png", "PNG")
elif (sys.argv[1]=='1'):
	im.save("imageAFTER.png", "PNG")



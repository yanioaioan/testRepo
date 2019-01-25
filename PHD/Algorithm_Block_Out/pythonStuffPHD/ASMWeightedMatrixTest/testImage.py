from PIL import Image

img = Image.new( 'RGB', (800,600), "black") # create a new black image
pixels = img.load() # create the pixel map

#for i in range(img.size[0]):    # for every pixel:
    #for j in range(img.size[1]):
    #    pixels[i,j] = (i, j, 100) # set the colour accordingly

'''This will produce an image which is  the same as the shape 1 trained set of landmarks "extracted" '''
'''It resembles the target image'''


#pixels[100, 100 ]= (255, 255, 255)
#pixels[110, 200 ]= (255, 255, 255)
#pixels[210, 100 ]= (255, 255, 255)
#pixels[210, 200 ]= (255, 255, 255)
#pixels[110, 110 ]= (255, 255, 255)
#pixels[120, 210 ]= (255, 255, 255)
#pixels[230, 120 ]= (255, 255, 255)
#pixels[220, 240 ]= (255, 255, 255)
#pixels[220, 250 ]= (255, 255, 255)
#pixels[220, 260 ]= (255, 255, 255)

'''
pixels[249.599, 152.934]= (255, 255, 255)
pixels[246.122, 204.482]= (255, 255, 255)
pixels[249.53, 254.293]= (255, 255, 255)
pixels[254.88, 310.967]= (255, 255, 255)
pixels[274.546, 352.816]= (255, 255, 255)
pixels[296.05, 386.909]= (255, 255, 255)
pixels[324.603, 410.212]= (255, 255, 255)
pixels[372.341, 422.403]= (255, 255, 255)
pixels[421.079, 417.749]= (255, 255, 255)
pixels[452.742, 394.765]= (255, 255, 255)
pixels[476.355, 362.102]= (255, 255, 255)
pixels[492.315, 327.926]= (255, 255, 255)
pixels[504.016, 274.566]= (255, 255, 255)
pixels[514.482, 222.694]= (255, 255, 255)
pixels[519.696, 180.315]= (255, 255, 255)
pixels[504.877, 157.074]= (255, 255, 255)
pixels[476.355, 127.552]= (255, 255, 255)
pixels[442.436, 120.643]= (255, 255, 255)
pixels[420.214, 129.002]= (255, 255, 255)
pixels[441.18, 141.999]= (255, 255, 255)
pixels[472.586, 146.396]= (255, 255, 255)
pixels[272.84, 140.115]= (255, 255, 255)
pixels[302.619, 112.477]= (255, 255, 255)
pixels[354.075, 110.684]= (255, 255, 255)
pixels[374.698, 131.319]= (255, 255, 255)
pixels[338.739, 124.493]= (255, 255, 255)
pixels[306.131, 129.436]= (255, 255, 255)
pixels[300.812, 152.8]= (255, 255, 255)
pixels[324.975, 140.743]= (255, 255, 255)
pixels[354.754, 157.702]= (255, 255, 255)
pixels[324.975, 164.24]= (255, 255, 255)
pixels[324.975, 151.421]= (255, 255, 255)
pixels[482.636, 174.034]= (255, 255, 255)
pixels[460.024, 154.818]= (255, 255, 255)
pixels[430.501, 167.124]= (255, 255, 255)
pixels[454.627, 180.315]= (255, 255, 255)
pixels[456.883, 166.496]= (255, 255, 255)
pixels[372.67, 159.452]= (255, 255, 255)
pixels[362.011, 206.475]= (255, 255, 255)
pixels[349.193, 241.916]= (255, 255, 255)
pixels[352.788, 259.676]= (255, 255, 255)
pixels[388.587, 264.522]= (255, 255, 255)
pixels[421.17, 262.591]= (255, 255, 255)
pixels[426.946, 244.884]= (255, 255, 255)
pixels[416.333, 206.54]= (255, 255, 255)
pixels[412.189, 160.575]= (255, 255, 255)
pixels[369.806, 251.861]= (255, 255, 255)
pixels[409.349, 252.842]= (255, 255, 255)
pixels[324.524, 317.989]= (255, 255, 255)
pixels[351.357, 308.454]= (255, 255, 255)
pixels[370.201, 304.686]= (255, 255, 255)
pixels[384.276, 307.826]= (255, 255, 255)
pixels[396.582, 306.57]= (255, 255, 255)
pixels[414.426, 310.967]= (255, 255, 255)
pixels[430.975, 322.931]= (255, 255, 255)
pixels[417.808, 334.877]= (255, 255, 255)
pixels[399.791, 344.17]= (255, 255, 255)
pixels[382.522, 344.826]= (255, 255, 255)
pixels[359.237, 341.592]= (255, 255, 255)
pixels[342.62, 330.207]= (255, 255, 255)
pixels[354.76, 321.15]= (255, 255, 255)
pixels[384.132, 326.125]= (255, 255, 255)
pixels[404.766, 324.389]= (255, 255, 255)
pixels[404.094, 320.9]= (255, 255, 255)
pixels[384.347, 319.55]= (255, 255, 255)
pixels[354.961, 317.865]= (255, 255, 255)
pixels[384.186, 322.087]= (255, 255, 255)
pixels[394.303, 234.871]= (255, 255, 255)
'''

'''
#rectangle
pixels[105,105]= (255, 255, 255)
pixels[105,205]= (255, 255, 255)
pixels[205,105]= (255, 255, 255)
pixels[205,205]= (255, 255, 255)
'''

#diagonal line
pixels[100,100]= (255, 255, 255)
pixels[150,200]= (255, 255, 255)
pixels[200,300]= (255, 255, 255)
pixels[250,500]= (255, 255, 255)




#img.show()
img.save("imageTarget.png", "PNG")


'''
def drange(start, stop, step):
  r = start
  while r < stop:
    yield r
    r += step
    
dist=drange(0,10,1)
for i in dist:
    print i
'''


import math
'''
x1=10
x2=20
y1=10
y2=20
unitsAway=10
d = math.sqrt((x2-x1)**2 + (y2 - y1)**2) #distance

r = unitsAway / d #segment ratio

x3 = r * x2 + (1 - r) * x1 #find point that divides the segment
y3 = r * y2 + (1 - r) * y1 #into the ratio (1-r):r

print "(x3=%f,y3=%f)"%(x3,y3)
'''


'''Sample along line'''
'''
def get_normal_to_pointP(p,p2):
    # Normal to first point
    x = 0; y = 0; mag = 0

    x = p[0] - p2[0]
    y = p[1] - p2[1]

    mag = math.sqrt(x**2 + y**2)
    return (-y/mag, x/mag)


x1=10
x2=20
y1=10
y2=20

p=[x1,y1]
p2=[x2,y2]

norm=get_normal_to_pointP(p,p2)

x = p[0] - p2[0]
y = p[1] - p2[1]
mag = math.sqrt(x**2 + y**2)

step=(20-10)/mag
point=[0,0]


for i in range(-6,6):
    point[0]=(x1)+i*(step*norm[0])
    point[1]=(y1)+i*(step*norm[1])
    print point
'''





























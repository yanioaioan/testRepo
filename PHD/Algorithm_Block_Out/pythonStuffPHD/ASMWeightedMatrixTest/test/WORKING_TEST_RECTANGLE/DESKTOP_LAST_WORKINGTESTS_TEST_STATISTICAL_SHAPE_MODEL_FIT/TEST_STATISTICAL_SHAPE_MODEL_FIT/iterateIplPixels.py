import cv

path = 'SAMPLING TEST.jpg'
mat = cv.LoadImageM(path, cv.CV_LOAD_IMAGE_UNCHANGED)
x, y = 252, 107
print type(mat)
print mat[y, x]

for x in xrange(mat.cols):
    for y in xrange(mat.rows):
        # multiply all 3 components by 0.5
        #mat[y, x] = tuple(c*0.5 for c in mat[y, x])

        # or multiply only the red component by 0.5
        b, g, r = mat[y, x]
        #mat[y, x] = (b, g, r * 0.5)
        
        print b,g,r

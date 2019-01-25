import cv2
import numpy as np
#from matplotlib import pyplot as plt

img = cv2.imread('grey_image_1 (copy).jpg',0)
img2 = img.copy()
#template = cv2.imread('templateTest.jpg',0)

'''template initialize the vertebrae template in all 10 frames-->then place the actual mean shape precomputed where the tepmlate is'''
#template = cv2.imread('templateTestVertebrae.jpg',0)#worked with TM_CCOEFF_NORMED,TM_CCORR_NORMED,TM_SQDIFF,TM_SQDIFF_NORMED

'''template initialize the vertebrae template in all 10 frames-->then place the actual mean shape precomputed where the tepmlate is'''
template = cv2.imread('templateTestVertebrae2.jpg',0)#worked with TM_CCOEFF_NORMED,TM_CCORR_NORMED,TM_SQDIFF,TM_SQDIFF_NORMED

w, h = template.shape[::-1]

# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
    img = img2.copy()
    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img,top_left, bottom_right, 255, 2)

    '''
    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)

    plt.show()
    '''

    cv2.namedWindow('templateMatced', cv2.WINDOW_NORMAL)
    cv2.imshow('templateMatced',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

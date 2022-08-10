import numpy as np
import cv2 as cv

# 缩放 ################################################################
img = cv.imread('./data/messi5.jpg')
res = cv.resize(img,None,fx=2, fy=2, interpolation = cv.INTER_CUBIC)
cv.imshow('orignal', img)
cv.imshow('res1', res)
#或者
height, width = img.shape[:2]
res = cv.resize(img,(2*width, 2*height), interpolation = cv.INTER_CUBIC)
cv.imshow('res2', res)

# 平移 ################################################################
img = cv.imread('./data/messi5.jpg',0)
rows,cols = img.shape
M = np.float32([[1,0,100],[0,1,50]])
dst = cv.warpAffine(img,M,(cols,rows))
cv.imshow('dst1',dst)

# 旋转 ################################################################
img = cv.imread('./data/messi5.jpg',0)
rows,cols = img.shape
# cols-1 和 rows-1 是坐标限制
M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),90,1)
dst = cv.warpAffine(img,M,(cols,rows))
cv.imshow('dst2',dst)

# 仿射变换 ##############################################################
img = cv.imread('drawing.png')
rows,cols,ch = img.shape
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
M = cv.getAffineTransform(pts1,pts2)
dst = cv.warpAffine(img,M,(cols,rows))
cv.imshow('Input',img)
cv.imshow('Output',dst)

# 透视变换 ##############################################################
img = cv.imread('./data/sudoku.png')
rows,cols,ch = img.shape
pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
M = cv.getPerspectiveTransform(pts1,pts2)
dst = cv.warpPerspective(img,M,(cols,rows))
cv.imshow('Input2',img)
cv.imshow('Output2',dst)

cv.waitKey(0)
cv.destroyAllWindows()
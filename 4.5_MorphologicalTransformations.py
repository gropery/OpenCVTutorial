import cv2 as cv
import numpy as np

img = cv.imread('j.png',0)
cv.imshow('origin', img)

# 矩形内核
kernel = np.ones((5,5),np.uint8)

# 侵蚀
erosion = cv.erode(img,kernel,iterations = 1)
cv.imshow('erosion', erosion)

# 扩张
dilation = cv.dilate(img,kernel,iterations = 1)
cv.imshow('dilation', dilation)

# 开运算(先侵蚀然后扩张)
opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
cv.imshow('opening', opening)

# 闭运算(先扩张然后侵蚀)
closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
cv.imshow('closing', closing)

# 形态学梯度(扩张和侵蚀之间的区别)
gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)
cv.imshow('gradient', gradient)

# 顶帽(输入图像和图像开运算之差)
tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
cv.imshow('tophat', tophat)

#  黑帽(输入图像和图像闭运算之差)
blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)
cv.imshow('blackhat', blackhat)

# 矩形内核
print(cv.getStructuringElement(cv.MORPH_RECT,(5,5)))
# array([[1, 1, 1, 1, 1],
#        [1, 1, 1, 1, 1],
#        [1, 1, 1, 1, 1],
#        [1, 1, 1, 1, 1],
#        [1, 1, 1, 1, 1]], dtype=uint8)
# 椭圆内核
print(cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5)))
# array([[0, 0, 1, 0, 0],
#        [1, 1, 1, 1, 1],
#        [1, 1, 1, 1, 1],
#        [1, 1, 1, 1, 1],
#        [0, 0, 1, 0, 0]], dtype=uint8)
# 十字内核
print(cv.getStructuringElement(cv.MORPH_CROSS,(5,5)))
# array([[0, 0, 1, 0, 0],
#        [0, 0, 1, 0, 0],
#        [1, 1, 1, 1, 1],
#        [0, 0, 1, 0, 0],
#        [0, 0, 1, 0, 0]], dtype=uint8)

cv.waitKey(0)
cv.destroyAllWindows


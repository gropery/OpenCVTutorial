import numpy as np
import cv2 as cv

# im = cv.imread('./data/sudoku.png')
im = cv.imread('test.png')
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

# 应用阈值处理灰度原图
adaptiveMeanThresh = cv.adaptiveThreshold(imgray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 21, 8)
cv.imshow('Adaptive Mean', adaptiveMeanThresh)

# 应用canny边缘检测处理灰度原图
edges = cv.Canny(imgray, 20, 115)
cv.imshow('edges', edges)

# 寻找轮廓前，应用阈值处理，或者canny边缘处理原图
imgContourOrigin = adaptiveMeanThresh

# 寻找轮廓
contours, hierarchy = cv.findContours(imgContourOrigin, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# 绘制所有轮廓
imgAllContours = cv.drawContours(im, contours, -1, (0,255,0), 3)
cv.imshow('imgAllContours', imgAllContours)

# 绘制第四个单独轮廓
img4thContours = cv.drawContours(im, contours, 3, (0,255,0), 3)
cv.imshow('img4thContours', img4thContours)

# 绘制第四个单独轮廓(大多数情况下，更有用)
cnt = contours[4]
img4thContours2 = cv.drawContours(im, [cnt], 0, (0,255,0), 3)
cv.imshow('img4thContours2', img4thContours2)

cv.waitKey(0)
cv.destroyAllWindows()

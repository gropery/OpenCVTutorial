import cv2 as cv
import numpy as np

img = cv.imread(cv.samples.findFile('./data/sudoku.png'))
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray,50,150,apertureSize = 3)

# 霍夫曼变换 ########################################################
img1 = img.copy()
lines = cv.HoughLines(edges,1,np.pi/180,200)
for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    print('x1=',x1)
    y1 = int(y0 + 1000*(a))
    print('y1=',y1)
    x2 = int(x0 - 1000*(-b))
    print('x2=',x2)
    y2 = int(y0 - 1000*(a))
    print('y2=',y2)
    cv.line(img1,(x1,y1),(x2,y2),(0,0,255),2)
cv.imshow('HoughLines',img1)

# 概率霍夫变换 ########################################################
img2 = img.copy()
lines = cv.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
for line in lines:
    x1,y1,x2,y2 = line[0]
    cv.line(img2,(x1,y1),(x2,y2),(0,255,0),2)
cv.imshow('HoughLinesP',img2)

cv.waitKey(0)
cv.destoryAllWindow()
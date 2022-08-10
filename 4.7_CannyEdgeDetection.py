import numpy as np
import cv2 as cv


def nothing(x):
    pass

# img = cv.imread('./data/messi5.jpg', 0)
img = cv.imread('./data/sudoku.png', 0)

cv.namedWindow('edge')
cv.createTrackbar('maxVal', 'edge', 115, 255, nothing)
cv.createTrackbar('minVal', 'edge', 20, 255, nothing)
cv.createTrackbar('l2gradient', 'edge', 0, 1, nothing)

while True:
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break
    # 得到2条轨迹的当前位置
    maxVal = cv.getTrackbarPos('maxVal', 'edge')
    minVal = cv.getTrackbarPos('minVal', 'edge')
    l2gradient = cv.getTrackbarPos('l2gradient', 'edge')
    edges = cv.Canny(img, minVal, maxVal, L2gradient=l2gradient)

    cv.imshow('origin', img)
    cv.imshow('edge', edges)

cv.destroyAllWindow()

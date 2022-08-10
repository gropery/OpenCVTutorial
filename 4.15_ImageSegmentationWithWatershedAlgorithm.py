import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('./data/coins.png')
cv.imshow('originImg', img)

# Otsu二值化
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
cv.imshow('Otsu', thresh)

# 噪声去除-开运算
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

# 确定背景区域-扩张
sure_bg = cv.dilate(opening, kernel, iterations=3)
# 寻找前景区域-侵蚀(由于硬币间有接触，因此最好使用距离变换并应用适当的阈值来找前景区域)
sure_fg1 = cv.erode(opening, kernel, iterations=3)
cv.imshow('fore/back ground', np.hstack((sure_fg1, sure_bg)))

# 寻找前景区域-距离变换并应用适当的阈值-得到的dist_transform和sure_fg2为‘特殊’的img
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
cv.imshow('new fore ground', sure_fg)

# 找到未知区域(边界)
unknown = cv.subtract(sure_bg, sure_fg)
cv.imshow('unknown', unknown)

# 类别标记
ret, markers = cv.connectedComponents(sure_fg)
# connectedComponents 用 0 标记图像的背景，然后其他对象用从 1 开始的整数标记。
# 而分水岭算法中0为位置区域，为了统一，需要为所有的标记(对象)加1
markers = markers + 1
# 现在让所有的未知区域为0
markers[unknown == 255] = 0

# 使用分水岭算法
markers = cv.watershed(img, markers)
img[markers == -1] = [255, 0, 0]  # 标记中-1的像素即为边界
cv.imshow('Result', img)

cv.waitKey(0)
cv.destroyAllWindows()

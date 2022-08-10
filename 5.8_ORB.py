# ORB基本上是FAST关键点检测器和Brief描述符的融合，并进行了许多修改以增强性能

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('./data/blox.jpg',0)
# 初始化ORB检测器
orb = cv.ORB_create()
# 用ORB寻找关键点
kp = orb.detect(img,None)
# 用ORB计算描述符
kp, des = orb.compute(img, kp)
# 仅绘制关键点的位置，而不绘制大小和方向
img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)

cv.imshow('img2', img2)

cv.waitKey(0)
cv.destroyAllWindows()

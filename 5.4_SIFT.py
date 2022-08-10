# SIFT尺度不变特征变换

import numpy as np
import cv2 as cv

img = cv.imread('./data/home.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)

# 构造SIFT对象
sift = cv.xfeatures2d.SIFT_create()

# 寻找关键点
kp = sift.detect(gray, None)
# 画关键点，输入源图像维gray，输出图像为img
img1 = cv.drawKeypoints(gray, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow('sift_keypoints', img1)

# 通过关键点计算描述符
kp,des = sift.compute(gray,kp)
print(len(kp))
print(des.shape)

# 也可直接使用以下函数，直接找到关键点和描述符
# 寻找关键点
kp, des = sift.detectAndCompute(gray,None)
print(len(kp))
print(des.shape)
# 画关键点
img2 = cv.drawKeypoints(gray, kp, None)
cv.imshow('sift_keypoints2', img2)

cv.waitKey(0)
cv.destroyAllWindows()


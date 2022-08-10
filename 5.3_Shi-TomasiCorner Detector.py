# Shi-tomas拐角检测器和益于跟踪的特征

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('./data/blox.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print(gray.shape)

# 返回找到的25个2维x,y数组
corners = cv.goodFeaturesToTrack(gray, 25, 0.01, 10)
corners = np.int0(corners)
print(corners.shape)

# 找到25个最佳的弯角
for i in corners:
    x, y = i.ravel()
    cv.circle(img, (x, y), 3, 255, -1)
plt.imshow(img), plt.show()

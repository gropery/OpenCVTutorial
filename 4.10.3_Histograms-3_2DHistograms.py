import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('./data/home.jpg')
hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)

# Numpy 方式获取二维直方图
h, s, v = cv.split(hsv)
hist, xbins, ybins = np.histogram2d(h.ravel(),s.ravel(),[180,256],[[0,180],[0,256]])

# Matplotlib 方式绘制(X轴显示饱和度S值，Y轴显示色相H值)
plt.subplot(121), plt.imshow(img, 'gray')
plt.subplot(122), plt.imshow(hist,interpolation = 'nearest')

# OpenCV 方式获取二维直方图
hist = cv.calcHist( [hsv], [0, 1], None, [180, 256], [0, 180, 0, 256] )

# cv.imshow() 方式绘制
print(hist.shape)
cv.imshow('cvshow', hist)

plt.show()
cv.waitKey(0)
cv.destroyAllWindows()


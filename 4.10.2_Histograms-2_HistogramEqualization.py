import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# numpy 方式,直方图均衡 ###################################################
img = cv.imread('./data/aero1.jpg', 0)
hist, bins = np.histogram(img.flatten(), 256, [0, 256])  # 将2维灰度图转换成1维,再统计得到hist[256,1] bins=256
cdf = hist.cumsum()  # 累加前n元素的值作为新元素的值的[256,1]
cdf_normalized = cdf * float(hist.max()) / cdf.max()
plt.subplot(321)
plt.imshow(img, 'gray')
plt.subplot(322)
plt.plot(cdf_normalized, color='b')
plt.hist(img.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc='upper left')

# 应用直方图均衡公式得到img2
cdf_m = np.ma.masked_equal(cdf, 0)
cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
cdf = np.ma.filled(cdf_m, 0).astype('uint8')
img2 = cdf[img]

hist, bins = np.histogram(img2.flatten(), 256, [0, 256])  # 将2维灰度图转换成1维,再统计得到hist[256,1] bins=256
cdf = hist.cumsum()  # 累加前n元素的值作为新元素的值的[256,1]
cdf_normalized = cdf * float(hist.max()) / cdf.max()
plt.subplot(323)
plt.imshow(img2, 'gray')
plt.subplot(324)
plt.plot(cdf_normalized, color='b')
plt.hist(img2.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc='upper left')


# opencv 方式,直方图均衡 ###################################################
img = cv.imread('./data/aero1.jpg', 0)
equ = cv.equalizeHist(img)
res = np.hstack((img, equ))  # stacking images side-by-side
cv.imshow('res', res)

# CLAHE算法 ##############################################################
# 当图像的直方图限制在特定区域时，直方图均衡化效果很好
# 当直方图覆盖较大区域（即同时存在亮像素和暗像素）的强度变化较大的地方，效果不好
img3 = cv.imread('./data/tsukuba_l.png',0)
hist, bins = np.histogram(img3.flatten(), 256, [0, 256])  # 将2维灰度图转换成1维,再统计得到hist[256,1] bins=256
cdf = hist.cumsum()  # 累加前n元素的值作为新元素的值的[256,1]
cdf_normalized = cdf * float(hist.max()) / cdf.max()
plt.subplot(325)
plt.imshow(img3, 'gray')
plt.subplot(326)
plt.plot(cdf_normalized, color='b')
plt.hist(img3.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc='upper left')

# 直方图均衡算法
equ = cv.equalizeHist(img3)

# CLAHE算法
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img3)

# 将原图-直方图均衡-CLAHE算法图堆叠在一张图里面显示
res2 = np.hstack((img3, equ))  # stacking images side-by-side
res3 = np.hstack((res2, cl1))
cv.imshow('res3', res3)

plt.show()
cv.waitKey(0)
cv.destroyAllWindows()

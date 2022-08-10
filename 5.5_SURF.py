# SURF简介（加速的强大功能）

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('./data/butterfly.jpg', 0)

# 创建SURF对象。你可以在此处或以后指定参数。#################################
# 这里设置海森矩阵的阈值为400
surf = cv.xfeatures2d.SURF_create(400)

# 直接查找关键点和描述符
kp, des = surf.detectAndCompute(img, None)
print(len(kp))  # 699

# 检查海森矩阵阈值 #######################################################
print(surf.getHessianThreshold())
# 将其设置为50000。记住，它仅用于表示图片。
# 在实际情况下，最好将值设为300-500
surf.setHessianThreshold(50000)

# 再次计算关键点并检查其数量。最好数量小于50
kp, des = surf.detectAndCompute(img, None)
print(len(kp))  # 47

# 画关键点
img2 = cv.drawKeypoints(img, kp, None, (255, 0, 0), 4)
plt.imshow(img2)

# 现在应用U-SURF，以便它不会找到方向(速度更快) ################################
# 检查flag标志，如果为False(默认)，则将其设置为True
print(surf.getUpright())
surf.setUpright(True)

# 重新计算特征点并绘制
kp = surf.detect(img, None)
img3 = cv.drawKeypoints(img, kp, None, (255, 0, 0), 4)
plt.imshow(img3)

plt.show()
# 检查描述符的大小，如果只有64维，则将其更改为128 ################################
# 找到算符的描述
print(surf.descriptorSize())  # 64

# 获取flag “extened” 为False。
surf.getExtended()

# 因此，将其设为True即可获取128尺寸的描述符。
surf.setExtended(True)
kp, des = surf.detectAndCompute(img, None)
print(surf.descriptorSize())  # 128
print(des.shape)  # (47, 128)

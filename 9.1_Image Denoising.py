import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# cv.fastNlMeansDenoisingColored() 用于消除彩色图像中的噪点 #########################
# img = cv.imread('noisy2.png')
# dst = cv.fastNlMeansDenoisingColored(img,None,10,10,7,21)
# plt.subplot(121),plt.imshow(img)
# plt.subplot(122),plt.imshow(dst)
# plt.show()

# cv.fastNlMeansDenoisingMulti() 处理在短时间内捕获的图像序列（灰度图像）###############
cap = cv.VideoCapture('./data/vtest.avi')
# 创建5个帧的列表
img = [cap.read()[1] for i in range(5)]
print('img.shape=',np.array(img).shape)

# 将所有转化为灰度
gray = [cv.cvtColor(i, cv.COLOR_BGR2GRAY) for i in img]
print('gray.shape=',np.array(gray).shape)

# 将所有转化为float64
gray = [np.float64(i) for i in gray]

# 创建具有正态分布的随机噪声
noise = np.random.randn(*gray[1].shape)*10
print(*gray[1].shape)

# 在所有5帧图像上添加噪声
noisy = [i+noise for i in gray]

# 对每一帧图像中像素值归一化为0-255，且转化为unit8
noisy = [np.uint8(np.clip(i,0,255)) for i in noisy]

# noisy总共5帧图像，索引为 0，1，2，3，4
# imgToDenoiseIndex = 2: 对索引为2的帧进行噪声去除，
# temporalWindowSize =5: 指定用于降噪的附近帧的数量为5
# h = 4: 滤波器强度为4
# templateWindowSize=7：应为奇数。
# searchWindowSize=35：应为奇数。
dst = cv.fastNlMeansDenoisingMulti(noisy, 2, 5, None, 4, 7, 35)

# plot 索引号为2的 原图/添加噪声后的图/去噪后的图
plt.subplot(131),plt.imshow(gray[2],'gray')
plt.subplot(132),plt.imshow(noisy[2],'gray')
plt.subplot(133),plt.imshow(dst,'gray')
plt.show()

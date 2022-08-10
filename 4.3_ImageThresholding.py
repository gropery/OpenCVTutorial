import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 简单阈值 #######################################################
# img = cv.imread('./data/gradient.png',0)
# ret,thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
# ret,thresh2 = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
# ret,thresh3 = cv.threshold(img,127,255,cv.THRESH_TRUNC)
# ret,thresh4 = cv.threshold(img,127,255,cv.THRESH_TOZERO)
# ret,thresh5 = cv.threshold(img,127,255,cv.THRESH_TOZERO_INV)
# titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
# images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
# for i in range(6):
#     cv.imshow(titles[i], images[i])

# 自适应阈值 #######################################################
img = cv.imread('./data/sudoku.png', 0)
rows,cols = img.shape
ret,global_thresh = cv.threshold(img,127,255,cv.THRESH_BINARY)
cv.imshow('Global(v=127)', global_thresh)
adaptive_thresh_mean = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 21, 8)
cv.imshow('Adaptive Mean', adaptive_thresh_mean)
adaptive_thresh_gaussian = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 8)
cv.imshow('Adaptive Gaussian', adaptive_thresh_mean)

cv.waitKey(0)
cv.destroyAllWindows()

# # Otsu的二值化 #######################################################
# img = cv.imread('noisy2.png',0)
# # 全局阈值
# ret1,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
# # Otsu阈值
# ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# # 高斯滤波后再采用Otsu阈值
# blur = cv.GaussianBlur(img,(5,5),0)
# ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# # 绘制所有图像及其直方图
# images = [img, 0, th1,
#           img, 0, th2,
#           blur, 0, th3]
# titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
#           'Original Noisy Image','Histogram',"Otsu's Thresholding",
#           'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
# for i in range(3):
#     plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
#     plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
#     plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
#     plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
#     plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
#     plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
# plt.show()


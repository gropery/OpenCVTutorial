import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# 2D巻积平均滤波 #############################################
img = cv.imread('./data/opencv-logo.png')
kernel = np.ones((5,5),np.float32)/25
dst = cv.filter2D(img,-1,kernel)
plt.subplot(5, 2, 1),plt.imshow(img),plt.title('Original'),plt.xticks([]), plt.yticks([])
plt.subplot(5, 2, 2),plt.imshow(dst),plt.title('Averaging'),plt.xticks([]), plt.yticks([])

# 平均滤波 ##################################################
img = cv.imread('./data/opencv-logo-white.png')
blur = cv.blur(img,(5,5))
plt.subplot(5, 2, 3),plt.imshow(img),plt.title('Original'),plt.xticks([]), plt.yticks([])
plt.subplot(5, 2, 4),plt.imshow(blur),plt.title('Blurred'),plt.xticks([]), plt.yticks([])

# 高斯滤波 ##################################################
blur = cv.GaussianBlur(img,(5,5),0)
plt.subplot(5, 2, 5),plt.imshow(img),plt.title('Original'),plt.xticks([]), plt.yticks([])
plt.subplot(5, 2, 6),plt.imshow(blur),plt.title('GaussianBlur'),plt.xticks([]), plt.yticks([])

# 中位滤波 ##################################################
img = cv.imread('opencv-logo-mediaBlurOrigin.jpg')
median = cv.medianBlur(img,5)
plt.subplot(5, 2, 7),plt.imshow(img),plt.title('Original'),plt.xticks([]), plt.yticks([])
plt.subplot(5, 2, 8),plt.imshow(median),plt.title('medianBlur'),plt.xticks([]), plt.yticks([])

# 双边滤波 ##################################################
img = cv.imread('bilateral-origin.jpg')
blur = cv.bilateralFilter(img,9,75,75)
plt.subplot(5, 2, 9),plt.imshow(img),plt.title('Original'),plt.xticks([]), plt.yticks([])
plt.subplot(5, 2, 10),plt.imshow(blur),plt.title('bilateralFilter'),plt.xticks([]), plt.yticks([])


plt.show()
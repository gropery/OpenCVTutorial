import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# 灰度图按强度绘制直方图 #################################
# img = cv.imread('./data/home.jpg',0)
# cv.imshow('imgOrigin', img)
#
# # matplotlib 绘图
# plt.hist(img.ravel(),256,[0,256]);
# plt.show()
#
# # opencv 绘图
# imgDrawLine = np.zeros((300, 256, 3))                       # 新建一个彩色像素的画布
# hist_item = cv.calcHist([img], [0], None, [256], [0, 256])  # 计算img图中的直方图,得到256x1的数组
# cv.normalize(hist_item, hist_item, 0, 255, cv.NORM_MINMAX)  # 将数组中元素范围归一化为0-255之间
# hist = np.int32(np.around(hist_item))                       # 将小数四舍五入至整数,再类型转换为int32
# for x, y in enumerate(hist):                                # 枚举每个y,x从0开始
#     cv.line(imgDrawLine, (x, 0), (x, y[0]), (255, 255, 255)) # 注意这里y为[]数组类型,需要使用[]符号解码
# im = np.flipud(imgDrawLine)                                  # 上下翻转图片
# cv.imshow('im', im)
#
# cv.waitKey(0)
# cv.destroyAllWindows()

# 彩色图按BGR绘制直方图 ##################################
img = cv.imread('./data/home.jpg')
cv.imshow('imgOrigin', img)

# matplotlib 绘图
color = ('b','g','r')
for ch,col in enumerate(color):
    histr = cv.calcHist([img],[ch],None,[256],[0,256])           # 分别计算B,G,R,3种像素的统计数据
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()

# opencv 绘图
bins = np.arange(256).reshape(256, 1)                            # 生成一个0~255值的序列,并将其转换成[256行,1列]的形状
imgDrawLine = np.zeros((300, 256, 3))                            # 新建一个彩色像素的画布
color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
for ch, col in enumerate(color):
    hist_item = cv.calcHist([img], [ch], None, [256], [0, 256])  # 分别计算B,G,R,3种像素的统计数据,得到3个[256x1]的数组
    cv.normalize(hist_item, hist_item, 0, 255, cv.NORM_MINMAX)   # 将数组中元素范围归一化为0-255之间(小于画布的300就行)
    hist = np.int32(np.around(hist_item))                        # 将小数四舍五入至整数,再类型转换为int32
    pts = np.int32(np.column_stack((bins, hist)))                # 按列合并bins和hist
    cv.polylines(imgDrawLine, [pts], False, col)                 # 绘制曲线
im = np.flipud(imgDrawLine)
cv.imshow('im', im)

cv.waitKey(0)
cv.destroyAllWindows()

# 掩码的作用 ############################################
# img = cv.imread('./data/home.jpg',0)
# # create a mask
# mask = np.zeros(img.shape[:2], np.uint8)
# mask[100:300, 100:400] = 255
# masked_img = cv.bitwise_and(img,img,mask = mask)
# # 计算掩码区域和非掩码区域的直方图
# # 检查作为掩码的第三个参数
# hist_full = cv.calcHist([img],[0],None,[256],[0,256])
# hist_mask = cv.calcHist([img],[0],mask,[256],[0,256])
# plt.subplot(221), plt.imshow(img, 'gray')
# plt.subplot(222), plt.imshow(mask,'gray')
# plt.subplot(223), plt.imshow(masked_img, 'gray')
# plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
# plt.xlim([0,256])
# plt.show()


import numpy as np
import cv2 as cv

# # 引入原图
# img = cv.imread('./data/messi5.jpg')
# cv.imshow('origin', img)
#
# # 高斯金字塔 ########################################
# # 降低一层分辨率
# lower_reso_d1 = cv.pyrDown(img)
# cv.imshow('lower_reso_d1', lower_reso_d1)
#
# # 再降低一层分辨率
# lower_reso_d2 = cv.pyrDown(lower_reso_d1)
# cv.imshow('lower_reso_d2', lower_reso_d2)
#
# # 提升一层分辨率
# high_reso_1 = cv.pyrUp(lower_reso_d2)
# cv.imshow('high_reso_1', high_reso_1)
#
# # 再提升一层分辨率
# high_reso_0 = cv.pyrUp(lower_reso_d1)
# cv.imshow('high_reso_0', high_reso_0)
#
# # 拉普拉斯金字塔 #####################################
# lp0 = cv.subtract(img, high_reso_0) # 原图 - 降低1次再提升1次后的图高斯图
# cv.imshow('lp0', lp0)
#
# row,col,ch = lower_reso_d1.shape
# high_reso_1 = cv.resize(high_reso_1,(col,row))
# print(high_reso_1.shape)
# print(lower_reso_d1.shape)
#
# lp1 = cv.subtract(lower_reso_d1, high_reso_1) # 高斯图 - 降低再提升后的图
# cv.imshow('lp1', lp1)

# 使用金字塔进行图像融合 ###############################
A = cv.imread('apple.jpg')
B = cv.imread('orange.jpg')
# 生成A的高斯金字塔
G = A.copy()
gpA = [G]
for i in range(6):
    G = cv.pyrDown(G)
    gpA.append(G)
# 生成B的高斯金字塔
G = B.copy()
gpB = [G]
for i in range(6):
    G = cv.pyrDown(G)
    gpB.append(G)
# 生成A的拉普拉斯金字塔
lpA = [gpA[5]]
for i in range(5,0,-1):
    GE = cv.pyrUp(gpA[i])
    row,col,ch = GE.shape
    gpA[i-1] = cv.resize(gpA[i-1], (col, row))
    L = cv.subtract(gpA[i-1],GE)
    lpA.append(L)
# 生成B的拉普拉斯金字塔
lpB = [gpB[5]]
for i in range(5,0,-1):
    GE = cv.pyrUp(gpB[i])
    row,col,ch = GE.shape
    gpB[i-1] = cv.resize(gpB[i-1], (col, row))
    L = cv.subtract(gpB[i-1],GE)
    lpB.append(L)
# 现在在每个级别中添加左右两半图像
LS = []
for la,lb in zip(lpA,lpB):
    rows,cols,dpt = la.shape
    ls = np.hstack((la[:,0:int(cols/2)], lb[:,int(cols/2):]))
    LS.append(ls)
# 现在重建
ls_ = LS[0]
for i in range(1,6):
    ls_ = cv.pyrUp(ls_)
    row,col,ch = ls_.shape
    LS[i] = cv.resize(LS[i], (col, row))
    ls_ = cv.add(ls_, LS[i])
# # 图像与直接连接的每一半
# real = np.hstack((A[:,:int(cols/2)],B[:,int(cols/2):]))
cv.imshow('Pyramid_blending2', ls_)
# cv.imshow('Direct_blending', real)

cv.waitKey(0)
cv.destroyAllWindow()

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time

img1 = cv.imread('./data/box.png',cv.IMREAD_GRAYSCALE)          # 索引图像
img2 = cv.imread('./data/box_in_scene.png',cv.IMREAD_GRAYSCALE) # 训练图像

# ORB描述符+BFMatcher进行Brute-Force匹配 ###########################################
t1 = time.time()
# 初始化ORB检测器
orb = cv.ORB_create()
# 基于ORB找到关键点和描述符
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# 创建BF匹配器的对象(ORB，BRIEF，BRISK 时参数用cv.NORM_HAMMING，可选crossCheck=True 精度更高)
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# 匹配描述符
matches = bf.match(des1,des2)  # len(matches) = 148
print(matches)
# 根据距离的升序排序，以使最佳匹配(低距离)排在前面
matches = sorted(matches, key = lambda x:x.distance)
# 绘制前10的匹配项
print(matches)
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
t2 = time.time()
print(t2-t1)

# SIFT描述符+BFMatcher进行Brute-Force匹配 #########################################
t1 = time.time()
# 初始化SIFT描述符
sift = cv.xfeatures2d.SIFT_create()
# 基于SIFT找到关键点和描述符
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# 默认参数初始化BF匹配器对象(SIFT，SURF 时参数用cv.NORM_L2, 可选crossCheck默认为False)
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)   # len(matches) = 604
print(matches)
# 根据Lowe的论文进行比例测试
# 挑选第一个距离小于第二个距离0.75倍的match
# 注意: m 一定是小于 n的
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
        # good.append(m)   # 也可用drawMatches绘制
print(good)
# cv.drawMatchesKnn将列表作为匹配项。
img4 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# 也可用drawMatches绘制
# img4 = cv.drawMatches(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
t2 = time.time()
print(t2-t1)

# SIFT描述符+FLANN匹配器 ####################################################
t1 = time.time()
# 初始化SIFT描述符
sift = cv.xfeatures2d.SIFT_create()
# 基于SIFT找到关键点和描述符
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# FLANN的参数(SIFT,SURF使用以下参数)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # 或传递一个空字典，指定索引中递归遍历的次数
# 初始化flann匹配器
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
# 只需要绘制好匹配项，因此创建一个掩码
matchesMask = [[0,0] for i in range(len(matches))]
# 根据Lowe的论文进行比例测试
for i,(m,n) in enumerate(matches):
    if m.distance < 0.75*n.distance:
        matchesMask[i]=[1,0]
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv.DrawMatchesFlags_DEFAULT)
img5 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
t2 = time.time()
print(t2-t1)

# ORB描述符+FLANN匹配器 ####################################################
t1 = time.time()
# 初始化ORB检测器
orb = cv.ORB_create()
# 基于ORB找到关键点和描述符
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# FLANN的参数(ORB使用以下参数)，文档建议使用带注释的值，但某些情况下其他值也正常工作
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
search_params = dict(checks=50)   # 或传递一个空字典，指定索引中递归遍历的次数
# 初始化flann匹配器
flann = cv.FlannBasedMatcher(index_params,search_params)
# 匹配描述符
matches = flann.match(des1,des2)
# 根据距离的升序排序，以使最佳匹配(低距离)排在前面
matches = sorted(matches, key = lambda x:x.distance)
# 绘制前10的匹配项
img6 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
t2 = time.time()
print(t2-t1)

cv.imshow('img3', img3)
cv.imshow('img4', img4)
cv.imshow('img5', img5)
cv.imshow('img6', img6)

cv.waitKey()
cv.destroyAllWindows()

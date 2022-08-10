import numpy as np
import cv2 as cv

# 特征距 ################################################################
# # 加载图片
# im = cv.imread('./data/star.jpg')
# img = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
#
# # 阈值判定
# ret,thresh = cv.threshold(img,127,255,0)
# cv.imshow('thresh', thresh)
#
# # 寻找轮廓
# contours,hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
#
# # 绘制第0(绿色)和第1个轮廓(红色)
# cnt = contours[0]
# imgContours = cv.drawContours(im, [cnt], 0, (0,255,0), 3)
# cnt = contours[1]
# imgContours = cv.drawContours(imgContours, [cnt], 0, (0,0,255), 3)
# cv.imshow('imgContours', imgContours)
#
# # 打印第一个轮廓特征矩
# M = cv.moments(cnt)
# print( M )
#
# # 质心
# cx = int(M['m10']/M['m00'])
# cy = int(M['m01']/M['m00'])
# print('cx=',cx,' cy=',cy)
#
# # 面积
# area1 = cv.contourArea(cnt)
# area2 = M['m00']
# print('area1=', area1, 'area2=', area2)
#
# # 周长
# perimeter = cv.arcLength(cnt,True)
# print('perimeter=', perimeter)

# # 轮廓近似 ################################################################
# # 加载原图
# im = cv.imread('Approximation.png')
# cv.imshow('im', im)
#
# # 阈值判定
# img = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
# ret,thresh = cv.threshold(img,127,255,0)
#
# # 寻找轮廓
# contours,hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
#
# # 绘制第0条轮廓(红色显示)
# im0 = im.copy()    # cv.drawContours 会修改原图，所以要拷贝留作后用
# cnt = contours[0]
# imgContours = cv.drawContours(im0, [cnt], 0, (0,0,255), 3)
# cv.imshow('imgContours', imgContours)
#
# # 轮廓近似度10%(绿色显示)
# im1 = im.copy()
# epsilon10 = 0.1*cv.arcLength(cnt,True)
# approx10 = cv.approxPolyDP(cnt,epsilon10,True)
# imgApprox10 = cv.drawContours(im1, [approx10], 0, (0,255,0), 3)
# cv.imshow('imgApprox-10%', imgApprox10)
#
# # 轮廓近似度1%(蓝色显示)
# im2 = im.copy()
# epsilon1 = 0.01*cv.arcLength(cnt,True)
# approx1 = cv.approxPolyDP(cnt,epsilon1,True)
# imgApprox1 = cv.drawContours(im2, [approx1], 0, (255,0,0), 3)
# cv.imshow('imgApprox-%1', imgApprox1)

# 轮廓凸包 ################################################################
# 加载原图
im = cv.imread('hand.png')
cv.imshow('im', im)

# 阈值判定
img = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
ret,thresh = cv.threshold(img,127,255,0)

# 寻找轮廓
contours,hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

# 绘制第0条轮廓(红色显示)
im0 = im.copy()    # cv.drawContours 会修改原图，所以要拷贝留作后用
cnt = contours[0]
imgContours = cv.drawContours(im0, [cnt], 0, (0,0,255), 3)
cv.imshow('imgContours', imgContours)

# 绘制凸包轮廓(绿色显示)
im1 = im.copy()

hull = cv.convexHull(cnt)                             # 获取凸包
print(hull)
hullindx = cv.convexHull(cnt, returnPoints = False)   # 获取凸包索引
print(hullindx)
print(cnt[hullindx[0]])                               # 凸包索引[0]对应于凸包[0][x,y]

imgConvexHull = cv.drawContours(im1, [hull], 0, (0,255,0), 3)
cv.imshow('imgConvexHull', imgConvexHull)

# 检查曲线是否为凸多边形
k = cv.isContourConvex(cnt)   # 手指曲线-False
print('isContourConvex=', k)
k = cv.isContourConvex(hull)  # 凸轮廓-True
print('isContourConvex=', k)

cv.waitKey(0)
cv.destroyAllWindows()

# # 边界矩形 ################################################################
# # 加载原图
# im = cv.imread('./data/thunder.png')
# cv.imshow('im', im)
#
# # 阈值判定
# img = cv.cvtColor(im, cv.COLOR_BGR2GRAY) # 注意这里原图为白底黑图，要反转一下
# ret,thresh = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
#
# # 寻找轮廓
# contours,hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
#
# # 绘制第0条轮廓(红色显示)
# im0 = im.copy()    # cv.drawContours 会修改原图，所以要拷贝留作后用
# cnt = contours[0]
# cv.drawContours(im0, [cnt], 0, (0,0,255), 3)
#
# # 直角矩形(绿色显示)
# x,y,w,h = cv.boundingRect(cnt)
# cv.rectangle(im0,(x,y),(x+w,y+h),(0,255,0),2)
#
# # 旋转矩形(蓝色显示)
# rect = cv.minAreaRect(cnt) # 得到中心，宽度，高度，选装角度
# box = cv.boxPoints(rect)   # 转换为矩形的4个角
# box = np.int0(box)         # 格式化每个数据为int64
# cv.drawContours(im0,[box],0,(255,0,0),2)
# cv.imshow('im0', im0)
#
# # 最小闭合圈 ###########################################################
# im1 = im.copy()
# (x,y),radius = cv.minEnclosingCircle(cnt)
# center = (int(x),int(y))
# radius = int(radius)
# cv.circle(im1,center,radius,(0,255,0),2)
# cv.imshow('im1', im1)
#
# # 拟合一个椭圆 #########################################################
# im2 = im.copy()
# ellipse = cv.fitEllipse(cnt)
# cv.ellipse(im2,ellipse,(0,255,0),2)
# cv.imshow('im2', im2)
#
# # 拟合一条直线 #########################################################
# im3 = im.copy()
# rows,cols = im3.shape[:2]
# [vx,vy,x,y] = cv.fitLine(cnt, cv.DIST_L2,0,0.01,0.01)
# lefty = int((-x*vy/vx) + y)
# righty = int(((cols-x)*vy/vx)+y)
# cv.line(im3,(cols-1,righty),(0,lefty),(0,255,0),2)
# cv.imshow('im3', im3)
#
# cv.waitKey(0)
# cv.destroyAllWindows()

import numpy as np
import cv2 as cv
import glob

# 终止条件
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# 准备对象点， 如 (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
print('1-------------------------------------')
print(objp)
print('2-------------------------------------')
print(objp[:,:2])
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
print('3-------------------------------------')
print(np.mgrid[0:7,0:6])
print('4-------------------------------------')
print(np.mgrid[0:7,0:6].T)
print('5-------------------------------------')
print(objp[:,:2])

# 用于存储所有图像的对象点和图像点的数组。
objpoints = [] # 真实世界中的3d点
imgpoints = [] # 图像中的2d点

images = glob.glob('./chesspic/*.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 找到棋盘角落
    ret, corners = cv.findChessboardCorners(gray, (7,6), None)
    # 如果找到，添加对象点，图像点（细化之后）
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # 绘制并显示拐角
        cv.drawChessboardCorners(img, (7,6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)
cv.destroyAllWindows()

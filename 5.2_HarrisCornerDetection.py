# 哈里斯角检测

import numpy as np
import cv2 as cv

# img = cv.imread('./data/chessboard.png')
img = cv.imread('./data/subpixel5.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray = np.float32(gray)
print(gray.shape)

# 返回二维的标记角点
dst = cv.cornerHarris(gray,2,3,0.04)
print(dst.shape)

# 形态学扩张
dst = cv.dilate(dst,None)
# 将角点阈值大于最大值0.01倍的点，用红色标记出来
img[dst>0.01*dst.max()]=[0,0,255]

cv.imshow('dst',img)
cv.waitKey(0)
cv.destroyAllWindows()

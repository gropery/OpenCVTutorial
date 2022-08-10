import numpy as np
import cv2 as cv

# 加载原图
im = cv.imread('./data/thunder.png')

# 阈值判定
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY) # 注意这里原图为白底黑图，要反转一下
ret,thresh = cv.threshold(imgray,127,255,cv.THRESH_BINARY_INV)
cv.imshow('imgray', imgray)

# 寻找轮廓
contours,hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

# 绘制第0条轮廓(红色显示)
im0 = im.copy()    # cv.drawContours 会修改原图，所以要拷贝留作后用
cnt = contours[0]
cv.drawContours(im0, [cnt], 0, (0,0,255), 3)

# 直角矩形(绿色显示)
x,y,w,h = cv.boundingRect(cnt)
cv.rectangle(im0,(x,y),(x+w,y+h),(0,255,0),2)
cv.imshow('im0', im0)

# 面积
area = cv.contourArea(cnt)
print('area=',area)

# 长宽比(它是对象边界矩形的宽度与高度的比值)
aspect_ratio = float(w)/h
print('aspect_ratio=', aspect_ratio)

# 范围(范围是轮廓区域与边界矩形区域的比值)
rect_area = w*h
extent = float(area)/rect_area
print('extent=', extent)

# 坚实度(坚实度是等高线面积与其凸包面积之比)
hull = cv.convexHull(cnt)               # 凸包
hull_area = cv.contourArea(hull)        # 凸包面积
solidity = float(area)/hull_area
print('solidity=', solidity)

# 等效直径(等效直径是面积与轮廓面积相同的圆的直径)
equi_diameter = np.sqrt(4*area/np.pi)
print('equi_diameter=', equi_diameter)

# 取向(取向是物体指向的角度。以下方法还给出了主轴和副轴的长度)
(x,y),(MA,ma),angle = cv.fitEllipse(cnt)
print('x=',x,'y=',y,'MA=',MA,'ma',ma)

# 掩码和像素点（在某些情况下，我们可能需要构成该对象的所有点）
# Numpy给出的坐标是(行、列)格式，而OpenCV给出的坐标是(x,y)格式。
# 所以基本上答案是可以互换的。注意，row = x, column = y
mask = np.zeros(imgray.shape,np.uint8)
cv.drawContours(mask,[cnt],0,255,-1)
cv.imshow('mask',mask)
print('pixelpoints1--------------------------------')
pixelpoints1 = np.transpose(np.nonzero(mask))     #方法1
print(pixelpoints1)
print('pixelpoints2--------------------------------')
pixelpoints2 = cv.findNonZero(mask)               #方法2
print(pixelpoints2)

# 最大值，最小值和它们的位置，可以使用掩码图像找到这些参数
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(imgray,mask = mask)
print('min_val=',min_val,'max_val=',max_val,'min_loc=',min_loc,'max_loc=',max_loc)

# BGR平均颜色或Gray平均强度
mean_bgr_val = cv.mean(im,mask = mask)
mean_gray_val = cv.mean(imgray,mask = mask)
print('mean_bgr_val=',mean_bgr_val,'mean_gray_val=',mean_gray_val)

# 极端点 指对象的最顶部，最底部，最右侧和最左侧的点
leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])

cv.circle(im,leftmost,10, (0,0,255), -1)
cv.circle(im,rightmost,10, (0,255,0), -1)
cv.circle(im,topmost,10, (255,0,0), -1)
cv.circle(im,bottommost,10, (0,255,255), -1)
cv.imshow('imMost', im)

cv.waitKey(0)
cv.destroyAllWindows()

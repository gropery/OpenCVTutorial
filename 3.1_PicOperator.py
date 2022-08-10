import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('messi511.jpg')

# 显示原图
cv.imshow('image', img)

# 仅访问坐标（100，100）处,BRG
px = img[100,100]
print( 'BRG=' ,px )

# 仅访问坐标（100，100）处，单独像素的值
blue = img[100,100,0]
print( 'blue=', blue )
green = img[100,100,1]
print( 'green=', green )
red = img[100,100,2]
print( 'red=', red )

# 用相同的方式修改像素值
img[100,100] = [0,0,255]
print( 'img[100,100]=', img[100,100])

# 使用numpy来修改像素值是更好的方式
red = img.item(10,10,2)
print( 'img.item.red=', red)
img.itemset((10,10,2),255)
red = img.item(10,10,2)
print( 'img.item.red=', red)

# 访问图像属性，返回 行，列，和通道数的元组(如果图像是彩色的)
# 如果图像是灰度的，则返回元组仅包含行数和列数
print( 'img.shape=', img.shape )

# 访问像素总数
print( 'img.size=', img.size)

# 访问图像数据类型
print( 'img.dtype', img.dtype)

ball = img[280:340, 330:390]
img[173:233, 100:160] = ball

# 显示修改过像素后的图
cv.imshow('image2', img)

# 将所有红色像素设置为0，无需拆分通道，numpy索引更快
img [:, :, 2] = 0
cv.imshow('image3', img)

# 为图像设置边框（填充）
BLUE = [255,0,0]
img1 = cv.imread('opencv-logo.png')
replicate = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_REPLICATE)
reflect = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_REFLECT)
reflect101 = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_REFLECT_101)
wrap = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_WRAP)
constant= cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_CONSTANT,value=BLUE)
plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')
plt.show()

while True:
    k = cv.waitKey(0) & 0xFF
    if k == 27:         # 等待ESC退出
        cv.destroyAllWindows()
        break

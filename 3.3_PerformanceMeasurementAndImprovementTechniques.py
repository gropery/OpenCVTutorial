import cv2 as cv
import time

# 测量程序运行时间 ##########################
img1 = cv.imread('./data/messi5.jpg')
t1 = time.time()
e1 = cv.getTickCount()

for i in range(5,49,2):
    img1 = cv.medianBlur(img1,i)

t2 = time.time()
e2 = cv.getTickCount()
t = (t2 - t1)
e = (e2 - e1)/cv.getTickFrequency()

print( t )
print( e )

# 检查是否启用了优化 ##########################
print(cv.useOptimized())
t1 = time.time()
res1 = cv.medianBlur(img1,49)
t2 = time.time()
print(t2-t1)

cv.setUseOptimized(False)
print(cv.useOptimized())
t1 = time.time()
res1 = cv.medianBlur(img1,49)
t2 = time.time()
print(t2-t1)

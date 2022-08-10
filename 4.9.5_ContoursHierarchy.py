import cv2 as cv
import numpy as np

img = cv.imread('./data/hierarchy1.png')
imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)

# 只检索所有的轮廓，但不创建任何亲子关系。在这个规则下，父轮廓和子轮廓是平等的，他们只是轮廓。他们都属于同一层级
print('RETR_LIST==1----------------------------------------------------------------')
contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(hierarchy)

imRETR_LIST = img.copy()
cnt = contours[0]
cv.drawContours(imRETR_LIST, [cnt], 0, (255, 0, 0), 3)
cnt = contours[1]
cv.drawContours(imRETR_LIST, [cnt], 0, (0, 255, 0), 3)
cnt = contours[2]
cv.drawContours(imRETR_LIST, [cnt], 0, (0, 0, 255), 3)
cv.imshow('imRETR_LIST', imRETR_LIST)

# 如果使用此标志，它只返回极端外部标志
print('RETR_EXTERNAL==0----------------------------------------------------------------')
contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
print(hierarchy)

imRETR_EXTERNAL = img.copy()
cnt = contours[0]
cv.drawContours(imRETR_EXTERNAL, [cnt], 0, (255, 0, 0), 3)
cnt = contours[1]
cv.drawContours(imRETR_EXTERNAL, [cnt], 0, (0, 255, 0), 3)
cnt = contours[2]
cv.drawContours(imRETR_EXTERNAL, [cnt], 0, (0, 0, 255), 3)
cv.imshow('imRETR_EXTERNAL', imRETR_EXTERNAL)


img = cv.imread('./data/hierarchy2.png')
imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)

# 此标志检索所有轮廓并将其排列为2级层次结构。物体的外部轮廓(即物体的边界)放在层次结构1中。对象内部孔洞的轮廓(如果有)放在层次结构2中。
# 只需考虑在黑色背景上的“白色的零”图像。零的外层属于第一级，零的内层属于第二级。
print('RETR_CCOMP==2----------------------------------------------------------------')
contours, hierarchy = cv.findContours(thresh, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
print(hierarchy)

imRETR_CCOMP = img.copy()
cnt = contours[0]
cv.drawContours(imRETR_CCOMP, [cnt], 0, (255, 0, 0), 3)
cnt = contours[1]
cv.drawContours(imRETR_CCOMP, [cnt], 0, (0, 255, 0), 3)
cnt = contours[2]
cv.drawContours(imRETR_CCOMP, [cnt], 0, (0, 0, 255), 3)
cv.imshow('imRETR_CCOMP', imRETR_CCOMP)

# 它检索所有的轮廓并创建一个完整的家族层次结构列表。它甚至告诉，谁是爷爷，父亲，儿子，孙子，甚至更多
print('RETR_TREE==3----------------------------------------------------------------')
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
print(hierarchy)

imRETR_TREE = img.copy()
cnt = contours[0]
cv.drawContours(imRETR_TREE, [cnt], 0, (255, 0, 0), 3)
cnt = contours[1]
cv.drawContours(imRETR_TREE, [cnt], 0, (0, 255, 0), 3)
cnt = contours[2]
cv.drawContours(imRETR_TREE, [cnt], 0, (0, 0, 255), 3)
cv.imshow('imRETR_TREE', imRETR_TREE)


cv.waitKey(0)
cv.destroyAllWindows()

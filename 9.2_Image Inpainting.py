import numpy as np
import cv2 as cv

img = cv.imread('./data/messi_2.jpg')
mask = cv.imread('./data/mask2.jpg',0)
row,col,ch= img.shape
mask = cv.resize(mask,(col,row))

imgtelea = cv.inpaint(img,mask,3,cv.INPAINT_TELEA)
imgns = cv.inpaint(img,mask,3,cv.INPAINT_NS)

cv.imshow('img',img)
cv.imshow('mask',mask)
cv.imshow('imgtelea',imgtelea)
cv.imshow('imgns',imgns)

cv.waitKey(0)
cv.destroyAllWindows()

import cv2 as cv
import numpy as np

# 凸性缺陷 ############################################################
img = cv.imread('./data/star.jpg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(img_gray, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, 2, 1)
cnt = contours[0]  # cnt 为3维的数列
hull = cv.convexHull(cnt, returnPoints=False)  # hull 为2维的数列
defects = cv.convexityDefects(cnt, hull)
for i in range(defects.shape[0]):
    s, e, f, d = defects[i, 0]  # 返回hull中特定的值
    start = tuple(cnt[s][0])  # start 为1维的数列，保存了[x,y]，需要类型中转换为tuple(x,y)
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    cv.line(img, start, end, [0, 255, 0], thickness=2)
    cv.circle(img, far, 5, [0, 0, 255], -1)
cv.imshow('img', img)

# 点至轮廓线的最短距离 ############################################################
dist = cv.pointPolygonTest(cnt, (50,50), True)
print(dist)
within = cv.pointPolygonTest(cnt, (50,50), False)
print(within)

#  形状匹配 ############################################################
img1 = cv.imread('./data/star.jpg',0)
img2 = cv.imread('./data/star2.jpg',0)
img3 = cv.imread('./data/rectangle.jpg',0)
ret, thresh = cv.threshold(img1, 127, 255,0)
ret, thresh2 = cv.threshold(img2, 127, 255,0)
ret, thresh3 = cv.threshold(img3, 127, 255,0)
contours,hierarchy = cv.findContours(thresh,2,1)
cnt1 = contours[0]
contours,hierarchy = cv.findContours(thresh2,2,1)
cnt2 = contours[0]
contours,hierarchy = cv.findContours(thresh3,2,1)
cnt3 = contours[0]
ret = cv.matchShapes(cnt1,cnt1,1,0.0)
print( 'star-star',ret )
ret = cv.matchShapes(cnt1,cnt2,1,0.0)
print( 'star-star2',ret )
ret = cv.matchShapes(cnt1,cnt3,1,0.0)
print( 'star-rectangle',ret )

cv.waitKey(0)
cv.destroyAllWindows()

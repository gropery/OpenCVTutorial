import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('./data/messi5.jpg',0)
img2 = img.copy()
template = cv.imread('./data/messi_face.jpg',0)
w, h = template.shape[::-1]

# 列表中所有的6种比较方法
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

for meth in methods:
    img = img2.copy()
    method = eval(meth)
    # 应用模板匹配
    res = cv.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    # 如果方法是TM_SQDIFF或TM_SQDIFF_NORMED，则取最小值
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(img,top_left, bottom_right, 255, 2)
    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()


# img = cv.imread('./data/mario.png')
# img_rgb = img.copy()
# img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
# template = cv.imread('./data/mario_coin.png', 0)
# w, h = template.shape[::-1]
#
# res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
#
# threshold = 0.8
# loc = np.where(res >= threshold)
#
# # 方法1
# # for pt in zip(*loc[::-1]):
# #     cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
#
# # 方法2
# loc_ = loc[::-1]
# arr = np.transpose(loc_)
# for pt in arr:
#     cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
#
# cv.imshow('imOrigin', img)
# cv.imshow('res.png', img_rgb)
#
# cv.waitKey(0)
# cv.destroyAllWindows()

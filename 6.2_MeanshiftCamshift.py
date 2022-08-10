import numpy as np
import cv2 as cv
import argparse
parser = argparse.ArgumentParser(description='This sample demonstrates the meanshift algorithm. \
                                              The example file can be downloaded from: \
                                              https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
parser.add_argument('--image', type=str, help='path to image file',default='./data/slow_traffic_small.mp4')
args = parser.parse_args()
cap = cv.VideoCapture(args.image)

# 视频的第一帧
ret,frame = cap.read()
# 设置窗口的初始位置
x, y, w, h = 300, 200, 100, 50 # simply hardcoded the values
track_window = (x, y, w, h)
# 设置初始ROI来追踪
roi = frame[y:y+h, x:x+w]
hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
# 避免由于光线不足而产生错误的值，可以使用**cv.inRange**()函数丢弃光线不足的值
mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
# 对于直方图，此处仅考虑色相，然后再归一化
roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)
# 设置终止条件，可以是10次迭代，或者至少移动1 pt
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )
while(1):
    ret, frame = cap.read()
    if ret == True:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # meanShift 方法 ################################################
        # 应用meanshift来获取新位置
        # ret, track_window = cv.meanShift(dst, track_window, term_crit)
        # print(ret)
        # # 在图像上绘制
        # x,y,w,h = track_window
        # img2 = cv.rectangle(frame, (x,y), (x+w,y+h), 255,2)

        # camshift 方法 ################################################
        # 应用camshift 到新位置 center, size, and orientation.
        ret, track_window = cv.CamShift(dst, track_window, term_crit)
        print(ret)
        # 从旋转矩形中找到4个角坐标
        pts = cv.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv.polylines(frame, [pts], True, 255, 2)

        cv.imshow('img2',img2)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
    else:
        break

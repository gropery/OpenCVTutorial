import numpy as np
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
                                              The example file can be downloaded from: \
                                              https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
parser.add_argument('--image', type=str, help='path to image file',default='./data/slow_traffic_small.mp4')
args = parser.parse_args()
cap = cv.VideoCapture(args.image)


# 用于ShiTomasi拐点检测的参数
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# lucas kanade光流参数
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# 创建一些随机的颜色
color = np.random.randint(0,255,(100,3))
# 拍摄第一帧并在其中找到拐角
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# 创建用于作图的掩码图像
mask = np.zeros_like(old_frame)
while(1):
    ret,frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # 计算光流,传递前一帧，下一帧,前一点,后一点(None)。它返回下一个点以及状态码(如果找到下一个点，状态码的值为1，否则为零)
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # 选择良好点(st 是Array类型)
    good_new = p1[st==1]
    good_old = p0[st==1]
    print('p1:',p1.shape,'good_new:',good_new.shape)
    print('p0',p0.shape,'good_old:',good_old.shape)

    # 绘制跟踪
    for i,(new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv.line(mask, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 2)
        frame = cv.circle(frame,(int(a),int(b)),5,color[i].tolist(),-1)
    img = cv.add(frame,mask)
    cv.imshow('frame',img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    # 现在更新之前的帧和点
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
    print('p00', p0.shape)


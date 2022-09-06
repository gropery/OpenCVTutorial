"""
 功能：手势操作电脑音量
 1、使用OpenCV读取摄像头视频流；
 2、识别手掌关键点像素坐标；
 3、根据拇指和食指指尖的坐标，利用勾股定理计算距离；
 4、将距离等比例转为音量大小，控制电脑音量
 """

import math

import cv2
import mediapipe as mp
import time
import numpy as np


# 导入电脑音量控制模块
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# 获取电脑音量范围
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volume_range = volume.GetVolumeRange()
rect_height = 0
rect_percent_text = 0

# mediapipe 画线工具
mpDraw = mp.solutions.drawing_utils

# 画线参数：1、颜色，2、线条粗细，3、点的半径
DrawingSpec_point = mpDraw.DrawingSpec((0, 255, 0), 1, 3)
DrawingSpec_line = mpDraw.DrawingSpec((0, 0, 255), 1, 1)

# mediapipe 人手识别方法
mpHands = mp.solutions.hands

# 1、是否检测静态图片，2、手的数量，3、检测阈值，4、跟踪阈值
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)

pTime = 0  # 开始时间初始化
cTime = 0  # 目前时间初始化

# 获取视频对象，0为摄像头，也可以写入视频路径
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()



while True:
    # success是布尔型，读取帧正确返回True;img是每一帧的图像（BGR存储格式）
    success, img = cap.read()
    if not success:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # 将一幅图像从一个色彩空间转换为另一个,返回转换后的色彩空间图像
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = img.shape

    # 处理RGB图像并返回手的标志点和检测到的每个手对象
    results = hands.process(imgRGB)

    # # 判断是否有手掌，会得到手的列表记录了手的坐标
    if results.multi_hand_landmarks:
        # 遍历每个手掌
        for handLms in results.multi_hand_landmarks:
            # h-height,w-weight图像的宽度和高度


            # 找到食指和拇指关节点,计算2者距离,并显示于图片中
            shizhi_postion = (int(handLms.landmark[8].x * w), int(handLms.landmark[8].y * h))
            muzhi_postion = (int(handLms.landmark[4].x * w), int(handLms.landmark[4].y * h))
            distance = int(shizhi_postion[0] - muzhi_postion[0]) ** 2 + int(shizhi_postion[1] - muzhi_postion[1]) ** 2
            distance = int(math.sqrt(distance))
            distancexy = (int(((muzhi_postion[0] + shizhi_postion[0]) / 2)), int(((muzhi_postion[1] + shizhi_postion[1]) / 2)))

            cv2.line(img, muzhi_postion, shizhi_postion, (255, 0, 0), 5)
            cv2.putText(img, str(distance), distancexy, cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)

            # 获取电脑最大最小音量
            min_volume = volume_range[0]
            max_volume = volume_range[1]
            # 将指尖长度映射到音量上
            vol = np.interp(distance, [50, 300], [min_volume, max_volume])
            # 将指尖长度映射到矩形显示上
            rect_height = np.interp(distance, [50, 300], [0, 200])
            rect_percent_text = np.interp(distance, [50, 300], [0, 100])
            # 设置电脑音量
            volume.SetMasterVolumeLevel(vol, None)

            # landmark有21个（具体查阅上面的参考网址），id是索引，lm是x,y坐标
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)  # lm的坐标是点在图像中的比例坐标
                # 将landmark的比例坐标转换为在图像像素上的坐标
                cx, cy = int(lm.x * w), int(lm.y * h)
                # 将手的标志点个性化显示
                cv2.circle(img, (cx, cy), int(w / 50), (255, 0, 255), cv2.FILLED)

            # # 打印左右手
            # print('Handedness:', results.multi_handedness)
            # # 打印固定手指的位置
            # print(handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x * w)
            # 在图像上绘制手的标志点和他们的连接线
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, DrawingSpec_point, DrawingSpec_line)

    # 显示音量矩形
    cv2.putText(img, str(math.ceil(rect_percent_text)) + "%", (10, 350), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    img = cv2.rectangle(img, (30, 100), (70, 300), (255, 0, 0), 3)
    img = cv2.rectangle(img, (30, math.ceil(300 - rect_height)), (70, 300), (255, 0, 0), -1)

    # 计算并显示帧率
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    # 将帧率显示在图像上
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 1)

    # 在Image窗口上显示新绘制的图像img
    cv2.imshow("Image", img)

    # 按下q退出程序
    if cv2.waitKey(1) == ord('q'):
        break

# 完成所有操作后，释放捕获器
hands.close()
cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import time

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

# 全局变量
frameNum = 0
commandLst = []
commandSending = ''
commandDict = {'right': '10', 'left': '01', 'forward': '11', 'backforward': '00', 'unKnown': '-1'}

while True:
    # 初始化0关键点的坐标
    lst = [0, 0, 0]
    # 初始化字典
    distanceDict = {}

    # success是布尔型，读取帧正确返回True;img是每一帧的图像（BGR存储格式）
    success, img = cap.read()
    if not success:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # 将一幅图像从一个色彩空间转换为另一个,返回转换后的色彩空间图像
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 处理RGB图像并返回手的标志点和检测到的每个手对象
    results = hands.process(imgRGB)

    # results.multi_hand_landmarks返回None或手的标志点坐标
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # landmark有21个（具体查阅上面的参考网址），id是索引，lm是x,y坐标
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)  # lm的坐标是点在图像中的比例坐标
                # h-height,w-weight图像的宽度和高度
                h, w, c = img.shape
                # 将landmark的比例坐标转换为在图像像素上的坐标
                cx, cy = int(lm.x * w), int(lm.y * h)
                # 将手的标志点个性化显示
                cv2.circle(img, (cx, cy), int(w / 50), (255, 0, 255), cv2.FILLED)

                # 存储0关键点的三个坐标
                if id == 0:
                    lst = [lm.x, lm.y, lm.z]

                # 分别检测4，8，12，20四个关键结点与0结点间的距离判断手指指向
                if id == 4 or id == 8 or id == 12 or id == 20:
                    distanceDict[id] = (lst[0] - lm.x) ** 2 + (lst[1] - lm.y) ** 2 + (lst[2] - lm.z) ** 2

            # # 打印左右手
            # print('Handedness:', results.multi_handedness)
            # # 打印固定手指的位置
            # print(handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x * w)
            # 在图像上绘制手的标志点和他们的连接线
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, DrawingSpec_point, DrawingSpec_line)

    command = 'UnKnown'
    if distanceDict == {}:
        MaxId = 0
    else:
        # 取4根手指中最长距离的手指对应的ID
        MaxId = [key for key, value in distanceDict.items() if value == max(distanceDict.values())][0]
    # 判断哪个结点距离根节点最远，并由此给出相应的命令
    if MaxId == 4:
        command = 'right'
    if MaxId == 8:
        command = 'left'
    if MaxId == 12:
        command = 'forward'
    if MaxId == 20:
        command = 'backforward'
    if MaxId == 0:
        command = 'unKnown'
    cv2.putText(img, command, (10, 150), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # 计算并显示帧率
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    # 将帧率显示在图像上
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 1)

    # 在Image窗口上显示新绘制的图像img
    cv2.imshow("Image", img)

    # 连续判断5帧图像,如果都指示同一手指ID,则判定准确并输出
    frameNum += 1
    commandLst.append(command)
    if frameNum == 5:
        if max(commandLst) == min(commandLst):
            commandSending = commandDict[command]
        else:
            commandSending = commandDict['unKnown']

        # 数据重新置零
        frameNum = 0
        commandLst = []

        print(commandSending)
    else:
        continue

    # 按下q退出程序
    if cv2.waitKey(1) == ord('q'):
        break

# 完成所有操作后，释放捕获器
hands.close()
cap.release()
cv2.destroyAllWindows()

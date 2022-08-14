import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# img1:待寻找图片
# img2:主图
def findplace(img1, img2):
    MIN_MATCH_COUNT = 10
    # img1 = cv.imread('./data/box2.png', 0)  # 索引图像
    # img2 = cv.imread('./data/box_in_scene.png', 0)  # 训练图像
    # img1 = cv.imread('./data/test2.png', 0)  # 索引图像
    # img2 = cv.imread('./data/test1.png', 0)  # 训练图像
    # print(img1.shape)
    # print(img2.shape)

    # 初始化SIFT检测器
    sift = cv.xfeatures2d.SIFT_create()
    # 用SIFT找到关键点和描述符
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # FLANN的参数(SIFT,SURF使用以下参数)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    # 初始化flann匹配器
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # 根据Lowe的论文进行比例测试,并存储所有符合条件的匹配项。
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    # 如果找到足够的匹配项>10，我们将在两个图像中提取匹配的关键点的位置。他们被传递以寻找预期的转变。
    # 一旦获得了这个3x3转换矩阵，就可以使用它将索引图像的角转换为训练图像中的相应点。然后我们画出来。
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # print('src_pts--------------------------\r\n', src_pts)
        # print('dst_pts--------------------------\r\n', dst_pts)
        # findHomography 向该函数传递点集，函数将在两个图像中提取匹配的关键点的位置，以在复杂图像中找到该已知对象的透视变换关系
        # 参数 RANSAC/LEAST_MEDIAN 用以解决匹配时可能会出现一些可能影响结果的错误
        # 函数返回指定内部和外部点的掩码mask(提供正确估计的良好匹配称为“内部点”，其余的称为“外部点”)
        # 函数返回3x3的M转换矩阵，使用它将索引图像的4个边角，经过M矩阵透视转换为训练图像中的相应点。然后我们画出来。
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        # print('M--------------------------\r\n', M)
        # print('mask--------------------------\r\n', mask)
        matchesMask = mask.ravel().tolist()
        # print('matchesMask--------------------------\r\n', matchesMask)
        # 手动输入索引图像的4个角坐标
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # print('pts--------------------------\r\n', pts)
        # perspectiveTransform 透视转换，转换至少需要四个正确的点。
        # 输入2通道或3通道浮点array(pts)，M为3x3浮点转换matrix
        # 输出透视转换后训练图中的4个坐标
        dst = cv.perspectiveTransform(pts, M)
        # print('dst--------------------------\r\n', dst)
        # 将img2中透视后的4个点连起来，框选找到目标
        img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None

    # 如果成功找到对象-绘制内部线，如果失败-绘制匹配关键点
    # 对象在混乱的图像中标记为白色
    draw_params = dict(matchColor=(0, 255, 0),  # 用绿色绘制匹配
                       singlePointColor=None,
                       matchesMask=matchesMask,  # 只绘制内部点
                       flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

    img3 = cv.resize(img3, (1024, 768))
    cv.imshow('img3', img3)


mainpic = cv.imread('./data/test1.png', 0)  # 主图
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # 逐帧捕获
    ret, frame = cap.read()
    # 如果正确读取帧，ret为True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # 我们在框架上的操作到这里
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # 显示结果帧e
    # cv.imshow('frame', gray)
    img2 = mainpic.copy()
    findplace(gray, img2)
    if cv.waitKey(100) == ord('q'):
        break

# 完成所有操作后，释放捕获器
cap.release()
cv.destroyAllWindows()

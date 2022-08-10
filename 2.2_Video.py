# 从摄像机中读取视频显示 ##############################################
# import cv2 as cv
# cap = cv.VideoCapture(0)
# if not cap.isOpened():
#     print("Cannot open camera")
#     exit()
# while True:
#     # 逐帧捕获
#     ret, frame = cap.read()
#     # 如果正确读取帧，ret为True
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
#     # 转换为gray图像
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#
#     # 显示结果帧e
#     cv.imshow('frame', gray)
#     if cv.waitKey(1) == ord('q'):
#         break
# # 完成所有操作后，释放捕获器
# cap.release()
# cv.destroyAllWindows()

# 从文件中播放视频 ##############################################
import cv2 as cv
cap = cv.VideoCapture('./data/vtest.avi')
while cap.isOpened():
    ret, frame = cap.read()
    # 如果正确读取帧，ret为True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()

# 从摄像机中读取视频，并写入视频 ###################################
# import cv2 as cv
# cap = cv.VideoCapture(0)
# # 定义编解码器并创建VideoWriter对象
# fourcc = cv.VideoWriter_fourcc(*'XVID')
# out = cv.VideoWriter('output.avi', fourcc, 20.0, (640,  480))
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
#     frame = cv.flip(frame, 0)
#     # 写翻转的框架
#     out.write(frame)
#     cv.imshow('frame', frame)
#     if cv.waitKey(1) == ord('q'):
#         break
# # 完成工作后释放所有内容
# cap.release()
# out.release()
# cv.destroyAllWindows()

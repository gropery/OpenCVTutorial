import cv2 as cv

# 打印鼠标响应事件 ###################################
events = [i for i in dir(cv) if 'EVENT' in i]
print(events)
# ['EVENT_FLAG_ALTKEY',
# 'EVENT_FLAG_CTRLKEY',
# 'EVENT_FLAG_LBUTTON',  左键单击
# 'EVENT_FLAG_MBUTTON',  中健单击
# 'EVENT_FLAG_RBUTTON',  右键单击
# 'EVENT_FLAG_SHIFTKEY',
# 'EVENT_LBUTTONDBLCLK', 左键双击
# 'EVENT_LBUTTONDOWN',
# 'EVENT_LBUTTONUP',
# 'EVENT_MBUTTONDBLCLK', 中健双击
# 'EVENT_MBUTTONDOWN',
# 'EVENT_MBUTTONUP',
# 'EVENT_MOUSEHWHEEL',   中间滚动
# 'EVENT_MOUSEMOVE',     鼠标移动
# 'EVENT_MOUSEWHEEL',
# 'EVENT_RBUTTONDBLCLK', 右键双击
# 'EVENT_RBUTTONDOWN',
# 'EVENT_RBUTTONUP']

# # 鼠标双击事件回调函数 #################################
# import numpy as np
# import cv2 as cv
#
#
# # 鼠标回调函数
# def draw_circle(event, x, y, flags, param):
#     if event == cv.EVENT_LBUTTONDBLCLK:
#         cv.circle(img, (x, y), 100, (255, 0, 0), -1)
#
#
# # 创建一个黑色的图像，一个窗口，并绑定到窗口的功能
# img = np.zeros((512, 512, 3), np.uint8)
# cv.namedWindow('image')
# cv.setMouseCallback('image', draw_circle)
# while True:
#     cv.imshow('image', img)
#     if cv.waitKey(20) & 0xFF == ord('q'):
#         break
# cv.destroyAllWindows()

# # 鼠标事件回调函数，绘制矩形或圆形 #######################
import numpy as np
import cv2 as cv

drawing = False  # 如果按下鼠标，则为真
mode = True  # 如果为真，绘制矩形。按 m 键可以切换到曲线
ix, iy = -1, -1


# 鼠标回调函数
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
            else:
                cv.circle(img, (x, y), 5, (0, 0, 255), -1)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
        else:
            cv.circle(img, (x, y), 5, (0, 0, 255), -1)


# 创建一个黑色的图像，一个窗口，并绑定到窗口的功能
img = np.zeros((512, 512, 3), np.uint8)
cv.namedWindow('image')
cv.setMouseCallback('image', draw_circle)
while True:
    cv.imshow('image', img)
    key = cv.waitKey(20)
    if key == ord('q'):
        break
    elif key == ord('m'):
        mode = not mode

cv.destroyAllWindows()

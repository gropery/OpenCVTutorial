import numpy as np
import cv2 as cv


# img = cv.imread('messi5.jpg', cv.IMREAD_COLOR)
# img = cv.imread('messi5.jpg', cv.IMREAD_GRAYSCALE)
img = cv.imread('messi5.jpg', cv.IMREAD_UNCHANGED)

cv.namedWindow('image', cv.WINDOW_NORMAL)
cv.imshow('image', img)


while True:
    k = cv.waitKey(0) & 0xFF
    if k == 27:         # 等待ESC退出
        cv.destroyAllWindows()
        break
    elif k == ord('s'): # 等待关键字，保存和退出
        cv.imwrite('messigray.png', img)
        cv.destroyAllWindows()



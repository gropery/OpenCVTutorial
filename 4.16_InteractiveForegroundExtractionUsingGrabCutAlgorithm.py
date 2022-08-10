import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('./data/messi5.jpg')
mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
rect = (50, 50, 450, 290)
cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

# mask 中位cv.GC_BGD=0,cv.GC_FGD=1, cv.GC_PR_BGD=2,cv.GC_PR_FGD=3
# 那么将mask中背景统一为0，前景值统一为1
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
print(mask2.shape)
print(mask2.ndim)
print(mask2)
plt.imshow(mask2)

# 改变 mask 2维至3维，与原图相乘后背景的部分即置为0，前景的部分保持原像素不变
# mask3 = mask2[:, :, np.newaxis]
mask3 = np.expand_dims(mask2, axis = 2)
print(mask3.shape)
print(mask3.ndim)
print(mask3)

print(img.shape)
print(img.ndim)
print(img)
img2 = img * mask3

plt.subplot(221), plt.imshow(mask2, 'gray')
plt.subplot(222), plt.imshow(mask3, 'gray')
plt.subplot(223), plt.imshow(img, 'gray')
plt.subplot(224), plt.imshow(img2, 'gray')

plt.show()


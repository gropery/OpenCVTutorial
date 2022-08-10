import cv2 as cv
import numpy as np

# 将曝光图像加载到列表中
img_fn = ["img0.jpg", "img1.jpg", "img2.jpg", "img3.jpg"]
img_list = [cv.imread(fn) for fn in img_fn]
exposure_times = np.array([15.0, 2.5, 0.25, 0.0333], dtype=np.float32)
print(np.array(img_list).shape)
print(np.array(exposure_times).shape)

# # 将曝光合成HDR图像
merge_debevec = cv.createMergeDebevec()      # 方法1
hdr_debevec = merge_debevec.process(img_list, times=exposure_times.copy())
merge_robertson = cv.createMergeRobertson()  # 方法2
hdr_robertson = merge_robertson.process(img_list, times=exposure_times.copy())

# 色调图HDR图像(其他色调图算法：cv::TonemapDrago,cv::TonemapMantiuk和cv::TonemapReinhard)
tonemap1 = cv.createTonemap(gamma=2.2)
res_debevec = tonemap1.process(hdr_debevec.copy())
tonemap2 = cv.createTonemap(gamma=2.2)
res_robertson = tonemap2.process(hdr_robertson.copy())

# 使用Mertens融合曝光
merge_mertens = cv.createMergeMertens()
res_mertens = merge_mertens.process(img_list)

# 转化数据类型为8-bit
res_debevec_8bit = np.clip(res_debevec*255, 0, 255).astype('uint8')
res_robertson_8bit = np.clip(res_robertson*255, 0, 255).astype('uint8')
res_mertens_8bit = np.clip(res_mertens*255, 0, 255).astype('uint8')

cv.imshow('debevec_8bit',res_debevec_8bit)
cv.imshow('robertson_8bit',res_robertson_8bit)
cv.imshow('mertens_8bit',res_mertens_8bit)
cv.waitKey(0)
cv.destroyAllWindows()

# # 估计相机响应函数(CRF)
# cal_debevec = cv.createCalibrateDebevec()
# crf_debevec = cal_debevec.process(img_list, times=exposure_times)
# hdr_debevec = merge_debevec.process(img_list, times=exposure_times.copy(), response=crf_debevec.copy())
# cal_robertson = cv.createCalibrateRobertson()
# crf_robertson = cal_robertson.process(img_list, times=exposure_times)
# hdr_robertson = merge_robertson.process(img_list, times=exposure_times.copy(), response=crf_robertson.copy())

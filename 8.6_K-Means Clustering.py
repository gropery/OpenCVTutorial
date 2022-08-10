import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# 单特征数据 只有一个身高特征作为单个列向量，每一行对应于一个输入测试样本 ######################
# 这里为50x1
# x = np.random.randint(25,100,25)
# print('x:',x)
# y = np.random.randint(175,255,25)
# print('y:',y)
# # 横向水平拼接，即将y水平拼接在x“右边”
# z = np.hstack((x,y))
# print('z',z)
# z = z.reshape((50,1))
# print('z.reshape',z)
# z = np.float32(z)
# # plt.hist(z,256,[0,256])
#
# # 定义终止标准 = ( type, max_iter = 10 , epsilon = 1.0 )
# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# # 设置标志
# flags = cv.KMEANS_RANDOM_CENTERS
# # 应用K均值，分为2个集群，执行10次算法
# compactness,labels,centers = cv.kmeans(z,2,None,criteria,10,flags)
#
# # 得到的中心分别为60和207。标签的大小将与测试数据的大小相同
# # 其中每个数据的质心都将标记为“ 0”，“ 1”，“ 2”等。
# # 现在，根据标签将数据分为不同的群集。
# A = z[labels==0]
# B = z[labels==1]
#
# # 现在绘制用红色'A'，用蓝色绘制'B'，用黄色绘制中心
# plt.hist(A,256,[0,256],color = 'r')
# plt.hist(B,256,[0,256],color = 'b')
# plt.hist(centers,32,[0,256],color = 'y')
#
# plt.show()

# 二维特征数据 每个特征排列在一列中，每一行对应于一个输入测试样本 ######################
# # 这里为50x2
# X = np.random.randint(25,50,(25,2))
# Y = np.random.randint(60,85,(25,2))
# # 纵向竖直拼接，即将y竖直拼接在x“下面”
# Z = np.vstack((X,Y))
# # 将数据转换为 np.float32
# Z = np.float32(Z)
# print(Z)
# # 定义停止标准，应用K均值
# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# ret,label,center=cv.kmeans(Z,2,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
# # 现在分离数据, Note the flatten()
# A = Z[label.ravel()==0]
# B = Z[label.ravel()==1]
# # 绘制数据
# plt.scatter(A[:,0],A[:,1])
# plt.scatter(B[:,0],B[:,1],c = 'r')
# plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
# plt.xlabel('Height'),plt.ylabel('Weight')
# plt.show()

# 三维特征数据 颜色量化 #########################################################
img = cv.imread('./data/home.jpg')
# 3个特征为BRG，将其重塑为Mx3大小的数组（M是图像中的像素数）
Z = img.reshape((-1,3))
print('Z.shape=',Z.shape)            # (196608, 3)
# 将数据转化为np.float32
Z = np.float32(Z)
# 定义终止标准 聚类数并应用k均值
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 8 # 分类为8个集群
ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

# 将数据转化为uint8
center = np.uint8(center)
print('center.shape=',center.shape)       # (8:3)
print('label.shape=',label.shape)         # (196608,1)

# 将原始图像中每个像素点，使用分类后归属的最接近的中心像素代替
# 即图片中每个像素由原来的BGR 24bit，现在简化为8种中心像素，但图片分辨率没变
res = center[label.flatten()]
print('res.shape=',res.shape)             # (196608, 3)

res2 = res.reshape((img.shape))
cv.imshow('res2',res2)
cv.waitKey(0)
cv.destroyAllWindows()

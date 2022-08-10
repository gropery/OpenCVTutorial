import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 包含(x,y)值的25个已知/训练数据的特征集
trainData = np.random.randint(0,100,(25,2)).astype(np.float32)
print(trainData.shape)
print('trainData--------------------------------\r\n',trainData)
# 用数字0和1分别标记红色或蓝色
responses = np.random.randint(0,2,(25,1)).astype(np.float32)
print('responses--------------------------------\r\n',responses)
# 取0红色族并绘图
red = trainData[responses.ravel()==0]
plt.scatter(red[:,0],red[:,1],80,'r','^')
# 取1蓝色族并绘图
blue = trainData[responses.ravel()==1]
plt.scatter(blue[:,0],blue[:,1],80,'b','s')

newcomer = np.random.randint(0,100,(1,2)).astype(np.float32)
plt.scatter(newcomer[:,0],newcomer[:,1],80,'g','o')
knn = cv.ml.KNearest_create()
knn.train(trainData, cv.ml.ROW_SAMPLE, responses)
ret, results, neighbours ,dist = knn.findNearest(newcomer, 3)
print( "result:  {}\n".format(results) )         # 返回0代表红色族，返回1代表蓝色族
print( "neighbours:  {}\n".format(neighbours) )
print( "distance:  {}\n".format(dist) )

# 10个新加入样本
newcomers = np.random.randint(0,100,(10,2)).astype(np.float32)
ret, results,neighbours,dist = knn.findNearest(newcomers, 3)
# 结果包含10个标签
print( "result:  {}\n".format(results) )         # 返回0代表红色族，返回1代表蓝色族
print( "result.size:  {}\n".format(results.size) )

plt.show()

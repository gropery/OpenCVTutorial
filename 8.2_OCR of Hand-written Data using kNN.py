import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

# 读取图片,像素尺寸为h=1000,w=2000,每行有100个数字，每5行变化一种数字
img = cv.imread('./data/digits.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print(gray.shape)

# 现在我们将图像分割为5000个单元格，每个单元格为20x20
# 具体操作为将h切成50等份，即每一等份为一行，每一行有100个相同的字母，因此再将w切成100等份，每一份为一个字母
cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
# 使其成为一个Numpy数组。它的大小将是（50,100,20,20），代表50行，100列，每个单元格是20*20的像素组成的单个字母
x = np.array(cells)
# 打印验证
print('x.shape=',x.shape,'x[0][0].shape=',x[0][0].shape,'x[49,99].shape=',x[49][99].shape)
print(x[0][0])
plt.subplot(131),plt.imshow(x[0][0],'gray')
plt.subplot(132),plt.imshow(x[49][99],'gray')

# 准备train_data和test_data
# 取100列中的前50列数据([0:50,0:50]),将每个字母的20*20的像素展平为400像素
# 最终得到(2500,400), 即2500行，400列的二维数组，其中2500的每250行(原5行x50列)数据为1种数字，其中400为单个数字的展平
# 同时取100列中的后50列数据([0:50,50:100]),作为测试输入
train = x[:, :50].reshape(-1, 400).astype(np.float32)
test = x[:, 50:100].reshape(-1, 400).astype(np.float32)
print('train.shape=',train.shape)
print('test.shape=',test.shape)
# 打印训练样本图样，总共有2500行，其中每一行(400个列)代表一个字母的数据排列，且每250行代表1种字母，因此每250行的数据在列上有相关性
plt.subplot(133),plt.imshow(train,'gray')

# 为训练和测试数据创建标签，2500行每一行都有一个标签，每250行是一种数字标签
k = np.arange(10)
train_labels = np.repeat(k, 250)[:, np.newaxis]
print('train_labels.shape=',train_labels.shape)
print('train_labels=',train_labels)
test_labels = train_labels.copy()

# 初始化kNN，训练数据，然后使用k = 5的测试数据对其进行测试
knn = cv.ml.KNearest_create()
# 每一行作为一个标签的训练样本(400个列像素)，每个数字有250行作为此标签的样本数，那么10个数字，总共就有2500个标签
knn.train(train, cv.ml.ROW_SAMPLE, train_labels)
# 作为测试输入的数据为(2500,400)，即输入了2500个测试数据/数字，每个数据400个像素作为一个数字。
ret, result, neighbours, dist = knn.findNearest(test, k=5)

# 现在，我们检查分类的准确性，返回2500个数字判定的结果
print(result.size)
# 将结果与test_labels进行比较(根据右半边图像的排列可以已知每250个会变化一次)，并检查哪个错误
matches = result == test_labels
correct = np.count_nonzero(matches)
accuracy = correct * 100.0 / result.size
print(accuracy)

plt.show()

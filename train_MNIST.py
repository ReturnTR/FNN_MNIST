import FNN
import cupy as np
import MNIST.mnist as mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape(60000, 28 * 28)  # 适配网络形状
train_images = train_images.astype("float32") / 255  # 适配网络数据大小，变为0-1的浮点数
test_images = test_images.reshape(10000, 28 * 28)
test_images = test_images.astype("float32") / 255  # 适配网络数据大小，变为0-1的浮点数

train_labels_temp = np.zeros((60000, 10))  # 适配输出形状
for i in range(len(train_labels)): train_labels_temp[i][train_labels[i]] = 1.0
train_labels = train_labels_temp

test_labels_temp = np.zeros((10000, 10))  # 适配输出形状
for i in range(len(test_labels)): test_labels_temp[i][test_labels[i]] = 1.0
test_labels = test_labels_temp

network=FNN.FNN(input_dim=28*28,size=0.01)
network.add(output_dim=512,activation='ReLU')
network.add(output_dim=512,activation='ReLU')
network.add(output_dim=10,activation='ReLU')
loss=network.train2(train_x=train_images,train_y=train_labels,loss_function='softmax_cross_entropy',batch_size=100,epochs=30,learning_rate=0.01)


network.test(test_images,test_labels)


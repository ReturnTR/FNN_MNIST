import cupy as np

class Layer:

    def forward(self, x): pass

    def backward(self, back_grad, learning_rate): pass


class Dense(Layer):
    def __init__(self, input_dim, output_dim, size):
        self.w = np.random.randn(input_dim, output_dim) * size  # 初始化问题
        self.b = np.zeros(output_dim)  # 初始化问题

    def forward(self, x):
        self.x = x

        return np.dot(x, self.w) + self.b

    def backward(self, back_grad, learning_rate):
        grad_w = (np.dot(self.x.T, back_grad) / self.x.shape[0])
        grad_b = back_grad.mean(axis=0)  # 取第一维的平均值
        self.w = self.w - learning_rate * grad_w
        self.b = self.b - learning_rate * grad_b
        return np.dot(back_grad, self.w.T)


class ReLU(Layer):
    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def backward(self, back_grad, learning_rate):
        return (self.x > 0) * back_grad


class Softmax(Layer):
    def forward(self, x):
        self.x = x
        x = np.exp(x - x.max())
        self.y = (x.T / np.sum(x, axis=1)).T
        print(self.y[0])
        return self.y

    def backward(self, back_grad, learning_rate):
        temp = (self.y - self.y * self.y) * back_grad
        for i in range(len(temp)):
            for j in range(len(temp)):
                if i != j: temp[i] = temp[i] - back_grad[j] * self.y[i] * self.y[j]
        return temp


class Sigmoid(Layer):
    def __init__(self):
        pass

    def forward(self, x):
        self.x = x
        self.y = 1.0 / (1 + np.exp(-x))
        return self.y

    def backward(self, input, back_grad):
        sigmoid_grad = self.y * (1 - self.y)
        return back_grad * sigmoid_grad


class Tanh(Layer):

    def forward(self, x):
        self.y = np.tanh(x)
        return self.y

    def backward(self, input, back_grad):
        grad_tanh = 1 - (self.y) ** 2
        return back_grad * grad_tanh

def cross_entropy(y, y_hat):
    loss = -np.sum(np.log(y_hat) * y, axis=1)
    loss_grad = -y / y_hat
    print(loss_grad[0])
    return loss, loss_grad

def softmax_cross_entropy(y,x):
    x = np.exp(x - x.max())
    y_hat = (x.T / np.sum(x, axis=1)).T
    loss = -np.sum(np.log(y_hat) * y, axis=1)
    loss_grad=y_hat-y
    return loss,loss_grad

def MSE(y, y_hat):
    loss = np.sum(np.square(y_hat - y),axis=1)
    loss_grad = 2.0 * (y_hat - y)
    return loss, loss_grad


class FNN:
    def __init__(self, input_dim, size=0.01):
        self.size = size
        self.input_dim = input_dim
        self.network = []

    def add(self, output_dim, activation):
        self.network.append(Dense(self.input_dim, output_dim, self.size))
        if activation != '': self.network.append(eval(activation)())
        self.input_dim = output_dim

    def forward(self, x):
        for layer in self.network: x = layer.forward(x)
        return x

    def train(self, train_x, train_y, batch_size, loss_function="MSE", learning_rate=0.1, epochs=10):
        """参数优化方法"""
        alpha=0.85
        for epoch in range(epochs):
            learning_rate *= alpha
            for i in range(int(train_x.shape[0] / batch_size)):
                x = train_x[i * batch_size:(i + 1) * batch_size]
                y = train_y[i * batch_size:(i + 1) * batch_size]
                y_hat = self.forward(x)  # 向前传播
                (loss, loss_grad) = eval(loss_function)(y, y_hat)  # 计算loss
                for i in range(len(self.network))[::-1]: loss_grad = self.network[i].backward(loss_grad,learning_rate)  # 反向传播
                print(loss.mean())

    def train2(self, train_x, train_y, batch_size, loss_function="MSE", learning_rate=0.1, epochs=10):
        # step=0
        for i in range(int(train_x.shape[0] / batch_size)):
            x = train_x[i * batch_size:(i + 1) * batch_size]
            y = train_y[i * batch_size:(i + 1) * batch_size]
            print('epoch')
            result=[]
            for epoch in range(epochs):
                # step+=1
                # if step>40:learning_rate*=0.99
                y_hat = self.forward(x)  # 向前传播
                (loss, loss_grad) = eval(loss_function)(y, y_hat)  # 计算loss
                for i in range(len(self.network))[::-1]: loss_grad = self.network[i].backward(loss_grad,learning_rate)  # 反向传播
                result.append(loss.mean())
                print(loss.mean())

    def test(self,train_x, train_y,is_Softmax=True):
        right_count=0
        for x,y in zip(train_x,train_y):
            y_hat=self.forward(x)
            if is_Softmax:
                y_hat = np.exp(y_hat - y_hat.max())
                y_hat = (y_hat.T / np.sum(y_hat)).T
                y_hat=y_hat.tolist()
                if y[y_hat.index(max(y_hat))]==1:right_count+=1
        print(right_count/train_x.shape[0])

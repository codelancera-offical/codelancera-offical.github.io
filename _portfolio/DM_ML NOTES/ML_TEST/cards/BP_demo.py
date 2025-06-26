import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# 随机生成三类数据，每类20个样本，数据维度为3, 并设置分别的中心点
np.random.seed(50)
class1 = np.random.randn(20, 3) + [1, 1, 1]
class2 = np.random.randn(20, 3) + [4, 2, 1]
class3 = np.random.randn(20, 3) + [5, 2, 9]

X = np.vstack((class1, class2, class3))  # 按行堆叠数据
y = np.array([0]*20 + [1]*20 + [2]*20)  # 指定每个样本的类别标签

# 划分训练验证集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

# 独热编码
encoder = OneHotEncoder(sparse=False)
y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_onehot = encoder.transform(y_test.reshape(-1, 1))

# 基类层
class Layer:
    # 前向传播函数，根据上一层输入x计算
    def forward(self, x):
        raise NotImplementedError  # 未实现错误

    # 反向传播函数，输入下一层回传的梯度grad, 输出当前层的梯度
    def backward(self, grad):
        raise NotImplementedError

    # 更新函数，用于更新当前层的参数
    def update(self, learning_rate):
        pass

class Linear(Layer):
    def __init__(self, num_in, num_out, use_bias=True):
        self.num_in = num_in  # 输入维度
        self.num_out = num_out  # 输出维度
        self.use_bias = use_bias  # 是否添加偏置

        # 参数的初始化（绝对不能初始化为0！不然后续计算失去意义）
        # 用正态分布来初始化W
        self.W = np.random.normal(loc=0, scale=1.0, size=(num_in, num_out))
        if use_bias:
            self.b = np.zeros((1, num_out))

    def forward(self, x):
        # 前向传播 y = xW + b
        # x的维度为(batch_size, num_in)
        self.x = x
        self.y = x @ self.W  # y的维度为(batch_size, num_out)
        if self.use_bias:
            self.y += self.b
        return self.y

    def backward(self, grad):
        # 反向传播，按照链式法则计算
        # grad的维度为(batch_size, num_out)
        # 梯度应该对batch_size去平均值
        # grad_W的维度应该与W相同，为(num_in, num_out)
        self.grad_W = self.x.T @ grad / grad.shape[0]
        if self.use_bias:
            # grad_b的维度与b相同，(1, num_out)
            self.grad_b = np.mean(grad, axis=0, keepdims=True)
        # 往上一层传递的grad维度应该为(batch_size, num_in)
        grad = grad @ self.W.T
        return grad

    def update(self, learning_rate):
        # 更新参数以完成梯度下降
        self.W -= learning_rate * self.grad_W
        if self.use_bias:
            self.b -= learning_rate * self.grad_b

# 激活层设计
class Identity(Layer):
    def forward(self, x):
        return x
    def backward(self, grad):
        return grad

class Sigmoid(Layer):
    def forward(self, x):
        self.x = x
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def backward(self, grad):
        return grad * self.y * (1 - self.y)

class Tanh(Layer):
    def forward(self, x):
        self.x = x
        self.y = np.tanh(x)
        return self.y

    def backward(self, grad):
        return grad * (1 - self.y ** 2)

class ReLU(Layer):
    def forward(self, x):
        self.x = x
        self.y = np.maximum(x, 0)
        return self.y

    def backward(self, grad):
        return grad * (self.x >= 0)

class Softmax(Layer):
    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.y = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.y

    def backward(self, grad):
        return grad

# 存储所有激活函数和对应名称，方便索引
activation_dict = {
    'identity': Identity,
    'sigmoid': Sigmoid,
    'tanh': Tanh,
    'relu': ReLU,
    'softmax': Softmax
}

# MLP类定义
class MLP:
    def __init__(self, layer_sizes, use_bias=True, activation='relu', out_activation='softmax'):
        self.layers = []
        num_in = layer_sizes[0]
        for num_out in layer_sizes[1:-1]:
            # 添加全连接层
            self.layers.append(Linear(num_in, num_out, use_bias))
            # 添加激活函数
            self.layers.append(activation_dict[activation]())
            num_in = num_out
        # 最后一层特殊处理
        self.layers.append(Linear(num_in, layer_sizes[-1], use_bias))
        self.layers.append(activation_dict[out_activation]())

    def forward(self, x):
        # 前向传播，将输入依次通过每一层
        for layer in self.layers:
            x = layer.forward(x)  # 确保x被正确传递
        return x

    def backward(self, grad):
        # 反向传播, grad为损失函数对输出的梯度，将该梯度依次回传，得到每一层参数的梯度
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def update(self, learning_rate):
        # 更新每一层的参数
        for layer in self.layers:
            layer.update(learning_rate)

# 设置训练参数
num_epochs = 10000
learning_rate = 0.05
batch_size = 20
eps = 1e-7  # 用于防止除0，log(0)等问题

# 创建一个层大小依次为3, 8, 3的多层感知机
# 对于多分类问题，使用softmax作为输出层的激活函数
mlp = MLP(layer_sizes=[3, 8, 3], use_bias=True, activation='relu', out_activation='softmax')

# 记录损失和准确率
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(num_epochs):
    st = 0
    loss = 0.0
    while st < len(X_train):
        ed = min(st + batch_size, len(X_train))
        # 取出batch
        x_batch = X_train[st:ed]
        y_batch = y_train_onehot[st:ed]
        # 计算MLP的预测
        y_pred = mlp.forward(x_batch)
        # 计算损失
        batch_loss = -np.sum(np.log(y_pred + eps) * y_batch) / y_batch.shape[0]
        loss += batch_loss
        # 计算梯度并进行反向传播
        grad = y_pred - y_batch
        mlp.backward(grad)
        # 更新参数
        mlp.update(learning_rate)
        st = ed
    loss /= (len(X_train) / batch_size)
    train_losses.append(loss)
    # 计算训练准确率
    train_acc = np.mean(np.argmax(mlp.forward(X_train), axis=1) == y_train)
    train_accuracies.append(train_acc)
    # 计算测试损失和准确率
    test_loss = -np.sum(np.log(mlp.forward(X_test) + eps) * y_test_onehot) / y_test_onehot.shape[0]
    test_losses.append(test_loss)
    test_acc = np.mean(np.argmax(mlp.forward(X_test), axis=1) == y_test)
    test_accuracies.append(test_acc)
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Train Loss: {loss:.4f}, Test Loss: {test_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

# 可视化训练和测试的损失与准确率
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.show()

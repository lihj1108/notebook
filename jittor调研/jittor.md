##### 1.简介

计图（Jittor）：一个完全基于动态编译（Just-in-time）,内部使用创新的元算子和统一计算图的深度学习框架， 元算子和Numpy一样易于使用，并且超越Numpy能够实现更复杂更高效的操作。而统一计算图则是融合了静态计算图和动态计算图的诸多优点，在易于使用的同时，提供高性能的优化。基于元算子开发的深度学习模型，可以被计图实时的自动优化并且运行在指定的硬件上，如CPU，GPU，TPU。

##### 2.简单使用

###### 1.变量的创建和操作

```python
import numpy as np
import jittor as jt

# 直接创建Var
var_data = jt.Var([[1, 2],[3, 4],[5, 6]])
var1 = jt.ones((3,4))

# 从numpy数组创建Var
np_array = np.array([[1, 2],[3, 4],[5, 6]]) 
var_np_array=jt.array(np_array) 

# 从其他Var中创建Var
var_ones = jt.ones_like(var_data) 
var_zeros = jt.zeros_like(var_data) 

# 常见的操作
var2 = var1.reshape(4,3) # 改变形状
var3 = var1*3 # 与实数相乘
var4 = var1*var2 # 两个Var进行标量相乘，即对应元素位置相乘
var5 = var1@var1 # 两个Var进行矩阵相乘
var3 = var1.transpose(3,4) # 转置
```

###### 2.训练神经网络模型（以MNIST图片分类为例）

- 1.导入必要的包

```python
import jittor as jt
from jittor import nn, Module
import numpy as np
import gzip
from PIL import Image
from jittor.dataset import Dataset
from jittor_utils.misc import download_url_to_local
```

- 2.构建数据集(Dataset)类

  需要继承jittor.dataset.Dataset，并且在类中实现init和getitem方法，如果数据集需要在线下载，则需实现download_url方法

```python
class MNIST(Dataset):
    def __init__(self, data_root="./mnist_data/", train=True ,download=True, batch_size=1, shuffle=False):
        super().__init__()
        self.data_root = data_root # 数据集的根目录
        self.batch_size = batch_size # 每次训练采样的批大小
        self.shuffle = shuffle # 是否打乱样本
        self.is_train = train # 是否是训练模式
        if download == True:
            self.download_url()

        filesname = [
                "train-images-idx3-ubyte.gz",
                "t10k-images-idx3-ubyte.gz",
                "train-labels-idx1-ubyte.gz",
                "t10k-labels-idx1-ubyte.gz"
        ]
        self.mnist = {}
        if self.is_train:
            with gzip.open(data_root + filesname[0], 'rb') as f:
                self.mnist["images"] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28, 28)
            with gzip.open(data_root + filesname[2], 'rb') as f:
                self.mnist["labels"] = np.frombuffer(f.read(), np.uint8, offset=8)
        else:
            with gzip.open(data_root + filesname[1], 'rb') as f:
                self.mnist["images"] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28, 28)
            with gzip.open(data_root + filesname[3], 'rb') as f:
                self.mnist["labels"] = np.frombuffer(f.read(), np.uint8, offset=8)
        assert(self.mnist["images"].shape[0] == self.mnist["labels"].shape[0])
        self.total_len = self.mnist["images"].shape[0]
        self.set_attrs(batch_size = self.batch_size, total_len=self.total_len, shuffle= self.shuffle)

    def __getitem__(self, index):
        img = Image.fromarray (self.mnist['images'][index])
        img = np.array (img)
        img = img[np.newaxis, :]
        return np.array((img / 255.0), dtype = np.float32), self.mnist['labels'][index]

    def download_url(self):
        resources = [
            ("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
            ("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
            ("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
            ("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
        ]
        for url, md5 in resources:
            filename = url.rpartition('/')[2]
            download_url_to_local(url, filename, self.data_root, md5)
```

- 3.构建模型(Model)类

  需要继承jittor.Module类，实现init方法和execute方法

```python
class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv(3, 32, 3, 1)           # 卷积层 1，参数含义：该层输入通道 3，输出通道 32，卷积核大小 3*3，移动步长为 1
        self.conv2 = nn.Conv(32, 64, 3, 1)          # 卷积层 2，参数含义：该层输入通道 32，输出通道 64，卷积核大小 3*3，移动步长为 1
        self.bn = nn.BatchNorm(64)                  # 批量归一化层，参数含义：该层输入通道数为 64

        self.max_pool = nn.Pool(2, 2)               # 池化层，参数含义：窗口大小为 2，窗口移动步长为 2
        self.relu = nn.Relu()                       # 非线性激活函数 Relu
        self.fc1 = nn.Linear(64 * 12 * 12, 256)     # 线性全连接 1，参数含义：输入通道数 64*12*12（由上一步reshape变化得来），输出通道数 256
        self.fc2 = nn.Linear(256, 10)               # 线性全连接 2，参数含义：输入通道数 256，输出通道数 10
    
    def execute(self, x) :
        x = self.conv1(x)                           # 作用第一层卷积层，输入由 batch_size*3*28*28 变为输出 batch_size*32*26*26
        x = self.relu(x)                            # 通过非线性激活函数 Relu
        
        x = self.conv2(x)                           # 作用第二层卷积层，输入由 batch_size*32*26*26 变为输出 batch_size*64*24*24
        x = self.bn(x)                              # 批量归一化操作
        x = self.relu(x)                            # 通过非线性激活函数 Relu

        x = self.max_pool(x)                        # 池化操作，输入由 batch_size*64*24*24 变为输出 batch_size*64*12*12
        x = jt.reshape(x, [x.shape[0], -1])         # 将 x 压缩成只保留第一维度，输入由 batch_size*64*12*12 变为输出 batch_size*(64*12*12)
        x = self.fc1(x)                             # 作用第一层全连接，输入由 batch_size*(64*12*12) 变为输出 batch_size*256
        x = self.relu(x)                            # 通过非线性激活函数 Relu
        x = self.fc2(x)                             # 第二层全连接，并控制最后输出为 batch_size*10，每个数据的 10 个分量，分别代表十个数字的相似度
        return x
```

- 4.定义训练和验证函数

```python
# 训练函数
def train(model, train_loader, optimizer, epoch, losses, losses_idx):
    model.train() # 开启训练模式
    lens = len(train_loader) # 初始化 Loss 容器，用于记录每一批次的 Loss
    for batch_idx, (inputs, targets) in enumerate(train_loader): # 通过训练集加载器，按批次迭代数据
        outputs = model(inputs) # 通过模型预测手写数字。outputs 中每个数据输出有 10 个分量，对应十个数字的相似度
        loss = nn.cross_entropy_loss(outputs, targets)  # 计算损失函数
        optimizer.step(loss) # 根据损失函数，对模型参数进行优化、更新
        losses.append(loss.numpy()[0])  # 记录该批次的 Loss
        losses_idx.append(epoch * lens + batch_idx)
        if batch_idx % 10 == 0: # 每十个批次，打印一次训练集上的 Loss 
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader),
                100. * batch_idx / len(train_loader), loss.numpy()[0]))
# 验证函数
def val(model, val_loader, epoch):
    model.eval()
    total_acc = 0
    total_num = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        batch_size = inputs.shape[0]
        outputs = model(inputs)
        pred = np.argmax(outputs.numpy(), axis=1) # 生成预测结果
        acc = np.sum(targets.numpy() == pred) # 计算准确率
        total_acc += acc
        total_num += batch_size
        acc = acc / batch_size
        print(f'Test Epoch: {epoch} [{batch_idx}/{len(val_loader)}]\tAcc: {acc:.6f}')
        print('Test Acc =', total_acc / total_num)
```

- 5.设置超参数，开始训练

```python
if __name__ == '__main__':
    batch_size = 64 # 批大小
    learning_rate = 0.1 # 学习率
    momentum = 0.9 # 动量
    weight_decay = 1e-4 # 权重衰减
    epochs = 1 # 训练的纪元数
    losses = []
    losses_idx = []
    train_loader = MNIST(train=True, batch_size=batch_size, shuffle=True) # 实例化训练集，得到样本采集器
    val_loader = MNIST(train=False, batch_size=1, shuffle=False) # 实例化验证集，得到样本采集器
    model = Model() # 实例化模型
    optimizer = nn.SGD(model.parameters(), learning_rate, momentum, weight_decay) # 设置优化器
    for epoch in range(epochs): # 开始训练循环
        train(model, train_loader, optimizer, epoch, losses, losses_idx)
        val(model, val_loader, epoch)
```



##### 3.分布式训练

计图分布式基于MPI（Message Passing Interface），需要安装OpenMPI，用户无需修改代码，需要做的仅仅是修改启动命令行，计图就会用数据并行的方式自动完成并行操作。

```shell
# 单卡训练代码
python3.7 -m jittor.test.test_resnet
# 分布式多卡训练代码
mpirun -np 4 python3.7 -m jittor.test.test_resnet
# 指定特定显卡的多卡训练代码
CUDA_VISIBLE_DEVICES="2,3" mpirun -np 2 python3.7 -m jittor.test.test_resnet
```

大部分情况下，单卡训练的代码可以直接使用`mpirun`实现分布式多卡运行。 但仍然如下几种情况下，需要对代码进行调整：

1. 对硬盘进行写操作（保存模型，保存曲线），通过指定jt.rank，在指定的进程上保存
2. 需要统计全局信息（validation 上的全局准确率），通过mpi_all_reduce()来实现

##### 4.总结

jittor预置了大量的模型仓库，例如GAN、实例分割、语义分割、点云等

jittor的api风格和pytorch的非常像，是一个python的第三方框架，感觉集成到平台上没啥问题

分布式支持多卡训练，目前的教程很少，只找到这一个：[计图MPI多卡分布式教程 — Jittor (tsinghua.edu.cn)](https://cg.cs.tsinghua.edu.cn/jittor/tutorial/2020-5-2-16-44-distributed/)
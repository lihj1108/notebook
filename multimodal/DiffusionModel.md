## DDPM(Denoising Diffusion Probabilistic Models)

Paper:https://arxiv.org/pdf/2006.11239.pdf

github:https://github.com/hojonathanho/diffusion

Website:https://hojonathanho.github.io/diffusion/

#### 概述：

训练集：

Laion，5.85B图片文本数据对

训练：

对原图不断的加高斯噪声，并记录每个step加的噪声，得到训练集。训练时，模型的输入是step id和噪声图片，训练优化的目标就是最小化ground truth噪声和模型输出的噪声的损失，通过KL散度来衡量两个噪声分布的差异。神经网络学习的去躁过程中生成噪声分布的参数

最大似然估计等价于最小化KL散度

推理：

推理就是一个去噪的过程，输入一个带噪声的图片输出一个去除噪声后的图片。去噪的过程是一步步进行的，每个step中，神经网络模型会预测一个噪声，上一个step的图片减去这个噪声，就得到这个step的图片，直至经历一定的step后，得到最终的图片（雕像本来就在大理石里面）
### vision models

#### 1.BEiT

https://arxiv.org/abs/2106.08254

15 Jun 2021

一种自监督视觉表示模型，BEiT 模型不是预先训练模型来预测图像的类别（如原始 ViT 论文中所做的那样），而是预先训练模型以根据给定屏蔽补丁的 OpenAI DALL-E 模型的代码本来预测视觉标记。

#### 2.Conditional DETR

https://arxiv.org/abs/2108.06152

13 Aug 2021

从解码器嵌入中学习条件空间查询，以实现解码器多头交叉注意

#### 3.ConvNeXT

https://arxiv.org/abs/2201.03545

10 Jan 2022

是一个纯卷积模型（ConvNet），受到 Vision Transformers 设计的启发

#### 4.ConvNeXt V2

https://arxiv.org/abs/2301.00808

2 Jan 2023

提出了一个完全卷积掩码自动编码器框架和一个新的全局响应归一化（GRN）层，可以将其添加到 ConvNeXt 架构中以增强通道间特征竞争

#### 5.Convolutional Vision Transformer (CvT)

https://arxiv.org/abs/2103.15808

29 Mar 2021

提出了一种名为卷积视觉变换器（CvT）的新架构，它通过将卷积引入到视觉变换器（ViT）中来提高视觉变换器（ViT）的性能和效率，以产生两种设计的最佳效果

#### 6.Deformable DETR

https://arxiv.org/abs/2010.04159

8 Oct 2020

通过利用新的可变形注意模块来缓解原始 DETR 的缓慢收敛问题和有限的特征空间分辨率，该模块仅关注参考周围的一小组关键采样点

#### 7.DeiT

https://arxiv.org/abs/2012.12877

23 Dec 2020

DeiT 模型使用所谓的蒸馏令牌来有效地向老师学习（在 DeiT 论文中，这是一个类似 ResNet 的模型）。 蒸馏标记是通过反向传播、通过与类（[CLS]）交互以及通过自注意力层修补标记来学习的

#### 8.DETA

https://arxiv.org/abs/2212.06137

12 Dec 2022

DETA（带有分配的检测变压器的缩写）通过将一对一二分匈牙利匹配损失替换为传统检测器中使用的非极大值抑制（NMS）的一对多标签分配来改进可变形DETR

#### 9.DETR

https://arxiv.org/abs/2005.12872

26 May 2020

由一个卷积主干和一个编码器-解码器 Transformer 组成，可以进行端到端的目标检测训练

#### 10.Dilated Neighborhood Attention Transformer

https://arxiv.org/abs/2209.15001

29 Sep 2022

介绍了扩张邻域注意力（DiNA），这是对（Neighborhood Attention）NA 的一种自然、灵活且高效的扩展，可以捕获更多的全局上下文并以指数方式扩展感受野，而无需额外成本

#### 11.DINOv2

https://arxiv.org/abs/2304.07193

14 Apr 2023

DINOv2 是 DINO 的升级版，DINO 是一种应用于 Vision Transformers 的自监督方法。 该方法支持通用视觉特征，即无需微调即可跨图像分布和任务工作的特征

#### 12.DIT

https://arxiv.org/abs/2203.02378

4 Mar 2022

一种自监督的预训练文档图像转换器模型，使用大规模未标记的文本图像来执行文档 AI 任务，这是至关重要的，因为由于缺乏人类标记的文档图像，不存在有监督的对应物

#### 13.DPT

https://arxiv.org/abs/2103.13413

*24 Mar 2021*

DPT 是一种利用 Vision Transformer (ViT) 作为语义分割和深度估计等密集预测任务骨干的模型

#### 14.EfficientFormer

https://arxiv.org/abs/2206.01191

2 Jun 2022 

EfficientFormer 提出了一种维度一致的纯 Transformer，可以在移动设备上运行，用于图像分类、对象检测和语义分割等密集预测任务。

#### 15.EfficientNet

https://arxiv.org/abs/1905.11946

28 May 2019

系统地研究了模型缩放，并发现仔细平衡网络深度、宽度和分辨率可以带来更好的性能。 基于这一观察，我们提出了一种新的缩放方法，使用简单而高效的复合系数来统一缩放深度/宽度/分辨率的所有维度

#### 16.FocalNets

https://arxiv.org/abs/2203.11926

22 Mar 2022

通过焦点调制机制完全取代了自注意力（在 ViT 和 Swin 等模型中使用），用于对视觉中的 token 交互进行建模

#### 17.GLPN

https://arxiv.org/abs/2201.07436

19 Jan 2022

将 SegFormer 的分层混合 Transformer 与用于单目深度估计的轻量级解码器相结合

#### 18.ImageGPT

https://openai.com/research/image-gpt#conclusion

17 June 2020

是一种类似 GPT-2 的模型，经过训练可以预测下一个像素值，允许无条件和条件图像生成

#### 19.LeViT

https://arxiv.org/abs/2104.01136

2 Apr 2021

通过一些架构差异（例如 Transformer 中分辨率降低的激活图以及引入注意力偏差来集成位置信息）提高了 Vision Transformer (ViT) 的性能和效率

#### 20.Mask2Former

https://arxiv.org/abs/2112.01527

2 Dec 2021

是一个用于全景、实例和语义分割的统一框架，其关键组成部分包括屏蔽注意力，它通过限制预测屏蔽区域内的交叉注意力来提取局部特征

#### 21.MobileNet V2

https://arxiv.org/abs/1801.04381

13 Jan 2018

MobileNet 基于流线型架构，使用深度可分离卷积来构建轻量级深度神经网络

#### 22.MobileViT

https://arxiv.org/abs/2110.02178

5 Oct 2021

是一种适用于移动设备的轻量级通用视觉转换器。 MobileViT 为使用 Transformer 进行全局信息处理提供了不同的视角，即将 Transformer 作为卷积

#### 23.NAT

https://arxiv.org/abs/2204.07143

14 Apr 2022

是第一个高效且可扩展的视觉滑动窗口注意力机制。（Neighborhood Attention）NA 是一种逐像素操作，将自注意力 (SA) 定位到最近的相邻像素，因此与 SA 的二次复杂度相比，具有线性时间和空间复杂度。 与 Swin Transformer 的窗口自注意力 (WSA) 不同，滑动窗口模式允许 NA 的感受野增长，而无需额外的像素移位，并保留平移等方差

#### 24.PoolFormer

[[2111.11418\] MetaFormer Is Actually What You Need for Vision (arxiv.org)](https://arxiv.org/abs/2111.11418)

22 Nov 2021

这项工作的目标不是设计复杂的令牌混合器来实现 SOTA 性能，而是证明变压器模型的能力很大程度上源于 MetaFormer 的通用架构

#### 25.Pyramid Vision Transformer (PVT)

https://arxiv.org/abs/2102.12122

24 Feb 2021

PVT 是一种视觉转换器，利用金字塔结构使其成为密集预测任务的有效骨干。空间减少注意（SRA）层用于进一步减少学习高分辨率特征时的资源消耗

#### 26.RegNet

https://arxiv.org/abs/2003.13678

30 Mar 2020

设计了搜索空间来执行神经架构搜索（NAS）。 他们首先从高维搜索空间开始，并通过基于当前搜索空间采样的最佳性能模型凭经验应用约束来迭代地减少搜索空间。良好网络的宽度和深度可以通过量化的线性函数来解释

#### 27.ResNet

https://arxiv.org/abs/1512.03385

10 Dec 2015

提出了一个残差学习框架，以简化比以前使用的网络更深的网络训练，明确地将层重新表示为参考层输入的学习残差函数，而不是学习未引用的函数

#### 28.SegFormer

https://arxiv.org/abs/2105.15203

31 May 2021

该模型由分层 Transformer 编码器和轻量级全 MLP 解码头组成

#### 29.SwiftFormer

https://arxiv.org/abs/2303.15446

27 Mar 2023

介绍了一种新颖的高效加性注意力机制，该机制可以用线性逐元素乘法有效地取代自注意力计算中的二次矩阵乘法运算

#### 30.Swin Transformer

https://arxiv.org/abs/2103.14030

25 Mar 2021

提出了一个分层 Transformer，其表示是用Shifted windows 计算的。 移位窗口方案通过将自注意力计算限制在非重叠的本地窗口，同时还允许跨窗口连接，带来了更高的效率。

#### 31.Swin2SR

https://arxiv.org/abs/2209.11345

22 Sep 2022

Swin2R 通过合并 Swin Transformer v2 层改进了 SwinIR 模型，从而缓解了训练不稳定、预训练和微调之间的分辨率差距以及数据匮乏等问题。

#### 32.Table Transformer

https://arxiv.org/abs/2110.00061

30 Sep 2021

引入了一个新的数据集 PubTables-1M，用于对非结构化文档中的表格提取以及表格结构识别和功能分析的进度进行基准测试。 作者训练了 2 个 DETR 模型，一个用于表格检测，一个用于表格结构识别，称为 Table Transformers

#### 33.TimeSformer

https://arxiv.org/abs/2102.05095

*9 Feb 2021*

提出了一种无卷积的视频分类方法，专门建立在空间和时间上的自注意力基础上。通过直接从帧级补丁序列进行时空特征学习，将标准 Transformer 架构应用于视频，表明“分散注意力”（即时间注意力和空间注意力分别应用在每个块内）可以在所考虑的设计选择中获得最佳的视频分类准确性

#### 34.VAN

https://arxiv.org/abs/2202.09741

20 Feb 2022

引入了一种基于卷积运算的新注意力层，能够捕获本地和远程关系。 这是通过组合普通卷积层和大内核卷积层来完成的。 后者使用扩张卷积来捕获远距离相关性。

#### 35.VideoMAE

https://arxiv.org/abs/2203.12602

23 Mar 2022

VideoMAE 将屏蔽自动编码器 (MAE) 扩展到视频，受到最近的 ImageMAE 的启发，提出了定制的视频管遮蔽和重建

#### 36.Vision Transformer (ViT)

https://arxiv.org/abs/2010.11929

22 Oct 2020

是第一篇在 ImageNet 上成功训练 Transformer 编码器的论文，与熟悉的卷积架构相比，取得了非常好的结果

#### 37.ViTDet

https://arxiv.org/abs/2203.16527

30 Mar 2022

VitDet 利用普通 Vision Transformer 来完成物体检测任务，探索简单的、非分层的 Vision Transformer (ViT) 作为对象检测的骨干网络

#### 38.ViTMAE

https://arxiv.org/abs/2111.06377v2

11 Nov 2021

通过预训练视觉变换器（ViT）来重建掩模补丁的像素值，在微调后可以获得优于监督预训练的结果。

#### 39.ViTMatte

https://arxiv.org/abs/2305.15272

24 May 2023

ViTMatte 利用普通 Vision Transformer 来执行图像抠图任务，这是准确估计图像和视频中前景对象的过程

#### 40.ViTMSN

https://arxiv.org/abs/2103.15691

29 Mar 2021

提出了第一套成功的基于纯 Transformer 的视频理解模型

#### 41.YOLOS

https://arxiv.org/abs/2106.00666

1 Jun 2021

受到 DETR 的启发，YOLOS 建议仅利用普通视觉变换器 (ViT) 进行目标检测
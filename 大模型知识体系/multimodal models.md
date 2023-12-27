### multimodal models

#### 1.ALIGN

https://arxiv.org/abs/2102.05918

11 Feb 2021

采用双编码器架构，以EfficientNet作为视觉编码器，以BERT为文本编码器，并学习通过对比学习来对齐视觉和文本表示

#### 2.AltCLIP

https://arxiv.org/abs/2211.06679v2

12 Nov 2022

将 CLIP 的文本编码器替换为预训练的多语言文本编码器XLM-R

#### 4.BLIP

https://arxiv.org/abs/2201.12086

28 Jan 2022

BLIP是一种新的 vision-language pre-trained框架，可以灵活地迁移到视觉语言理解和生成任务。 BLIP 通过引导字幕来有效地利用嘈杂的网络数据，其中字幕生成器生成合成字幕，而过滤器则去除嘈杂的字幕

#### 5.BLIP2

https://arxiv.org/abs/2301.12597

30 Jan 2023

BLIP-2 利用冻结的预训练图像编码器和大型语言模型 (LLM)，在它们之间训练轻量级 12 层 Transformer 编码器，从而在各种视觉语言任务上实现最先进的性能。BLIP-2 通过轻量级查询转换器弥补了模态差距，该转换器分两个阶段进行预训练。 第一阶段从冻结图像编码器引导视觉语言表示学习。 第二阶段从冻结的语言模型引导视觉到语言的生成学习

#### 6.BridgeTower

https://arxiv.org/abs/2206.08657

17 Jun 2022

引入了多个桥接层，在单模态编码器的顶层和跨模态编码器的每一层之间建立连接。 这使得跨模态编码器中的预训练单模态编码器的不同语义级别的视觉和文本表示之间能够实现有效的自下而上的跨模态对齐和融合

#### 7.BROS

https://arxiv.org/abs/2108.04539

10 Aug 2021

对2D空间中文本的相对位置进行编码，并使用区域遮蔽策略从未标记的文档中学习

#### 8.CLIP

https://arxiv.org/abs/2103.00020

26 Feb 2021

使用类似 ViT 的转换器来获取视觉特征，并使用因果语言模型来获取文本特征。 然后，文本和视觉特征都被投影到具有相同维度的潜在空间。 然后将投影图像和文本特征之间的点积用作相似分数

#### 9.Data2Vec

https://arxiv.org/abs/2202.03555

7 Feb 2022

提出了一个统一的框架，用于跨不同数据模式(文本、音频和图像)的自我监督学习, 其核心思想是在使用标准Transformer体系结构的自蒸馏设置中，基于输入的屏蔽视图来预测完整输入数据的潜在表示

#### 10.DePlot

https://arxiv.org/abs/2212.10505

20 Dec 2022

将图像或图表转换为表格，DePlot的输出可以利用LLM的小样本推理能力直接用于提示预训练的大型语言模型(LLM)，建立了统一的任务格式和指标来标准化图表到表格的任务

#### 11.Donut

https://arxiv.org/abs/2111.15664

30 Nov 2021

是一个无OCR的视觉文档理解（visual document understanding）模型，由图像Transformer编码器和自回归文本Transformer解码器组成

#### 12.FLAVA

https://arxiv.org/abs/2112.04482

8 Dec 2021

提出了一个基本的视觉和语言对齐模型

#### 13.GIT

https://arxiv.org/abs/2205.14100

27 May 2022

是一个生成式图像到文本转换器，GIT 是一个encoder-only的Transformer，它利用CLIP的视觉编码器根据视觉输入来调节模型

#### 14.InstructBLIP

https://arxiv.org/abs/2305.06500

11 May 2023

基于预训练的BLIP-2模型对视觉语言指令调优进行了系统、全面的研究。引入了指令感知视觉特征提取，使模型能够提取适合给定指令的信息特征

#### 15.KOSMOS-2

https://arxiv.org/abs/2306.14824

26 Jun 2023

是一种基于 Transformer 的因果语言模型，并使用基于图像文本对 GRIT 的网络规模数据集上的下一个单词预测任务进行训练。支持感知对象描述（例如边界框）和将文本融入视觉世界的功能，

#### 16.LayoutLM

https://arxiv.org/abs/2204.08387

18 Apr 2022

对扫描文档图像中的文本和布局信息之间的交互进行联合建模，适用于以文本为中心和以图像为中心的文档 AI 任务的通用预训练模型

#### 17.LXMERT

https://arxiv.org/abs/1908.07490

20 Aug 2019

该模型由三个编码器组成：对象关系编码器、语言编码器和跨模态编码器

#### 18.MGP-STR

https://arxiv.org/abs/2209.03592

8 Sep 2022

场景文本识别模型，以隐式方式将语言模态的信息注入到模型中，即将 NLP 中广泛使用的子词表示（BPE 和 WordPiece）引入输出空间

#### 19.Nougat

https://arxiv.org/abs/2308.13418

25 Aug 2023

使用与 Donut 相同的架构，即图像 Transformer 编码器和自回归文本 Transformer 解码器，可将科学 PDF 转换为 Markdown

#### 20.OneFormer

https://arxiv.org/abs/2211.06220

10 Nov 2022

一种通用图像分割框架，借助新的 ConvNeXt 和 DiNAT 主干

#### 21.OWLv2

https://arxiv.org/abs/2306.09683

16 Jun 2023

使用标准的 Vision Transformer 架构，是一个开放词汇对象检测网络，在各种（图像、文本）对上进行训练。 它可以用于通过一个或多个文本查询来查询图像，以搜索和检测文本中描述的目标对象

#### 22.Perceiver

https://arxiv.org/abs/2107.14795

30 Jul 2021

除了处理任意输入外，还可以处理任意输出。分类标签之外，Perceiver IO 还可以生成（例如）语言、光流和带有音频的多模态视频

#### 23.Pix2Struct

https://arxiv.org/abs/2210.03347

7 Oct 2022

用于纯粹视觉语言理解的预训练图像到文本模型，可以针对包含视觉情境语言的任务进行微调，包括图像字幕、不同输入（书籍、图表、科学图表）的视觉问答 (VQA)、字幕 UI 组件等

#### 24.SAM

https://arxiv.org/abs/2304.02643

5 Apr 2023

该模型可用于预测给定输入图像的任何感兴趣对象的分割掩模

#### 25.TrOCR

https://arxiv.org/abs/2109.10282

21 Sep 2021

由图像 Transformer 编码器和自回归文本 Transformer 解码器组成，用于执行光学字符识别 (OCR)

#### 26.TVLT

https://arxiv.org/abs/2209.14156

28 Sep 2022

提出了无文本的视觉语言转换器（TVLT），其中同质转换器块采用原始视觉和音频输入，以最小的特定模态设计进行视觉和语言表示学习。TVLT 通过重建连续视频帧和音频频谱图的屏蔽补丁（屏蔽自动编码）以及对比建模来对齐视频和音频来进行训练

#### 27.ViLT

https://arxiv.org/abs/2102.03334

5 Feb 2021

是一个视觉和语言转换器

#### 28.VisualBERT

https://arxiv.org/abs/1908.03557

9 Aug 2019

是一种多模态视觉和语言模型，VisualBERT 使用类似 BERT 的转换器来准备图像文本对的嵌入。 然后，文本和视觉特征都被投影到具有相同维度的潜在空间。

#### 29.X-CLIP

https://arxiv.org/abs/2208.02816

4 Aug 2022

X-CLIP 是视频 CLIP 的最小扩展。 该模型由文本编码器、跨帧视觉编码器、多帧集成 Transformer 和视频专用提示生成器组成。为了捕获帧在时间维度上的远程依赖性，我们提出了一种跨帧注意机制，可以显式地跨帧交换信息。提出了一种特定于视频的提示方案，该方案利用视频内容信息来生成有区别的文本提示
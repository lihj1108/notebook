## text models

#### 1.ALBERT

https://arxiv.org/abs/1909.11942

26 Sep 2019

结构：encoder-only

训练特点：

采用绝对位置编码

使用重复层，因此减少内存占用

embedding size 小于 hidden size，减少参数

将bert中的下一个句子预测改成句子排序预测，预测两个句子的顺序是否被交换

#### 2.BART

论文：[[1910.13461\] BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension (arxiv.org)](https://arxiv.org/abs/1910.13461)

时间：29 Oct 2019

结构：encoder-decoder

训练特点：

采用绝对位置编码

使用单个掩码标记来掩码k个标记范围（spans of text are replaced with a single mask token）

排列句子

旋转文档以使其从特定标记开始

#### 3.BERT

论文：[[1810.04805\] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (arxiv.org)](https://arxiv.org/abs/1810.04805)

时间：11 Oct 2018

结构：encoder-only

训练特点：

采用绝对位置编码

掩码语言建模目标和下一句预测相结合的方式进行预训练

#### 4.BigBird

论文：[[2007.14062\] Big Bird: Transformers for Longer Sequences (arxiv.org)](https://arxiv.org/abs/2007.14062)

时间：28 Jul 2020

结构：encoder-only

训练特点：

提出用稀疏注意力、全局注意力和随机注意力来替代完全注意力

#### 5.Blenderbot

论文：[[2004.13637\] Recipes for building an open-domain chatbot (arxiv.org)](https://arxiv.org/abs/2004.13637)

时间：28 Apr 2020

结构：encoder-decoder

训练特点：

采用绝对位置编码

增大了模型参数到9.4B

#### 6.BLOOM

论文：[[2211.05100\] BLOOM: A 176B-Parameter Open-Access Multilingual Language Model (arxiv.org)](https://arxiv.org/abs/2211.05100)

时间：9 Nov 2022

结构：decoder-only

训练特点：

大量的模型参数，达到了176B

#### 7.CANINE

论文：[[2103.06874\] CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation (arxiv.org)](https://arxiv.org/abs/2103.06874)

时间：11 Mar 2021

结构：encoder-only

训练特点：

直接对字符进行编码，没有显示的是有volcabulary和tokenizer

#### 8.CodeGen

论文：[[2203.13474\] CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis (arxiv.org)](https://arxiv.org/abs/2203.13474)

时间：25 Mar 2022

结构：decoder-only

训练特点：

自然语言和代码生成模型

#### 9.CodeLlama

论文：[Code Llama: Open Foundation Models for Code | Research - AI at Meta](https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/)

时间：August 24, 2023

结构：decoder-only

训练特点：

和llama2一样的结构，用于自然语言和代码生成模型

#### 10.ConvBERT

论文：[[2008.02496\] ConvBERT: Improving BERT with Span-based Dynamic Convolution (arxiv.org)](https://arxiv.org/abs/2008.02496)

时间：6 Aug 2020

结构：encoder-only

训练特点：

基于跨度的动态卷积来取代部分的自注意力头，新颖的卷积头与其余的自注意力头一起形成了一个新的混合注意力块，在全局和局部上下文学习方面都更加有效

#### 11.CPM

论文：[[2012.00413\] CPM: A Large-scale Generative Chinese Pre-trained Language Model (arxiv.org)](https://arxiv.org/abs/2012.00413)

时间：1 Dec 2020

结构：decoder-only

训练特点：

结构和gpt-2一样

#### 12.CTRL

论文：[[1909.05858\] CTRL: A Conditional Transformer Language Model for Controllable Generation (arxiv.org)](https://arxiv.org/abs/1909.05858)

时间：11 Sep 2019

结构：decoder-only

训练特点：

采用绝对位置编码，16.3 亿参数

#### 13.DeBERTa

论文：[[2006.03654\] DeBERTa: Decoding-enhanced BERT with Disentangled Attention (arxiv.org)](https://arxiv.org/abs/2006.03654)

时间：5 Jun 2020

结构：encoder-only

训练特点：

解缠结注意力机制，其中每个单词使用分别编码其内容和位置的两个向量来表示，并且单词之间的注意力权重使用其内容和相对位置的解缠结矩阵来计算

增强型掩码解码器替换输出 softmax 层来预测模型预训练的掩码标记

nGiE（nGram 诱导输入编码）DeBERTa-v2 模型除了第一个转换器层之外还使用了一个额外的卷积层，以更好地学习输入标记的局部依赖性

#### 14.DialoGPT

论文：[[1911.00536\] DialoGPT: Large-Scale Generative Pre-training for Conversational Response Generation (arxiv.org)](https://arxiv.org/abs/1911.00536)

时间：1 Nov 2019 

结构：decoder-only

训练特点：

结构跟gpt-2一样

#### 15.DistilBERT

论文：[[1910.01108\] DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter (arxiv.org)](https://arxiv.org/abs/1910.01108)

时间：2 Oct 2019

结构：encoder-only

训练特点：

语言建模、蒸馏和余弦距离损失的三重损失

#### 16.ERNIE3.0

论文：[[2107.02137\] ERNIE 3.0: Large-scale Knowledge Enhanced Pre-training for Language Understanding and Generation (arxiv.org)](https://arxiv.org/abs/2107.02137)

时间：5 Jul 2021

结构：encoder-decoder

训练特点：

10B参数量，训练数据包含了纯文本和大规模知识图谱

#### 17.Falcon

论文：[Falcon LLM (tii.ae)](https://falconllm.tii.ae/falcon.html)

时间：06 Sep, 2023

结构：decoder-only

训练特点：

180b参数量

#### 18.FNet

论文：[[2105.03824\] FNet: Mixing Tokens with Fourier Transforms (arxiv.org)](https://arxiv.org/abs/2105.03824)

时间：9 May 2021

结构：encoder-only

训练特点：

用傅里叶变换替换了 BERT 模型中的自注意力层

#### 19.Funnel Transformer

论文：[[2006.03236\] Funnel-Transformer: Filtering out Sequential Redundancy for Efficient Language Processing (arxiv.org)](https://arxiv.org/abs/2006.03236)

时间：5 Jun 2020

结构：encoder-only

训练特点：

它是一个双向 Transformer 模型，类似于 BERT，但在每个层块之后都有一个池化操作，有点像计算机视觉中的传统卷积神经网络 (CNN)

#### 20.Fuyu

论文：https://www.adept.ai/blog/fuyu-8b

时间：17 October 2023

结构：a decoder-only

训练特点：

具有查询和密钥归一化功能。 添加线性编码器以根据图像输入创建多模态嵌入，通过将图像标记视为文本标记并使用特殊的图像换行符，模型可以知道图像行何时结束。 图像位置嵌入被删除。 这避免了对不同图像分辨率进行不同训练阶段的需要

#### 21.GPT

https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf

是一个因果（单向）转换器，使用大型语料库（多伦多图书语料库）的语言建模进行预训练，该语料库具有长期依赖性

#### 22.GPT2

https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

它是一个因果（单向）转换器，使用语言建模在约 40 GB 文本数据的非常大的语料库上进行预训练, GPT-2 的训练目标很简单：根据某个文本中所有先前的单词来预测下一个单词

#### 23.GPTBigCode

https://arxiv.org/abs/2301.03988

9 Jan 2023

该模型是优化的 GPT2 模型，支持多查询注意力

#### 24.Jukebox

https://arxiv.org/abs/2005.00341

30 Apr 2020

引入了一种生成音乐模型，可以根据艺术家、流派和歌词生成分钟长的样本，使用多尺度 VQ-VAE 将其压缩为离散代码来处理原始音频的长上下文，并使用自回归 Transformer 对其进行建模

#### 25.LED

https://arxiv.org/abs/2004.05150

10 Apr 2020

引入了带有注意力机制的 Longformer，该机制随序列长度线性扩展，从而可以轻松处理数千个标记或更长的文档

#### 26.LLaMA

https://arxiv.org/abs/2302.13971

27 Feb 2023

这是一个基础语言模型的集合，参数范围从 7B 到 65B。在数万亿个代币上训练我们的模型，并表明可以专门使用公开可用的数据集来训练最先进的模型，而无需诉诸专有的和无法访问的数据集

#### 27.LLama2

https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/

July 18, 2023

这是一组经过预训练和微调的大型语言模型 (LLM)，其参数规模从 70 亿到 700 亿不等。 经过微调的LLM（称为 Llama 2-Chat）针对对话用例进行了优化

#### 28.Longformer

https://arxiv.org/abs/2004.05150

10 Apr 2020

引入了带有注意力机制的 Longformer，该机制随序列长度线性扩展，从而可以轻松处理数千个标记或更长的文档。Longformer 的注意力机制是标准自注意力的直接替代品，并将局部窗口注意力与任务驱动的全局注意力结合起来

#### 29.LongT5

https://arxiv.org/abs/2112.07916

15 Dec 2021

LongT5 模型是 T5 模型的扩展，它可以使用两种不同的有效注意力机制之一 - (1) 局部注意力，或 (2) 瞬态全局注意力。

#### 30.LUKE

https://arxiv.org/abs/2010.01057

2 Oct 2020

我们提出了基于双向变压器的新的预训练的单词和实体的上下文表示。 所提出的模型将给定文本中的单词和实体视为独立的标记，并输出它们的上下文表示

#### 31.M2M100

https://arxiv.org/abs/2010.11125

21 Oct 2020

创建了一个真正的多对多多语言翻译模型，可以在 100 种语言中的任意对之间直接进行翻译

#### 32.MarkupLM

https://arxiv.org/abs/2110.08518

16 Oct 2021

我们提出了 MarkupLM，用于以标记语言为骨干的文档理解任务，例如基于 HTML/XML 的文档，其中文本和标记信息是联合预训练的。 实验结果表明，预训练的 MarkupLM 在多个文档理解任务上显着优于现有的强基线模型

#### 33.MBart and MBart-50

https://arxiv.org/abs/2001.08210

22 Jan 2020

MBART 是一种使用 BART 目标在多种语言的大规模单语语料库上进行预训练的序列到序列去噪自动编码器

#### 34.MEGA

https://arxiv.org/abs/2209.10655

21 Sep 2022

一种简单的、有理论依据的单头门控注意力机制，配备（指数）移动平均值，将位置感知局部依赖项的归纳偏差纳入位置无关的注意力机制中

#### 35.Mistral

https://mistral.ai/news/announcing-mistral-7b/

September 27, 2023

是一个基于解码器的 LM，具有以下架构特点：滑动窗口注意力 - 使用 8k 上下文长度和固定缓存大小进行训练，理论注意力跨度为 128K token GQA（分组查询注意）- 允许更快的推理和更低的缓存大小。 字节回退 BPE 标记器 - 确保字符永远不会映射到词汇表标记之外

#### 36.mLUKE

https://arxiv.org/abs/2110.08151

15 Oct 2021

它基于 XLM-RoBERTa 并添加了实体嵌入，这有助于提高涉及实体推理的各种下游任务的性能，例如命名实体识别、提取式问答、关系分类、完形填空式知识完成

#### 37.MPNet

[[2004.09297\] MPNet: Masked and Permuted Pre-training for Language Understanding (arxiv.org)](https://arxiv.org/abs/2004.09297)

20 Apr 2020

一种新颖的预训练方法，继承了 BERT 和 XLNet 的优点并避免了它们的局限性。 MPNet 通过排列语言建模来利用预测标记之间的依赖关系（相对于 BERT 中的 MLM），并以辅助位置信息作为输入，使模型看到完整的句子，从而减少位置差异（相对于 XLNet 中的 PLM）

#### 38.MPT

https://www.mosaicml.com/blog/mpt-7b

May 5, 2023

MPT 模型是 GPT 风格的解码器专用转换器，具有多项改进：性能优化的层实现、提供更高训练稳定性的架构更改，以及通过用 ALiBi 替换位置嵌入来消除上下文长度限制

#### 39.MT5

https://arxiv.org/abs/2010.11934

22 Oct 2020

它是 T5 的多语言变体，在涵盖 101 种语言的新的基于 Common Crawl 的数据集上进行了预训练

#### 40.MVP

https://arxiv.org/abs/2206.12131

24 Jun 2022

MVP 遵循标准 Transformer 编码器-解码器架构， 使用标记数据集进行有监督的预训练，还具有特定于任务的软提示，以激发模型执行特定任务的能力。 MVP是专门为自然语言生成而设计的，可以适应广泛的生成任务，包括但不限于摘要、数据到文本生成、开放式对话系统、故事生成、问答、问题生成、任务生成 面向对话系统、常识生成、释义生成、文本风格迁移和文本简化

#### 41.NEZHA

https://arxiv.org/abs/1909.00204

31 Aug 2019

介绍了在中文语料库上预训练名为 NEZHA（中文语言理解的神经语境表示）的语言模型的实践以及针对中文 NLU 任务的微调。 NEZHA 的当前版本基于 BERT，并进行了一系列经过验证的改进，其中包括作为有效位置编码方案的功能相对位置编码、全字掩码策略、混合精度训练以及训练模型时的 LAMB 优化器

#### 42.NLLB、NLLB-MoE

https://arxiv.org/abs/2207.04672

11 Jul 2022

开发了一种基于稀疏门控混合专家的条件计算模型，该模型根据针对低资源语言量身定制的新颖有效的数据挖掘技术获得的数据进行训练。M2M100ForConditionalGeneration 是 NLLB 和 NLLB MoE 的基础模型

#### 43.Nyströmformer

https://arxiv.org/abs/2102.03902

7 Feb 2021

一种作为序列长度的函数表现出良好的可扩展性的模型。 我们的想法基于采用 Nyström 方法来以 O(n) 复杂度近似标准自注意力。 Nyströmformer 的可扩展性使应用程序能够处理具有数千个令牌的更长序列

#### 44.OPT

https://arxiv.org/abs/2205.01068

2 May 2022

这是一套仅限解码器的预训练 Transformer，参数范围从 125M 到 175B. OPT 与 BartDecoder 具有相同的架构。 与 GPT2 相反，OPT 将 EOS 令牌添加到每个提示的开头。

#### 45.PEGASUS-X

https://arxiv.org/abs/2208.04347

8 Aug 2022

具有全局编码器令牌的交错、块本地 Transformer 在性能和效率之间取得了良好的平衡，并且长序列上的额外预训练阶段有意义地提高了下游汇总性能

#### 46.Persimmon

https://www.adept.ai/blog/persimmon-8b

September 7, 2023

这是一种基于经典 Transformer 架构的解码器模型，具有查询和密钥归一化功能

#### 47.PLBart

https://arxiv.org/abs/2103.06333

10 Mar 2021

是一个类似 BART 的模型, 这是一种序列到序列模型，能够执行广泛的程序和语言理解和生成任务

#### 48.ProphetNet

https://arxiv.org/abs/2001.04063

13 Jan 2020 

提出了一种名为 ProphetNet 的新序列到序列预训练模型，该模型引入了一种名为未来 n 元语法预测的新型自监督目标和所提出的 n 流自注意力机制

#### 49.QDQBert

https://arxiv.org/abs/2004.09602

20 Apr 2020

回顾了量化参数的数学方面，并评估了它们在不同应用领域（包括视觉、语音和语言）的各种神经网络模型上的选择

#### 50.RAG

https://arxiv.org/abs/2005.11401

22 May 2020

探索了一种用于检索增强生成（RAG）的通用微调方法，该模型结合了预先训练的参数和非参数记忆来生成语言。 我们引入 RAG 模型，其中参数存储器是预训练的 seq2seq 模型，非参数存储器是维基百科的密集向量索引，可通过预训练的神经检索器访问

#### 51.REALM

https://arxiv.org/abs/2002.08909

10 Feb 2020

它是一种检索增强语言模型，首先从文本知识语料库中检索文档，然后利用检索到的文档来处理问答任务。

#### 52.Reformer

https://arxiv.org/abs/2001.04451

13 Jan 2020 

介绍两种提高 Transformers 效率的技术。 其一，将点积注意力替换为使用局部敏感哈希的注意力，将其复杂度从 O(L^2) 更改为 O(Llog(L))，其中 L 是序列的长度。 此外，使用可逆残差层而不是标准残差，这允许在训练过程中仅存储激活一次而不是 N 次，其中 N 是层数

#### 53.RemBERT

https://arxiv.org/abs/2010.12821

24 Oct 2020

RemBERT 可以被认为是 mBERT 的更大版本，具有类似 ALBERT 的嵌入层因式分解

#### 54.RoBERTa

https://arxiv.org/abs/1907.11692

26 Jul 2019

该研究仔细测量了许多关键超参数和训练数据大小的影响, RoBERTa 与 BERT 具有相同的架构，但使用字节级 BPE 作为标记器（与 GPT-2 相同）并使用不同的预训练方案。

#### 55.RoCBert

https://aclanthology.org/2022.acl-long.65.pdf

一种预训练的中文 Bert，对各种形式的对抗性攻击（如单词扰动、同义词、拼写错误等）具有鲁棒性。它使用对比学习目标进行预训练，最大限度地提高不同合成对抗性示例下的标签一致性

#### 56.RoFormer

https://arxiv.org/abs/2104.09864

20 Apr 2021

研究了在基于 Transformer 的语言模型中编码位置信息的各种方法，并提出了一种名为旋转位置嵌入（RoPE）的新颖实现。 所提出的 RoPE 使用旋转矩阵对绝对位置信息进行编码，并自然地将显式相对位置依赖性纳入自注意力公式中

#### 57.RWKV

https://www.rwkv.com/

它建议对传统 Transformer 注意力进行调整，使其呈线性. RWKV 是一个具有 Transformer 级别 LLM 性能的 RNN，也可以像 GPT Transformer 一样直接训练（可并行）

#### 58.Splinter

https://arxiv.org/abs/2101.00438

2 Jan 2021 

提出了一种专为问答而设计的新预训练方案：循环跨度选择

#### 59.SqueezeBERT

https://arxiv.org/abs/2006.11316

19 Jun 2020

它是一个类似于 BERT 模型的双向变压器。 BERT 架构和 SqueezeBERT 架构之间的主要区别在于，SqueezeBERT 对于 Q、K、V 和 FFN 层使用分组卷积而不是全连接层。

#### 60.SwitchTransformers

https://arxiv.org/abs/2101.03961

11 Jan 2021

Switch Transformer 模型使用稀疏 T5 编码器-解码器架构，其中 MLP 被专家混合 (MoE) 取代。 路由机制（在本例中为 top 1）将每个令牌与一位专家相关联，其中每个专家都是一个密集的 MLP

#### 61.T5

https://arxiv.org/abs/1910.10683

23 Oct 2019

通过引入一个统一的框架来探索 NLP 迁移学习技术的前景，该框架将每个语言问题转换为文本到文本的格式。 我们的系统研究比较了数十种语言理解任务的预训练目标、架构、未标记数据集、迁移方法和其他因素

#### 62.TAPEX

https://arxiv.org/abs/2107.07653

16 Jul 2021 

提出 TAPEX 来表明表预训练可以通过在合成语料库上学习神经 SQL 执行器来实现，合成语料库是通过自动合成可执行 SQL 查询及其执行输出而获得的

#### 63.Transformer XL

https://arxiv.org/abs/1901.02860

9 Jan 2019

它是一个具有相对定位（正弦）嵌入的因果（单向）转换器，可以重用先前计算的隐藏状态来处理更长的上下文（记忆）。 该模型还使用自适应 softmax 输入和输出（并列）

#### 64.UL2

https://arxiv.org/abs/2205.05131

10 May 2022 

UL2 是一种编码器-解码器模型，经过混合去噪函数的预训练，并针对一系列下游任务进行了微调。 UL2 具有与 T5v1.1 相同的架构，但使用 Gated-SiLU 激活函数而不是 Gated-GELU

#### 65.UMT5

https://openreview.net/forum?id=kXwdL1cWOAi

02 Feb 2023

提出了一种新的采样方法 UniMax，它可以提供更均匀的头语言覆盖，同时通过明确限制每种语言语料库的重复次数来减轻尾语言的过度拟合. UMT5 仅在 mC4 上进行预训练，不包括任何监督训练

#### 66.X-MOD

https://aclanthology.org/2022.naacl-main.255/

X-MOD 扩展了 XLM-R 等多语言掩码语言模型，以在预训练期间包含特定于语言的模块化组件（语言适配器）。 为了进行微调，每个转换器层中的语言适配器都被冻结。

#### 67.XGLM

https://arxiv.org/abs/2112.10668

20 Dec 2021

在涵盖多种语言的平衡语料库上训练多语言自回归语言模型，并研究它们在各种任务中的少样本和零样本学习能力

#### 68.XLM

https://arxiv.org/abs/1901.07291

22 Jan 2019

提出了两种学习跨语言语言模型（XLM）的方法：一种是仅依赖于单语言数据的无监督方法，另一种是利用具有新的跨语言语言模型目标的并行数据的监督方法

#### 69.XLNet

https://arxiv.org/abs/1906.08237

19 Jun 2019

种广义的自回归预训练方法，它 (1) 通过最大化分解顺序的所有排列的预期可能性来学习双向上下文，(2) 由于其自回归克服了 BERT 的局限性公式

#### 70.YOSO

https://arxiv.org/abs/2111.09714

18 Nov 2021

我们展示了基于局部敏感哈希（LSH）的伯努利采样注意机制，将此类模型的二次复杂度降低为线性. 通过将自注意力视为与伯努利随机变量相关的各个令牌的总和来绕过二次成本，原则上可以通过单个哈希立即采样（尽管在实践中，这个数字可能是一个小常数）

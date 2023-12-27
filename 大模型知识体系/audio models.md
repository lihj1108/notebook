### audio models

#### 1.Audio Spectrogram Transformer

https://arxiv.org/abs/2104.01778

5 Apr 2021

音频频谱图转换器将视觉转换器应用于音频，将音频转换为图像（频谱图）

#### 2.Bark

https://github.com/suno-ai/bark

基于转换器的文本到音频模型, Bark可以生成高度逼真的多语言语音以及其他音频 - 包括音乐、背景噪音和简单的音效

#### 3.CLAP

https://arxiv.org/abs/2211.06687

12 Nov 2022

是一种在各种（音频、文本）对上进行训练的神经网络, CLAP 模型使用 SWINTransformer 从 log-Mel 频谱图输入获取音频特征，并使用 RoBERTa 模型获取文本特征。 然后，文本和音频特征都被投影到具有相同维度的潜在空间。 然后将投影的音频和文本特征之间的点积用作相似分数。

#### 4.EnCodec

https://arxiv.org/abs/2210.13438

24 Oct 2022

由一个流式编码器-解码器架构组成，具有以端到端方式训练的量化潜在空间. 引入了一种新颖的损失平衡器机制来稳定训练：损失的权重现在定义了它应代表的总体梯度的分数，从而将该超参数的选择与损失的典型规模脱钩

#### 5.Hubert

https://arxiv.org/abs/2106.07447

14 Jun 2021

提出了用于自监督语音表示学习的隐藏单元 BERT (HuBERT) 方法，该方法利用离线聚类步骤为类似 BERT 的预测损失提供对齐的目标标签. 方法的一个关键要素是仅在屏蔽区域应用预测损失，这迫使模型在连续输入上学习组合的声学和语言模型. HuBERT 主要依赖于无监督聚类步骤的一致性，而不是分配的聚类标签的内在质量

#### 6.MMS

https://arxiv.org/abs/2305.13516

22 May 2023

构建了涵盖 1,406 种语言的预训练 wav2vec 2.0 模型、针对 1,107 种语言的单一多语言自动语音识别模型、针对相同数量语言的语音合成模型以及针对 4,017 种语言的语言识别模型

#### 7.MusicGen

https://arxiv.org/abs/2306.05284

8 Jun 2023

是一种单级自回归 Transformer 模型，能够根据文本描述或音频提示生成高质量的音乐样本. 文本描述通过冻结文本编码器模型以获得一系列隐藏状态表示。 然后，MusicGen 被训练来预测以这些隐藏状态为条件的离散音频标记或音频代码。 然后使用音频压缩模型（例如 EnCodec）对这些音频令牌进行解码，以恢复音频波形

#### 8.Pop2Piano

https://arxiv.org/abs/2211.00895

2 Nov 2022

是一个基于T5的编码器-解码器Transformer模型,  输入音频被转换为其波形并传递到编码器，编码器将其转换为潜在表示。 解码器使用这些潜在表示以自回归方式生成令牌 id。 每个令牌 ID 对应于四种不同令牌类型之一：时间、速度、音符和“特殊”。 然后令牌 ID 被解码为其等效的 MIDI 文件。简单来说，就是将流行音乐转换成钢琴音乐

#### 9.SeamlessM4T

https://dl.fbaipublicfiles.com/seamless/seamless_m4t_paper.pdf

支持以下多任务：Speech-to-speech translation (S2ST)、Speech-to-text translation (S2TT)、Text-to-speech translation (T2ST)、Text-to-text translation (T2TT)、Automatic speech recognition (ASR)

#### 10.SEW

https://arxiv.org/abs/2109.06870

14 Sep 2021

研究了自动语音识别 (ASR) 预训练模型的性能与效率权衡，专注于 wav2vec 2.0

#### 11.Speech2Text

https://arxiv.org/abs/2010.05171

11 Oct 2020

它是一种基于变压器的 seq2seq（编码器-解码器）模型，专为端到端自动语音识别 (ASR) 和语音翻译 (ST) 而设计。在将语音输入输入编码器之前，它使用卷积下采样器将语音输入的长度减少 3/4。 该模型使用标准自回归交叉熵损失进行训练，并自回归生成转录本/翻译。

#### 12.Speech2Text2

https://arxiv.org/abs/2104.06678

14 Apr 2021

Speech2Text2 是仅解码器的转换器模型，可与任何仅语音编码器一起使用，例如用于语音转文本任务的 Wav2Vec2 或 HuBERT

#### 14.SpeechT5

https://arxiv.org/abs/2110.07205

14 Oct 2021

由一个共享的编码器-解码器网络和六个特定于模态（语音/文本）的前/后网络组成。 通过前置网络对输入语音/文本进行预处理后，共享编码器-解码器网络对序列到序列的变换进行建模，然后后置网络根据解码器的输出生成语音/文本模态的输出

#### 15.UniSpeech

https://arxiv.org/abs/2101.07597

19 Jan 2021

提出了一种称为 UniSpeech 的统一预训练方法，用于学习未标记和标记数据的语音表示，其中监督语音 CTC 学习和语音感知对比自监督学习以多任务学习方式进行

#### 16.VITS

https://arxiv.org/abs/2106.06103

11 Jun 2021

是一种端到端语音合成模型，可根据输入文本序列预测语音波形， 它是一种条件变分自动编码器（VAE），由后验编码器、解码器和条件先验组成。基于流的模块预测一组基于声谱图的声学特征，该模块由基于 Transformer 的文本编码器和多个耦合层组成。

#### 17.Wav2Vec2

https://arxiv.org/abs/2006.11477

20 Jun 2020

wav2vec 2.0 屏蔽潜在空间中的语音输入，并解决通过共同学习的潜在表示的量化定义的对比任务

#### 18.WavLM

https://arxiv.org/abs/2110.13900

26 Oct 2021

提出了一种新的预训练模型 WavLM 来解决全栈下游语音任务。 WavLM 基于 HuBERT 框架构建，重点关注语音内容建模和说话者身份保存

#### 19.whisper

https://cdn.openai.com/papers/whisper.pdf

https://github.com/openai/whisper

是一种通用语音识别模型，基于编码器-解码器结构

#### 20.XLS-R

https://arxiv.org/abs/2111.09296

17 Nov 2021

一种基于 wav2vec 2.0 的跨语言语音表示学习的大规模模型，用于语音识别
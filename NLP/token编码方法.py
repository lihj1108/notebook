from datasets import load_dataset
from transformers import AutoTokenizer, BertTokenizerFast, GPT2TokenizerFast, AlbertTokenizerFast
from tokenizers import normalizers, pre_tokenizers, models, processors, decoders, Tokenizer, trainers


# 加载数据集
dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")  
# 定义生成器，不断的从数据集中生成样本语料
batch_size = 1000
def batch_iterator():
    for i in range(0, len(dataset), batch_size):
        yield dataset[i: i + batch_size]["text"]


"""
一.在已有的tokenizer上构建
"""
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # 加载gpt2的tokenizer
new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=25000)  # 将语料加入到tokenizer中，训练新的tokenizer
new_tokenizer.save_pretrained("new-tokenizer")  # 保存新的tokenizer


"""
二.从头自定义构建tokenizer
需要进行以下几步：
1.初始化Tokenizer类，选择tokenize模型（wordpiece、bpe等）
2.标准化(normalization)：对初始输入字符串进行初始转换，例如小写文本、unicode规范编码、去掉字符串两端的空格
3.预分割(pre-tokenization)：分割初始字符串，决定在何处以及如何对原始字符串进行预分段。最简单的例子就是简单地按空格和标点符号分割
4.tokenize模型训练(trainers)：初始化tokenize模型训练器，将语料加入到tokenizer中，训练新的tokenizer
5.后处理(post-processing)：实现构造功能，例如把句子包裹在[CLS],[SEP]中
6.解码(decoding)：将标记化输入映射回原始字符串，解码器通常根据之前使用的Tokenizer类来选择
7.初始化transformers的Tokenizer类
8.保存新的tokenizer
"""

### WordPiece model like Bert
tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]")) # 初始化tokenizer，选择WordPiece的tokenizer方法
tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True) # 初始化Bert normalize方法
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer() # 自定义预分割方法
trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]) # 初始化wordpiece训练器
tokenizer.train_from_iterator(batch_iterator(), trainer=trainer) # 将语料加入到tokenizer中，训练新的tokenizer
tokenizer.post_processor = processors.TemplateProcessing( # 后处理，把单句子或者句子对包裹在[CLS]和[SEP]特殊字符之间
    single=f"[CLS]:0 $A:0 [SEP]:0", 
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1", 
    special_tokens=[("[CLS]", tokenizer.token_to_id("[CLS]")), ("[SEP]", tokenizer.token_to_id("[SEP]"))]
)
tokenizer.decoder = decoders.WordPiece(prefix="##") # 设置解码器
new_tokenizer = BertTokenizerFast(tokenizer_object=tokenizer) # 组装成transformers fast tokenizer
new_tokenizer.save_pretrained("wordpiece-bert-tokenizer") # 保存新的tokenizer


### BPE model like GPT-2
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
trainer = trainers.BpeTrainer(vocab_size=25000, special_tokens=["<|endoftext|>"])
tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
tokenizer.decoder = decoders.ByteLevel()
new_tokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer)
new_tokenizer.save_pretrained("bpe-gpt2-tokenizer") # 保存新的tokenizer


### Unigram model like Albert
tokenizer = Tokenizer(models.Unigram())
tokenizer.normalizer = normalizers.Sequence([normalizers.Replace("``", '"'), normalizers.Replace("''", '"'), normalizers.Lowercase()])
tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
trainer = trainers.UnigramTrainer(vocab_size=25000, special_tokens=["[CLS]", "[SEP]", "<unk>", "<pad>", "[MASK]"], unk_token="<unk>")
tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
tokenizer.post_processor = processors.TemplateProcessing(
    single="[CLS]:0 $A:0 [SEP]:0",
    pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("[CLS]", tokenizer.token_to_id("[CLS]")), ("[SEP]", tokenizer.token_to_id("[SEP]"))]
)
tokenizer.decoder = decoders.Metaspace()
new_tokenizer = AlbertTokenizerFast(tokenizer_object=tokenizer)
new_tokenizer.save_pretrained("unigram-albert-tokenizer") # 保存新的tokenizer

### 编码和解码
encoding = new_tokenizer.encode(text="This is one sentence.", text_pair="With this one we have a pair.") # 编码一个句子对
print("encoding: ", encoding) # 打印编码结果
decoding = new_tokenizer.decode(encoding) # 将token解码成文本
print("decoding: ", decoding)
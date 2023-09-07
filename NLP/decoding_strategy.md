## decoding strategy

以huggingface为例

### 1.greedy search

贪婪搜索，意味着num_beams=1，do_sample=False

生成式模型在生成时默认采用贪婪策略，每次选择概率最大的token作为下一个token，短输出时非常有用，长输出就可能产生高度重复的结果

```python
outputs = model.generate(**inputs, num_beams=1, do_sample=False)
```



### 2.contrastive search

对比搜索，可以通过penalty_alpha和top_k来控制

它展示了生成非重复但连贯的长输出的卓越结果

```python
outputs = model.generate(**inputs, penalty_alpha=0.6, top_k=4, max_new_tokens=100)
```



### 3.Multinomial sampling

多项式采样，num_beams=1，do_sample=True

多项式采样（也称为祖先采样）根据模型给出的整个词汇表的概率分布随机选择下一个标记。 每个具有非零概率的令牌都有被选择的机会，从而降低了重复的风险

```python
outputs = model.generate(**inputs, do_sample=True, num_beams=1, max_new_tokens=100)
```



### 4.Beam-search decoding

波束搜索解码，可以通过将参数num_beams大于1来控制

波束搜索解码在每个时间步保留多个假设，并最终选择整个序列总体概率最高的假设。 这样做的优点是可以识别以较低概率初始标记开始的高概率序列，并且会被贪婪搜索忽略

```python
outputs = model.generate(**inputs, num_beams=5, max_new_tokens=50)
```

### 5.Beam-search multinomial sampling

波束多项式采样，num_beams>1, do_sample=True

联合使用Beam-search和Multinomial sampling

```python
outputs = model.generate(**inputs, num_beams=5, do_sample=True)
```

### 6.Diverse beam search decoding

多样化波束搜索解码，可以通过num_beams，num_beam_groups，diversity_penalty这三个参数来控制

是波束搜索策略的扩展，允许生成更多样化的波束序列集以供选择

```python
outputs = model.generate(**inputs, num_beams=5, num_beam_groups=5, diversity_penalty=1.0, max_new_tokens=30)
```

### 7.Assisted Decoding

辅助解码，

它使用具有相同tokenizer的辅助模型（一般是较主模型小的模型）来贪婪地生成一些候选标记。 然后，主模型在一次前向传递中验证候选标记，从而加快解码过程。 目前辅助解码仅支持贪婪搜索和采样，不支持批量输入。示例代码如下：

```python
prompt = "Alice and Bob"
checkpoint = "EleutherAI/pythia-1.4b-deduped"
assistant_checkpoint = "EleutherAI/pythia-160m-deduped"

tokenizer = AutoTokenizer.from_pretrained(checkpoint) # 共用的tokenizer
model = AutoModelForCausalLM.from_pretrained(checkpoint) # 主模型
assistant_model = AutoModelForCausalLM.from_pretrained(assistant_checkpoint) # 辅助模型

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, assistant_model=assistant_model)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
```


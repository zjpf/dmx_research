from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
import pdb

# 加载下载的模型
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-model")
model = GPT2LMHeadModel.from_pretrained("./gpt2-model")

# 测试推理
text = "The meaning of life is"
input_ids = tokenizer.encode(text, return_tensors="pt")
#outputs = model(**inputs)

print("下载的模型测试成功！")
#print("最后隐藏状态形状:", outputs.last_hidden_state.shape)
# 3. 生成文本 (关键步骤)
output = model.generate(
    input_ids,
    max_length=200,  # 生成文本的最大长度
    num_return_sequences=1,  # 返回的序列数
    no_repeat_ngram_size=2,  # 避免重复的n-gram
    temperature=0.7,  # 控制随机性 (0.0-1.0)
    top_k=50,  # 从概率最高的k个token中选择
    pad_token_id=tokenizer.eos_token_id  # 设置结束标记
)

# 4. 解析输出为文本
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("生成的文本:", generated_text)

# 5. 生成voc保存文件
voc = tokenizer.get_vocab()
with open("voc_gpt2", 'w') as f:
    f.write("{}".format(voc))
inv_voc = {v: k for k, v in voc.items()}
# 解析所有整数和float
ints, floats = [], [] 
for k, v in voc.items():
    try:
        i = int(k)
        ints.append(i)
    except:
        try:
            f = float(k)
            floats.append(f)
        except:
            pass
print(ints)
print(floats)
pdb.set_trace()

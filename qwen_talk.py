from transformers import AutoTokenizer as Tokenizer, AutoModelForCausalLM as Model
import pdb
import sys

# 加载下载的模型
tokenizer = Tokenizer.from_pretrained("qwen3")
model = Model.from_pretrained("qwen3")

print("下载的模型测试成功！")
import io
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
while True:
    text = input("Prompt: ")
    input_ids = tokenizer.encode(text, return_tensors="pt")
    # 3. 生成文本 (关键步骤)
    output = model.generate(
      input_ids,
      max_length=500,  # 生成文本的最大长度
      num_return_sequences=1,  # 返回的序列数
      no_repeat_ngram_size=2,  # 避免重复的n-gram
      temperature=0.7,  # 控制随机性 (0.0-1.0)
      top_k=50,  # 从概率最高的k个token中选择
      pad_token_id=tokenizer.eos_token_id  # 设置结束标记
    )
    # 4. 解析输出为文本
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("生成的文本:", generated_text)

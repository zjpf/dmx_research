from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, GPT2Tokenizer
import pdb
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 仅让程序看到 GPU 0 和 2

dataset = load_dataset("yelp_review_full")
tokenizer = AutoTokenizer.from_pretrained("./gpt2-model")
tokenizer.pad_token = tokenizer.eos_token

#pdb.set_trace()
def tokenize(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def add_labels(example):
    example["labels"] = example["input_ids"]
    return example

dataset = DatasetDict({k: v.select(range(100)) for k, v in dataset.items()})
dataset = dataset.map(tokenize, batched=True, batch_size=8).map(add_labels)

#pdb.set_trace()
import torch
from accelerate import init_empty_weights
from transformers import TrainingArguments, Trainer, GPT2LMHeadModel, AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
model = AutoModelForCausalLM.from_pretrained(
    "./gpt2-model", quantization_config=bnb_config,
    device_map='auto',  # 自动分配设备
    trust_remote_code=True
)#.to("cuda:0")

print("model loaded")
#pdb.set_trace()
from peft import LoraConfig, get_peft_model

# LoRA 配置
peft_config = LoraConfig(
    r=8,                  # LoRA 秩
    lora_alpha=32,        # 缩放因子
    target_modules=["c_proj", "c_fc"],  # 目标模块（Llama/GPT/Mistral）
    lora_dropout=0.05,    # Dropout 率
    bias="none",          # 偏置类型
    task_type="CAUSAL_LM" # 任务类型
)

# 应用 PEFT
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # 查看可训练参数
#model = model.to("cuda:0")          # 整个模型搬到同一设备

training_args = TrainingArguments(
    output_dir="yelp_review_classifier", #no_cuda=True,
    #eval_strategy="epoch",
    push_to_hub=False, per_device_eval_batch_size=8, per_device_train_batch_size=8
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    #compute_metrics=compute_metrics,
)
#pdb.set_trace()
trainer.train()

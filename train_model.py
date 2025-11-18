#训练模型，使用LoRA技术优化大模型训练

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# 1. 加载模型和分词器
model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float16,
    device_map="auto"
)

# 2. 配置LoRA
lora_config = LoraConfig(
    r=64,  # 提升LoRA秩，增强表达能力
    lora_alpha=128,  # 提升缩放系数
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 3. 加载并预处理数据集
dataset = load_dataset("json", data_files="datasets.json", split="train")
dataset = dataset.train_test_split(test_size=0.1, seed=42)

def preprocess_function(examples):
    texts = []
    for instruction, output in zip(examples['instruction'], examples['output']):
        # 优化prompt格式，和推理时保持一致
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}{tokenizer.eos_token}"
        texts.append(text)
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors=None,
        add_special_tokens=True
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_train_dataset = dataset["train"].map(preprocess_function, batched=True)
tokenized_eval_dataset = dataset["test"].map(preprocess_function, batched=True)


# 4. 数据整理器，使用默认data_collator确保padding
from transformers import default_data_collator
data_collator = default_data_collator

# 5. 训练参数
training_args = TrainingArguments(
    output_dir="./deepseek-1.5b-lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=10,  # 增大训练轮数
    learning_rate=5e-5,  # 降低学习率
    warmup_steps=100,
    logging_steps=10,
    eval_steps=200,
    save_steps=500,
    save_strategy="steps",
    # load_best_model_at_end=True,  # 已注释，避免策略不一致报错
    fp16=True,
    remove_unused_columns=False,
    report_to=None,  # 禁用wandb等记录
)

# 6. 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

# 7. 保存
trainer.model.save_pretrained("./deepseek-1.5b-lora-final")
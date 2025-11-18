
#测试训练后的模型，基础模型+LoRA练后的模型

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# 1. 加载基础模型和分词器
base_model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
lora_model_path = "deepseek-1.5b-lora-final"

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.pad_token = tokenizer.eos_token

# 2. 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    dtype=torch.float16,
    device_map="auto"
)

# 3. 加载LoRA权重
model = PeftModel.from_pretrained(base_model, lora_model_path)
model = model.merge_and_unload()  # 合并LoRA权重到主模型（可选）

# 4. 推理测试
prompt = "iNeuOS工业互联网？\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.95,
        temperature=0.8
    )
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("模型输出：")
print(result)

#合并LoRA权重到基础模型

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

base_model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
lora_model_path = "deepseek-1.5b-lora-final"
merged_model_path = "deepseek-1.5b-lora-merged"

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, lora_model_path)
model = model.merge_and_unload()  # 合并LoRA权重

# 保存合并后的全量模型
model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)
#测试基础模型DeepSeek-R1-Distill-Qwen-1.5B

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 加载模型和tokenizer
model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # 替换为实际路径

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float16,  # 半精度推理节省内存
    device_map="auto"  # 自动分配设备
)

# 生成文本
def generate_text(prompt, max_length=10000):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 使用示例
prompt = "iNeuOS工业互联网操作系统的功能介绍?\n"
print("提问："+prompt)
result = generate_text(prompt)
print("回答："+result)
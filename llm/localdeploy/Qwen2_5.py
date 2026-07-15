import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型和分词器
#model_path = "D:\PycharmWork\demo\llm\localdeploy\Qwen\Qwen2.5-0.5B-Instruct"
model_path = os.path.join(os.path.dirname(__file__), "Qwen", "Qwen2.5-0.5B-Instruct")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# 将模型移动到设备上
model.to(device)

# 准备输入数据
messages = [
    {"role": "system", "content": "你是一个乐于助人的助手，回答简洁准确。"},
    {"role": "user", "content": "用一句话介绍深度学习。"}
]

# 应用聊天模板并生成输入
"""
聊天模板如下：
system
你是一个乐于助人的助手。
user
用一句话介绍深度学习。
assistant```

"""
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer([text], return_tensors="pt").to(device)
print(f"聊天模板文本: {text}")
print(f"输入张量: {inputs}")

# 生成文本
generated_ids = model.generate(
    inputs.input_ids,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,     # 启用采样
    repetition_penalty=1.1,
    pad_token_id=tokenizer.eos_token_id     # 设置填充token为结束token
)

# 只保留新生成的部分
output_ids = generated_ids[0][inputs.input_ids.shape[1]:]
response = tokenizer.decode(output_ids, skip_special_tokens=True)
print(f"模型输出: {response}")

from modelscope import AutoModelForCausalLM, AutoTokenizer

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained('./gpt2')
tokenizer = AutoTokenizer.from_pretrained('./gpt2')

# 将模型设置为推理模式
model.eval()

# 准备输入
input_text = "The future of AI is"
inputs = tokenizer(input_text, return_tensors="pt")

# 生成文本
outputs = model.generate(**inputs, max_length=100, do_sample=True, top_p=0.95, temperature=0.8)

# 解码并打印结果
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
from openai import OpenAI

client = OpenAI(
    api_key="sk-qcpskosxzjnjmewklbsemkxntjrxeuldojdmpjbbnsvervfc",
    base_url="https://api.siliconflow.cn/v1"  # 指向第三方平台
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "用一句话介绍 Python"}],
    max_tokens=150,
    temperature=0.7
)

print(response.choices[0].message.content)
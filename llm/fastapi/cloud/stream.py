from openai import OpenAI

client = OpenAI(
    api_key="sk-qcpskosxzjnjmewklbsemkxntjrxeuldojdmpjbbnsvervfc",
    base_url="https://api.siliconflow.cn/v1"  # 指向第三方平台
)


stream = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "讲个程序员笑话"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
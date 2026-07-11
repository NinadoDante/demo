from openai import OpenAI

client = OpenAI(
    # API密钥
    api_key="sk-talrfpdubittuoctscpxqyuhotkkqgcmuxrxxlmmhkqpzlxd",
    # API暴露出来的HTTP网址（接口）
    base_url="https://api.siliconflow.cn/v1"
)

# 1、初始化一个空列表 用于存储聊天记录
messages = [{"role": "system", "content": "你是一个有用的助手"}]

# 2、开始循环对话
while True:
    # 3、用户输入 并 判断
    user_input = input("用户：")
    if user_input == "退出":
        break
    # 4、将用户 问 存起来
    messages.append({"role": "user", "content": user_input})
    # 5、发起请求，让LLM回复
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=messages
    )

    # 6、解析并输出结果
    ai_reply = response.choices[0].message.content
    print(f"AI助手：{ai_reply}")

    # 7、将 AI 的 答 存起来
    messages.append({"role": "assistant", "content": ai_reply})
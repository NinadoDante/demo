import gradio as gr
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY", "sk-talrfpdubittuoctscpxqyuhotkkqgcmuxrxxlmmhkqpzlxd"),
    base_url="https://api.siliconflow.cn/v1"
)


def chat_cloud(message, history, sys_prompt, temperature):
    # 构造消息列表
    messages = [{"role": "system", "content": sys_prompt}]

    for msg in history:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            messages.append({
                "role": str(msg["role"]),
                "content": str(msg["content"])
            })

    messages.append({"role": "user", "content": str(message)})

    try:
        stream = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=messages,
            temperature=temperature,
            stream=True
        )

        partial = ""
        for chunk in stream:
            # choices 和 delta 是否存在
            if chunk.choices and chunk.choices[0].delta:
                delta_content = chunk.choices[0].delta.content
                if delta_content:
                    partial += delta_content
                    yield partial

    except Exception as e:
        yield f"⚠️ API 调用失败: {str(e)}"


# 界面配置
demo = gr.ChatInterface(
    fn=chat_cloud,
    title="🌐 云端大模型对话",
    description="基于 SiliconFlow 部署的 Qwen2.5-7B-Instruct 模型 (Gradio 6.18.0)",
    additional_inputs=[
        gr.Textbox(
            label="系统提示词",
            value="你是一个乐于助人的助手。",
            lines=3
        ),
        gr.Slider(minimum=0.1, maximum=1.5, value=0.3, step=0.1, label="Temperature")
    ]
)

demo.launch()
import gradio as gr
import requests


def chat_ollama(message, history):
    messages = []

    # 清洗历史消息，确保只包含 role 和 content
    for msg in history:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            messages.append({
                "role": str(msg["role"]),
                "content": str(msg["content"])
            })

    # 添加当前用户消息
    messages.append({"role": "user", "content": str(message)})

    try:
        resp = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "qwen2.5:0.5b",
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": 512}
            },
            timeout=120
        )

        # 检查HTTP状态码
        if resp.status_code != 200:
            error_detail = resp.text[:500]
            print(f"[Ollama Error] Status {resp.status_code}: {error_detail}")
            return f"⚠️ Ollama 返回错误 (HTTP {resp.status_code}): {error_detail}"

        data = resp.json()

        # 安全获取响应内容
        if "message" not in data or "content" not in data["message"]:
            print(f"[Ollama Unexpected Response]: {data}")
            return f"⚠️ Ollama 响应格式异常: {str(data)[:300]}"

        return data["message"]["content"]

    except requests.exceptions.RequestException as e:
        print(f"[Connection Error]: {e}")
        return f"⚠️ 无法连接 Ollama 服务: {str(e)}"
    except Exception as e:
        print(f"[Unexpected Error]: {e}")
        return f"⚠️ 发生未知错误: {str(e)}"


demo = gr.ChatInterface(
    fn=chat_ollama,
    title="🤖 Qwen2.5 助手 (Ollama)",
    description="基于本地 Ollama 部署的 Qwen2.5-0.5B 模型"
)

demo.launch()
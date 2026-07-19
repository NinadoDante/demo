import gradio as gr
import requests

def chat_with_backend(message, history):
    resp = requests.post("http://localhost:8000/chat",
                         json={"message": message, "history": history})
    return resp.json()["reply"] if resp.status_code == 200 else f"错误：{resp.status_code}"

demo = gr.ChatInterface(fn=chat_with_backend, title="前后端分离演示")
demo.launch()
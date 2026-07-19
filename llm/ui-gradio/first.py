import gradio as gr

def greet(name):
    return f"Hello {name}!"

demo = gr.Interface(
    fn=greet,           # 要包装的函数
    inputs="text",      # 输入组件类型（简写）
    outputs="text"      # 输出组件类型（简写）
)

demo.launch()
import os
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("API_KEY", "sk-qcpskosxzjnjmewklbsemkxntjrxeuldojdmpjbbnsvervfc"),
    base_url="https://api.siliconflow.cn/v1"
)
app = FastAPI(title="大模型对话 API", version="1.0")
class TranslateRequest(BaseModel):
    text: str
    target_lang: str = "英文"

@app.post("/translate")
async def translate(request: TranslateRequest):
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{
            "role": "user",
            "content": f"将以下文本翻译为{request.target_lang}：{request.text}"
        }],
        temperature=0.3,
        max_tokens=512
    )
    return {"translation": response.choices[0].message.content}
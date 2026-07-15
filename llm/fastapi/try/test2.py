from fastapi.responses import StreamingResponse
import json
import os
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

from llm.fastapi.full import ChatRequest

client = OpenAI(
    api_key=os.getenv("API_KEY", "sk-qcpskosxzjnjmewklbsemkxntjrxeuldojdmpjbbnsvervfc"),
    base_url="https://api.siliconflow.cn/v1"
)
app = FastAPI(title="大模型对话 API", version="1.0")
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    def generate():
        stream = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.message}
            ],
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    return StreamingResponse(generate(), media_type="text/plain")
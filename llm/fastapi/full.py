import os
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="大模型对话 API", version="1.0")

client = OpenAI(
    api_key=os.getenv("API_KEY", "sk-qcpskosxzjnjmewklbsemkxntjrxeuldojdmpjbbnsvervfc"),
    base_url="https://api.siliconflow.cn/v1"
)

class ChatRequest(BaseModel):
    message: str
    system_prompt: str = "你是一个乐于助人的助手。"
    temperature: float = 0.7
    max_tokens: int = 512

class ChatResponse(BaseModel):
    reply: str
    model: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.message}
        ],
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )
    return ChatResponse(
        reply=response.choices[0].message.content,
        model=response.model
    )

@app.get("/health")
async def health():
    return {"status": "healthy"}
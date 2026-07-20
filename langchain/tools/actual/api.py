from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse
from llm_layer import stream_llm_response

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "欢迎使用三工具智能体！访问 /chat 开始对话"}

@app.post("/chat")
async def chat(query: str = Body(..., embed=True)):
    # 直接返回流式响应，mime 类型保持 text/plain
    return StreamingResponse(stream_llm_response(query), media_type="text/plain")
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os
from typing import List, Dict, Any

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 从环境变量读取 API Key（更安全）
client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY", "sk-talrfpdubittuoctscpxqyuhotkkqgcmuxrxxlmmhkqpzlxd"),  # 请替换为实际环境变量
    base_url="https://api.siliconflow.cn/v1"
)


class ChatReq(BaseModel):
    message: str
    history: List[Dict[str, str]] = []  # 明确指定为字典列表


@app.post("/chat")
async def chat(req: ChatReq):
    try:
        # 构建消息列表，先添加系统提示
        messages = [{"role": "system", "content": "你是一个乐于助人的助手。"}]

        # 兼容两种历史格式：
        # 1. 如果是字典列表（推荐），直接追加
        # 2. 如果是旧格式元组列表，尝试转换（但建议前端统一为字典）
        for msg in req.history:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            elif isinstance(msg, (list, tuple)) and len(msg) == 2:
                # 兼容旧式 (user, bot) 元组，但建议前端不要使用
                messages.append({"role": "user", "content": msg[0]})
                messages.append({"role": "assistant", "content": msg[1]})
            else:
                # 如果格式不识别，跳过或忽略
                continue

        # 添加当前用户消息
        messages.append({"role": "user", "content": req.message})

        # 调用 API（非流式）
        resp = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=messages,
            temperature=0.7
        )
        reply = resp.choices[0].message.content
        return {"reply": reply}

    except Exception as e:
        # 捕获异常，返回友好错误信息
        raise HTTPException(status_code=500, detail=f"调用模型失败: {str(e)}")


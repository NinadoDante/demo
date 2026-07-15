import requests

def chat(prompt, model="qwen2.5:0.5b"):
    resp = requests.post("http://localhost:11434/api/chat", json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.7, "top_p": 0.9}
    })
    return resp.json()["message"]["content"]

print(chat("用一句话介绍 Python"))
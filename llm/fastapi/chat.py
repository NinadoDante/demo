import requests

resp = requests.post("http://127.0.0.1:8000/chat", json={
    "message": "用一句话介绍深度学习",
    "temperature": 0.5
})
print(resp.json()["reply"])
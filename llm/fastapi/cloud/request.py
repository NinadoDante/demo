import requests

url = "https://api.siliconflow.cn/v1/chat/completions"
headers = {
    "Authorization": f"Bearer sk-qcpskosxzjnjmewklbsemkxntjrxeuldojdmpjbbnsvervfc",
    "Content-Type": "application/json"
}
data = {
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [
        {"role": "system", "content": "你是一个乐于助人的助手。"},
        {"role": "user", "content": "用一句话介绍 Python"}
    ],
    "max_tokens": 150,
    "temperature": 0.7
}

response = requests.post(url, headers=headers, json=data)
print(response)
# print(response.text)
if response.status_code == 200:
    answer = response.json()["choices"][0]["message"]["content"]
    print(f"回答: {answer}")
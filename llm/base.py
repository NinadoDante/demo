import tiktoken

enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 使用的编码
tokens = enc.encode("unhappy")
print(f"Token IDs: {tokens}")
print(f"Tokens: {[enc.decode([t]) for t in tokens]}")

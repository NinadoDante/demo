from fastapi import FastAPI

app = FastAPI(title="我的 API", version="1.0")

@app.get("/")
async def root():
    return {"message": "Hello, FastAPI!"}

@app.get("/greet/{name}")
async def greet(name: str, age: int = 18):
    return {"message": f"你好 {name}！", "age": age}

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    return {"user_id": user_id}

# /users/123 → {"user_id": 123}
# /users/abc → 422 错误（类型校验失败）

@app.get("/search")
async def search(keyword: str, page: int = 1, page_size: int = 20):
    return {"keyword": keyword, "page": page, "page_size": page_size}

# /search?keyword=python&page=2

from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float
    quantity: int = 1

@app.post("/items/")
async def create_item(item: Item):
    total = item.price * item.quantity
    return {"item": item, "total_price": total}
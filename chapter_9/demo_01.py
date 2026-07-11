import asyncio

# 定义一个协程函数
async def hello():
    print("Hello")
    await asyncio.sleep(0.1) # 模拟耗时操作，交出控制权
    print("World")

# # 直接调用不会执行
# h = hello()
# print(type(h)) # <class 'coroutine'>

# 必须用 asyncio.run() 来启动事件循环并执行
asyncio.run(hello())
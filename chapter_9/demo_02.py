import asyncio
import time

async def say_after(delay, what):
    # asyncio.sleep 是一个协程，会在这里交出控制权
    await asyncio.sleep(delay)
    print(f"{time.strftime('%X')}: {what}")

async def main():
    print(f"开始 {time.strftime('%X')}")
    # 这里虽然写了两个await，但它们是顺序执行的，总耗时 2+1=3秒
    await say_after(2, 'world')
    await say_after(1, 'hello')
    print(f"结束 {time.strftime('%X')}")

asyncio.run(main())
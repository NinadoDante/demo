import asyncio
import time

async def worker(name, delay):
    print(f"{name} 开始")
    await asyncio.sleep(delay)
    print(f"{name} 完成")
    return f"结果_{name}"

async def main():
    # 并发执行worker("A")和worker("B")，总耗时取最长的时间（2秒）
    results = await asyncio.gather(
        worker("A", 2),
        worker("B", 1),
        worker("C", 1.5)
    )
    print(results)  # ['结果_A', '结果_B']

print(f"开始 {time.strftime('%X')}")
asyncio.run(main())
print(f"结束 {time.strftime('%X')}")
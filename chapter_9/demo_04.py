import asyncio
import time
import datetime
async def worker(name, delay):
    print(f"{name} 开始")
    await asyncio.sleep(delay)
    print(f"{name} 完成")
    return f"结果_{name}"

async def main():
    # 将协程包装成任务，立即开始调度
    task_a = asyncio.create_task(worker("A", 2))
    task_b = asyncio.create_task(worker("B", 1))
    task_c = asyncio.create_task(worker("C", 1.5))
    
    # 在等待A和B的同时，主协程可以继续做其他事
    print("主协程在干别的事...")
    
    # 最后再等待任务完成并获取结果
    result_a = await task_a
    result_b = await task_b
    result_c = await task_c
    print(result_a, result_b, result_c)

print(f"开始 {time.strftime('%X')}")
asyncio.run(main())
print(f"结束 {time.strftime('%X')}")
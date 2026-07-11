import asyncio

async def slow_task():
    await asyncio.sleep(5)
    return "慢任务完成"

async def main():
    try:
        # result = await slow_task()
        result = await asyncio.wait_for(slow_task(), timeout=2)
        print(result)
    except asyncio.TimeoutError:
        print("任务超时，已取消！")

asyncio.run(main())
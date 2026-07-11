import asyncio
import time
# 创建一个信号量，最多允许2个协程同时运行
sem = asyncio.Semaphore(2)

async def limited_task(name):
    # async with sem 会在进入时获取令牌，退出时归还令牌
    async with sem:
        print(f"{name} 获得许可，开始执行")
        await asyncio.sleep(2)
        print(f"{name} 执行完毕，归还许可")

async def main():
    # 同时启动5个任务，但因为信号量限制，同时最多只有2个在执行
    await asyncio.gather(
        limited_task("任务1"),
        limited_task("任务2"),
        limited_task("任务3"),
        limited_task("任务4"),
        limited_task("任务5")
    )

print(f"开始 {time.strftime('%X')}")
asyncio.run(main())
print(f"结束 {time.strftime('%X')}")
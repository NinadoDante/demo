import asyncio
import aiohttp
import time

# 异步请求函数，接受session、url和信号量
async def fetch(session, url, sem):
    async with sem: # 使用信号量限制并发
        try:
            # 发起异步GET请求
            async with session.get(url, timeout=5) as resp:
                text = await resp.text()
                print(f"✅ {url} 状态: {resp.status}, 长度: {len(text)}")
                return len(text)
        except Exception as e:
            print(f"❌ {url} 异常: {e}")
            return 0

async def main():
    urls = [
        "https://httpbin.org/delay/1",
        "https://httpbin.org/delay/2",
        "https://httpbin.org/json",
        "https://www6.baidu.com/"
    ]
    # 限制最多3个并发请求
    sem = asyncio.Semaphore(3)

    start = time.time()
    # 创建一个客户端会话（session），可复用底层TCP连接，性能更高
    async with aiohttp.ClientSession() as session:
        # 为每个URL创建一个fetch协程
        tasks = [fetch(session, url, sem) for url in urls]
        # 并发执行所有任务
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    print(f"总耗时: {time.time() - start:.2f} 秒")
    print("各页面大小:", results)

asyncio.run(main())
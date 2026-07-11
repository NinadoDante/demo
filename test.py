import asyncio
import aiohttp

async def fetch_status(session, url, sem):
    async with sem:
        async with session.get(url) as resp:
            return url, resp.status

async def batch_download(urls, max_concurrent=2):
    sem = asyncio.Semaphore(max_concurrent)
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_status(session, url, sem) for url in urls]
        results = await asyncio.gather(*tasks)
        for url, status in results:
            print(f"{url} -> 状态码: {status}")

# 测试
urls = ["https://httpbin.org/status/200"] * 4
asyncio.run(batch_download(urls, max_concurrent=2))
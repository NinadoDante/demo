import time
import numpy as np

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        cost = (time.time() - start) * 1000
        print(f"函数 {func.__name__} 耗时: {cost:.4f} ms")
        return ret
    return wrapper

@timer
def list_square(lst):
    return [x ** 2 for x in lst]

@timer
def ndarray_square(arr):
    return arr ** 2

data = list(range(10**6))
print(type(data))
arr = np.array(data)
print(type(arr))

list_square(data)       # 约 250 ms
ndarray_square(arr)     # 约 2 ms
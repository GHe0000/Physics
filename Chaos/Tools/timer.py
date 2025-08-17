# 简单的函数计时器

import time
from functools import wraps 

def Timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} : {end_time - start_time:.5f} s")
        return result
    return wrapper

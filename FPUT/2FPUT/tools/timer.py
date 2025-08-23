import time
from functools import wraps 

def FunctionTimer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} : {end_time - start_time:.5f} s")
        return result
    return wrapper

class LapTimer:
    def __init__(self):
        self.time = None
    def __call__(self):
        if self.time is None:
            self.time = time.time()
            return 0
        else:
            dt = time.time() - self.time 
            self.time = time.time()
            return dt

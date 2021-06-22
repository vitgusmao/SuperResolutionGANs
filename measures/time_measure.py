import time
from contextlib import contextmanager


@contextmanager
def time_context(name):
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    print('[{}] terminou em {} ms'.format(name, int(elapsed_time * 1000)))
from contextlib import contextmanager
import time


@contextmanager
def timeit_context(name):
    startTime = time.time()
    yield
    elapsedTime = time.time() - startTime
    print('[{}] terminou em {} ms'.format(name, int(elapsedTime * 1000)))
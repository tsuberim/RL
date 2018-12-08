def take(n, it):
    for _ in range(n):
        yield next(it)


def last(it):
    for val in it:
        pass
    return val


def do(f, it):
    for i, val in enumerate(it):
        f(val)
        yield val

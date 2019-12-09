import time

class AverageMeter(object):
    def __init__(self):
        self.clear()

    def clear(self):
        self.sum = 0
        self.n = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.n += n

    @property
    def avg(self):
        return self.sum / self.n


class StopwatchMeter(object):
    """Computes the sum/avg duration of some event in seconds"""
    def __init__(self):
        self.clear()

    def clear(self):
        self.sum = 0
        self.n = 0
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self, n=1):
        if self.start_time is not None:
            delta = time.time() - self.start_time
            self.sum += delta
            self.n += n
            self.start_time = None

    @property
    def avg(self):
        return self.sum / self.n

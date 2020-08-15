
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

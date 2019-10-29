'''A wrapper class for optimizer '''

class LRScheduler(object):

    def __init__(self, optimizer, lr):
        self._optimizer = optimizer
        self.lr = lr
        self.set_lr()

    def step(self):
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def set_lr(self):
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = self.lr


class InverseSqrtScheduler(LRScheduler):

    def __init__(self, optimizer, peak_lr, warmup_steps):
        super().__init__(optimizer, 0)

        self.warmup_steps = warmup_steps
        self.current_step = 0
        # linearly warmup for the first warmup_steps
        self.warmup_factor = peak_lr / warmup_steps
        # then, decay prop. to the inverse square root of the step number
        self.decay_factor = peak_lr * warmup_steps**0.5

    def step(self):
        self._update_learning_rate()
        super().step()

    def _update_learning_rate(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            self.lr = self.warmup_factor * self.current_step
        else:
            self.lr = self.decay_factor * self.current_step**-0.5
        self.set_lr()


class LinearDecayScheduler(LRScheduler):

    def __init__(self, optimizer, peak_lr, warmup_steps, total_steps):
        super().__init__(optimizer, 0)

        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0
        # linearly warmup for the first warmup_steps
        self.warmup_factor = peak_lr / warmup_steps
        # then, linearly decay to 0
        self.decay_factor = peak_lr / (total_steps - warmup_steps)

    def step(self):
        self._update_learning_rate()
        super().step()

    def _update_learning_rate(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            self.lr = self.warmup_factor * self.current_step
        elif self.current_step < self.total_steps:
            self.lr = self.decay_factor * (self.total_steps - self.current_step)
        else:
            self.lr = 0
        self.set_lr()

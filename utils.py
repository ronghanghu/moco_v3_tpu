from collections import deque

import numpy as np


class AttrDict(dict):
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            self[name] = value


# adapted from
# https://github.com/facebookresearch/mmf/blob/master/mmf/common/meter.py
class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.window_size = window_size
        self.reset()

    def reset(self):
        self.deque = deque(maxlen=self.window_size)
        self.averaged_value_deque = deque(maxlen=self.window_size)
        self.batch_sizes = deque(maxlen=self.window_size)
        self.total_samples = 0
        self.total = 0.0
        self.count = 0

    def update(self, value, batch_size):
        self.deque.append(value * batch_size)
        self.averaged_value_deque.append(value)
        self.batch_sizes.append(batch_size)

        self.count += 1
        self.total_samples += batch_size
        self.total += value * batch_size

    @property
    def median(self):
        d = np.median(list(self.averaged_value_deque))
        return d

    @property
    def avg(self):
        d = np.sum(list(self.deque))
        s = np.sum(list(self.batch_sizes))
        return d / s

    @property
    def global_avg(self):
        return self.total / self.total_samples

    def get_latest(self):
        return self.averaged_value_deque[-1]

import numpy as np

class OUActionNoise(object):
    def __init__(self, mean, std_deviation = 0.15, theta = 2, dt = 1e-2, x0 = None):
        self.theta = theta
        self.mean = mean
        self.std_deviation = std_deviation
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt + \
            self.std_deviation * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mean)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mean, self.std_deviation)

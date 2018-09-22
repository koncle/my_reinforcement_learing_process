import numpy as np

class Memory:
    def __init__(self, batch_size, max_memory_size):
        self.exp = None
        self.batch_size = batch_size
        self.max_size = max_memory_size
        self.iter = 0

    def add(self, experience):
        if self.exp is None:
            self.exp = np.expand_dims(np.array(experience), axis=0)
        else:
            if self.exp.shape[0] >= self.max_size:
                self.exp[self.iter] = experience
                self.iter = (self.iter + 1) % self.max_size
            else:
                self.exp = np.vstack([self.exp, experience])

    def sample(self):
        idx = np.array(range(len(self.exp)))
        a = np.array(self.exp)
        return a[np.random.choice(idx, self.batch_size)]

    def is_enough(self):
        return self.exp.shape[0] >= self.batch_size

    def get_array_from(self, exp):
        states = []
        for state in exp:
            states.append(state)
        return np.array(states)


def discretize_action(action, low, high, num):
    if action == low:
        return 0
    if action == high:
        return num
    return int((action - low) / (high - low) * (num - 1.))


def de_discretize_action(descrete_action, low, high, num):
    return descrete_action / num * (high - low) + low


class RewardQueue:
    def __init__(self, max_size):
        self.last_total_rewards = []
        self.max_size = max_size

    def add_reward(self, reward):
        self.last_total_rewards.append(reward)
        if len(self.last_total_rewards) > self.max_size:
            self.last_total_rewards.pop(0)

    def get_average(self):
        return np.mean(self.last_total_rewards)

    def is_enough(self):
        if len(self.last_total_rewards) == self.max_size:
            return True
        else:
            return False


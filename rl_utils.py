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


def choose_action(action_probability):
    rand = np.random.uniform()
    sum = 0
    for i in range(len(action_probability)):
        prob = action_probability[i]
        if sum <= rand < (prob + sum):
            return i
        else:
            sum += prob
    if sum == 1:
        return len(action_probability) - 1


def random_action(action_n):
    interval = 1.0 / action_n
    action_probability = [interval for i in range(action_n)]
    idx = choose_action(action_probability)
    return idx

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


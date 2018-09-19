import gym
import tensorflow as tf
import numpy as np
from tensorboardX import SummaryWriter

def run_episode(env, parameter):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = 0 if np.matmul(parameter, np.transpose(state)) < 0 else 1
        next_state, reward, done, info = env.step(action)
        env.render()
        state = next_state
        total_reward += reward
    print('total reward: ', total_reward)
    return total_reward


# search by your luck
def HillCliming():
    env = gym.make('CartPole-v0')
    best_reward = 0
    best_param = np.random.rand(4) * 2 - 1
    noise_scale = 0.05
    for episode in range(200):
        cur_w = best_param + (np.random.rand(4) * 2 - 1) * noise_scale
        total_reward = run_episode(env, best_param)
        if total_reward > best_reward:
            best_param = cur_w
            best_reward = total_reward
    print('best reward ', best_param, 'best parameters ', best_param)
    env.close()


def RandomSearch():
    env = gym.make('CartPole-v0')
    best_reward = 0
    best_param = np.random.rand(4) * 2 - 1
    for episode in range(200):
        cur_w = np.random.rand(4) * 2 - 1
        total_reward = run_episode(env, cur_w)
        if total_reward > best_reward:
            best_param = cur_w
            best_reward = total_reward
    print('best reward ', best_reward, 'best parameters ', best_param)
    env.close()


class DQN:
    def __init__(self):
        self.update_count = 0
        self.update_freq = 4
        self.switch_freq = 16

        self.features = 4
        self.actions = 2
        self.gamma = 0.8

        self.start_eps = 0.5
        self.end_eps = 0.1
        self.eps_drop = (self.start_eps - self.end_eps) / 100
        self.epsilon = self.start_eps
        self.batch_size = 8

        self.memory = Memory(self.batch_size)

        self.state = tf.placeholder(dtype=tf.float32, shape=(None, self.features), name='state')
        self.action = tf.placeholder(dtype=tf.int32, shape=(None,), name='actions')
        self.target_q = tf.placeholder(dtype=tf.float32, shape=(None,), name='target_q')

        self.eval_net, t_params = self._get_net('eval_net')
        self.target_net, e_params = self._get_net('target_net')

        self.eval_max_value = tf.reduce_max(self.eval_net, axis=1)
        self.target_max_value = tf.reduce_max(self.target_net, axis=1)

        self.predict_action = tf.argmax(self.eval_net)

        # Q(s, a) = Q(s, a) - alpha * (R + gamma * max Q(s', a') - Q(s, a))
        # optimize : (Q(s, a) - TD_target) ** 2

        mask_to_update = tf.one_hot(self.action, self.actions)
        action_value = tf.reshape(tf.reduce_sum(self.eval_net * mask_to_update, axis=1, keepdims=True), [-1])
        self.loss = tf.losses.mean_squared_error(self.target_q, action_value)
        self.opt = tf.train.AdamOptimizer(learning_rate=0.1).minimize(self.loss)

        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter("D:\\NJU\\Games\\log", self.sess.graph_def)

        tf.summary.scalar('loss', self.loss)
        tf.summary.histogram(e_params[-1].name,
                             e_params[-1]),
        tf.summary.histogram(e_params[-2].name,
                             e_params[-2]),
    
        tf.summary.histogram(t_params[-1].name,
                             t_params[-1]),
        tf.summary.histogram(t_params[-2].name,
                             t_params[-2]),
        self.summaries = tf.summary.merge_all()

    def _get_net(self, scope):
        with tf.variable_scope(scope):
            Z = tf.contrib.layers.fully_connected(self.state, 30)
            net = tf.contrib.layers.fully_connected(Z, self.actions, activation_fn=None)
        return net, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

    def _get_states_from(self, exp):
        states = []
        for state in exp:
            states.append(state)
        return np.array(states)

    def update(self):
        if not self.memory.is_enough():
            return

        self.update_count += 1

        # decreatse epsilon
        if self.epsilon > self.end_eps:
            self.epsilon -= self.eps_drop

        # update
        if self.update_count % self.update_freq == 0:
            print('update')
            # [state, action, next_state, reward]
            experience = self.memory.sample()
            states = self._get_states_from(experience[:, 0])
            next_states = self._get_states_from(experience[:, 2])
            actions = experience[:, 1]

            # next_state's value = Q(s', a')
            target_value = self.sess.run(self.target_max_value,
                                         feed_dict={self.state: next_states})

            # TD_target = reward + gamma * max_a'(Q(s', a')) , [batch_size, 1]
            TD_target = experience[:, 3] + self.gamma * target_value

            summary, _ = self.sess.run([self.summaries, self.opt],
                                 feed_dict={self.state: states, self.action: actions, self.target_q: TD_target})
            self.writer.add_summary(summary, self.update_count)

        if self.update_count % self.switch_freq == 0:
            print('switch')
            self.sess.run(self.replace_target_op)

    def get_action(self, state, rand=True):
        # epsilon greedy
        if rand:
            if np.random.uniform() < self.epsilon:
                return 0 if np.random.uniform() > 0 else 1
        # eval action
        return np.reshape(self.sess.run(self.predict_action, feed_dict={self.state : np.reshape(state, (1, 4))}), (-1))[0]

    def add_memory(self, exp):
        self.memory.add(exp)

    def close(self):
        self.sess.close()
        self.writer.close()


class Memory:
    def __init__(self, batch_size):
        self.exp = None
        self.batch_size = batch_size
        self.max_size = 10000
        self.iter = 0

    def add(self, episode):
        if self.exp is None:
            self.exp = np.expand_dims(np.array(episode), axis=0)
        else:
            if self.exp.shape[0] >= self.max_size:
                self.exp[self.iter] = episode
                self.iter = (self.iter + 1) % self.max_size
            else:
                self.exp = np.vstack([self.exp, episode])

    def sample(self):
        idx = np.array(range(len(self.exp)))
        array = np.array(self.exp)
        return array[np.random.choice(idx, self.batch_size)]

    def is_enough(self):
        return self.exp.shape[0] >= self.batch_size


def DQNSearch():
    env = gym.make('CartPole-v0')
    model = DQN()
    for _ in range(1000):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = model.get_action(state)
            next_state, reward, done, info = env.step(action)
            env.render()

            model.add_memory([state, action, next_state, reward])
            model.update()

            state = next_state
            total_reward += reward
        print('total reward: ', total_reward)
    env.close()
    model.close()


if __name__ == '__main__':
    DQNSearch()

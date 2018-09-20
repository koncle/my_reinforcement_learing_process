import tensorflow as tf
import numpy as np

class DQN:
    def __init__(self):
        self.update_count = 0
        self.update_freq = 4
        self.switch_freq = 200

        self.feature_n = 4
        self.action_n = 2
        self.gamma = 0.8

        self.start_eps = 0.9
        self.end_eps = 0.1
        self.eps_drop = (self.start_eps - self.end_eps) / 1000
        self.epsilon = self.start_eps
        self.batch_size = 64

        self.memory = Memory(self.batch_size)

        self.state = tf.placeholder(dtype=tf.float32, shape=(None, self.feature_n), name='state')
        self.action = tf.placeholder(dtype=tf.float32, shape=(None, self.action_n), name='actions')
        self.reward = tf.placeholder(dtype=tf.float32, shape=(None,), name='reward')
        self.next_state = tf.placeholder(dtype=tf.float32, shape=(None, self.feature_n), name='next_state')

        self.eval_net, t_params = self._get_net('eval_net')
        self.target_net, e_params = self._get_net('target_net')

        q_value = tf.reduce_sum(self.eval_net * self.action, axis=1)

        target_value = self.reward + self.gamma * tf.reduce_max(self.target_net, axis=1)

        # **Important**, because the td_target is got by compute tensors
        # or you can compute it out of the net,
        # namely use a placeholder to store the td_target
        target_value = tf.stop_gradient(target_value)

        self.loss = tf.losses.mean_squared_error(q_value, target_value)
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
            net = tf.contrib.layers.fully_connected(Z, self.action_n, activation_fn=None)
        return net, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

    def _get_array_from(self, exp):
        states = []
        for state in exp:
            states.append(state)
        return np.array(states)

    def update(self):
        loss = 0

        if not self.memory.is_enough():
            return loss

        self.update_count += 1

        # decreatse epsilon
        if self.epsilon > self.end_eps:
            self.epsilon -= self.eps_drop

        # update
        if self.update_count % self.update_freq == 0:
            # [state, action, next_state, reward]
            experience = self.memory.sample()

            # extract elements
            states = self._get_array_from(experience[:, 0])
            actions = self._get_array_from(experience[:, 1])
            next_states = self._get_array_from(experience[:, 2])
            rewards = experience[:, 3]

            # feed data
            feed_dict = {self.reward: rewards, self.action: actions,
                         self.state: states, self.next_state: next_states}

            # run optimizer
            loss, summary, _ = self.sess.run([self.loss, self.summaries, self.opt], feed_dict=feed_dict)

            # add log
            self.writer.add_summary(summary, self.update_count)
        # switch object
        if self.update_count % self.switch_freq == 0:
            print('--------------switch-----------')
            self.sess.run(self.replace_target_op)
        return loss

    def get_action(self, state, rand=True):
        idx = 0
        # epsilon greedy
        if rand:
            if np.random.uniform() < self.epsilon:
                idx = 0 if np.random.uniform() > 0.5 else 1
        # eval action
        else:
            idx = np.argmax(
                self.sess.run(self.eval_net,
                              feed_dict={self.state: np.reshape(state, (1, 4))}))
        action = np.zeros((self.action_n,))
        action[idx] = 1
        return action

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
        a = np.array(self.exp)
        return a[np.random.choice(idx, self.batch_size)]

    def is_enough(self):
        return self.exp.shape[0] >= self.batch_size

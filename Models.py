import tensorflow as tf
import numpy as np
from rl_utils import Memory


class DQN:
    """
    A DQN model, which has a three layers in the model
    """
    def __init__(self, game_name,
                 feature_n=4,
                 action_n=2,
                 gamma=0.8,
                 learning_rate=1e-4,
                 eps_drop=0.95,
                 batch_size=256,
                 max_memory_size=1000
                 ):

        self.game_name = game_name

        self.feature_n = feature_n
        self.action_n = action_n

        self.gamma = gamma
        self.learning_rate = learning_rate
        self.eps_drop = eps_drop

        self.switch_freq = 800
        self.batch_size = batch_size
        self.max_memory_size = max_memory_size

        self.start_eps = 1.0
        self.end_eps = 1e-5
        self.annel_eps_drop = 10
        self.epsilon = self.start_eps

        self.memory = Memory(self.batch_size, self.max_memory_size)

        e_params, t_params = self._init_ops()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # ------------------- log ---------------
        self._init_log(e_params, t_params)

        self._load_model()

    def _init_ops(self):
        self.action = tf.placeholder(dtype=tf.float32, shape=(None, self.action_n), name='actions')
        self.reward = tf.placeholder(dtype=tf.float32, shape=(None,), name='reward')

        self.state, self.eval_net, e_params = self._get_net('eval_net')
        self.next_state, self.target_net, t_params = self._get_net('target_net')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        q_value = tf.reduce_sum(self.eval_net * self.action, axis=1)
        target_value = self.reward + self.gamma * tf.reduce_max(self.target_net, axis=1)
        # **Important**, because the td_target is got by compute tensors
        # or you can compute it out of the net,
        # namely use a placeholder to store the td_target
        target_value = tf.stop_gradient(target_value)
        self.loss = tf.losses.mean_squared_error(q_value, target_value)
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        return e_params, t_params

    def _init_log(self, e_params, t_params):
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
            state = tf.placeholder(tf.float32, [None, self.feature_n])
            Z = tf.contrib.layers.fully_connected(state, 128)
            Z = tf.contrib.layers.fully_connected(Z, 128)
            Z = tf.contrib.layers.fully_connected(Z, 128)
            Z = tf.contrib.layers.fully_connected(Z, self.action_n, activation_fn=None)
        return state, Z, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

    def _load_model(self):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state('checkpoints_' + self.game_name)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded model:", checkpoint.model_checkpoint_path)
        else:
            print("No model, start a new training process...")

    def save_model(self, global_step):
        self.saver.save(self.sess, 'checkpoints_' + self.game_name + '/checkpoints', global_step=global_step)

    def train(self, global_step, finished=False):
        loss = 0
        if not self.memory.is_enough() or finished:
            return loss

        # update
        # [state, action, next_state, reward]
        experience = self.memory.sample()

        # extract elements
        states = self.memory.get_array_from(experience[:, 0])
        actions = self.memory.get_array_from(experience[:, 1])
        rewards = experience[:, 2]
        next_states = self.memory.get_array_from(experience[:, 3])

        # feed data
        feed_dict = {self.reward: rewards, self.action: actions,
                     self.state: states, self.next_state: next_states}

        # run optimizer
        loss, summary, _ = self.sess.run([self.loss, self.summaries, self.opt], feed_dict=feed_dict)

        # add log
        self.writer.add_summary(summary, global_step)

        # switch object
        if global_step % self.switch_freq == 0:
            print('===========switch============')
            self.sess.run(self.replace_target_op)
        return loss

    def get_action(self, state, global_step, rand=True):
        # decreatse epsilon
        if global_step % self.annel_eps_drop == 0 and self.epsilon > self.end_eps:
            self.epsilon *= self.eps_drop

        """
        Important !!!!!!!!!!!!
        
        I didn't notice that I forget indent the code:
        My original code is like this,
        where the another max action will never be done since rand is always True
        
        if rand:
            # epsilon greedy
            if np.random.uniform() < self.epsilon:
                idx = 0 if np.random.uniform() > 0.5 else 1
        # eval action
        else:
            idx = np.argmax(
                self.sess.run(self.eval_net,
                              feed_dict={self.state: np.reshape(state, (1, 4))}))
        
        """
        if rand:
            # epsilon greedy
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


import tensorflow as tf
import numpy as np
import gym
from rl_utils import * #Memory, de_descrete_action


class DQN:
    """
    A DQN model, which has a three layers in the model
    """

    def __init__(self, game_name,
                 state_n=4,
                 action_n=2,
                 gamma=0.8,
                 learning_rate=1e-4,
                 eps_drop=0.95,
                 batch_size=256,
                 max_memory_size=1000,
                 swtich_freq=800,
                 double_q_leaning=True
                 ):

        self.game_name = game_name

        self.state_n = state_n
        self.action_n = action_n

        self.gamma = gamma
        self.learning_rate = learning_rate
        self.eps_drop = eps_drop

        self.switch_freq = swtich_freq
        self.batch_size = batch_size
        self.max_memory_size = max_memory_size

        self.start_eps = 1.0
        self.end_eps = 1e-5
        self.annel_eps_drop = 10
        self.epsilon = self.start_eps

        self.log = False
        self.double_q_learning = double_q_leaning

        self.memory = Memory(self.batch_size, self.max_memory_size)

        self.state_shape, self.action_shape, self.reward_shape = self.get_tensor_shape()

        self._init_ops()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self._load_model()

    def _init_ops(self):
        self.action = tf.placeholder(dtype=tf.float32, shape=self.action_shape, name='actions')
        self.reward = tf.placeholder(dtype=tf.float32, shape=self.reward_shape, name='reward')
        self.next_actions_for_target_value = tf.placeholder(dtype=tf.float32, shape=self.action_shape, name='actions_for_target_value')

        self.state, self.eval_net, self.e_params = self._get_net('eval_net')
        self.next_state, self.target_net, self.t_params = self._get_net('target_net')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]

        # q_value which is correspond to the action you have selected
        # Q(s, a)
        q_value = tf.reduce_sum(self.eval_net * self.action, axis=1)

        if self.double_q_learning:
            # Comput target value for double Q-learning
            # one hot index of max q_value
            # next_max_action_value = Q(s', argmax_a Q(s', a))  select
            double_q_action_idx = tf.one_hot(tf.argmax(self.next_actions_for_target_value, axis=1), self.action_n)
            next_max_action_value = tf.reduce_sum(self.target_net * double_q_action_idx, axis=1)
        else:
            # **Important**, You have to stop gradient to target_net
            # which updates by copy params from eval_net
            # or you can compute it out of the net,
            # namely use a placeholder to store the td_target
            #
            # next_max_action_value = max_a Q(s', a)
            next_max_action_value = tf.reduce_max(self.target_net, axis=1)

        # TD_target = Reward + gamma * next_max_action_value
        target_value = self.reward + self.gamma * next_max_action_value
        target_value = tf.stop_gradient(target_value)
        # minimize (Q(s, a) - Q(s', a')) ** 2
        self.loss = tf.losses.mean_squared_error(q_value, target_value)
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def start_log(self, path):
        self.writer = tf.summary.FileWriter(path, self.sess.graph_def)
        tf.summary.scalar('loss', self.loss)
        # write the last weight and bias in both eval_net and target_net
        tf.summary.histogram(self.e_params[-1].name,
                             self.e_params[-1]),
        tf.summary.histogram(self.e_params[-2].name,
                             self.e_params[-2]),
        tf.summary.histogram(self.t_params[-1].name,
                             self.t_params[-1]),
        tf.summary.histogram(self.t_params[-2].name,
                             self.t_params[-2]),
        self.summaries = tf.summary.merge_all()
        self.log = True

    def _get_net(self, scope):
        with tf.variable_scope(scope):
            state = tf.placeholder(tf.float32, self.state_shape)
            Z = self.get_net(state)
        return state, Z, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

    def _load_model(self):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state('checkpoints_' + self.game_name)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded model:", checkpoint.model_checkpoint_path)
        else:
            print("No model, start a new training process...")

    def save_model(self, global_step, path=""):
        self.saver.save(self.sess, 'checkpoints_' + self.game_name + '/checkpoints', global_step=global_step)

    def train(self, global_step, finished=False):
        """
        Train the model with history experience in memory.
        :param global_step:
        :param finished:
        :return:
        """
        loss = 0
        if not self.memory.is_enough() or finished:
            return loss

        # update
        # [state, action, reward, next_state]
        experience = self.memory.sample()
        states, actions, rewards, next_states = self.extract_from_experience(experience)

        # feed data
        feed_dict = {self.reward: rewards, self.action: actions,
                     self.state: states, self.next_state: next_states}

        if self.log:
            if not self.double_q_learning:
                # run optimizer
                loss, summary, _ = self.sess.run([self.loss, self.summaries, self.opt], feed_dict=feed_dict)
            else:
                # we have to compute the next actions for eval_net to compute the target value for Double DQN
                actions_for_target_value = self.sess.run(self.eval_net, feed_dict={self.state:next_states})
                feed_dict[self.next_actions_for_target_value] = actions_for_target_value
                loss, summary, _ = self.sess.run([self.loss, self.summaries, self.opt], feed_dict=feed_dict)
            # add log
            self.writer.add_summary(summary, global_step)
        else:
            loss, _ = self.sess.run([self.loss, self.opt], feed_dict=feed_dict)

        # switch object
        if global_step % self.switch_freq == 0:
            print('===========switch============')
            self.sess.run(self.replace_target_op)
        return loss

    def get_action(self, state, global_step, rand=True):
        """
        Get action from the model which calculate the eval_net
        and output an one-hot vector of the action.
        :param state: current state of the agent
        :param global_step: current global_step
        :param rand: if not rand, then this method will output the action of
               argmax(Q-value)
        :return: an one-hot of action
        """
        # decreatse epsilon
        if global_step % self.annel_eps_drop == 0 and self.epsilon > self.end_eps:
            self.epsilon *= self.eps_drop
        if rand:
            # epsilon greedy
            if np.random.uniform() < self.epsilon:
                # idx = 0 if np.random.uniform() > 0.5 else 1
                idx = random_action(self.action_n)
            # eval action
            else:
                idx = np.argmax(
                    self.sess.run(self.eval_net,
                                  feed_dict={self.state: np.reshape(state, (1, self.state_n))}))
        action = np.zeros((self.action_n,))
        action[idx] = 1
        return action

    def add_memory(self, exp):
        """
        Add memory to the model for later learning progress.
        :param exp: the experience tuple of (state, action, reward, next_state)
        """
        self.memory.add(exp)

    def close(self):
        """
        Close resources.
        """
        self.sess.close()
        if self.writer is not None:
            self.writer.close()

    def get_tensor_shape(self):
        """
        :return : a tuple contains (state, action, reward),
                  each element is a shape in tensorflow.
                  Such as :
                  ((None, self.state_n), (None, self.action_n), (None,))
        """
        raise Exception('You should implement this function')

    def get_net(self, state):
        """
        You should return a network for the DQN using its
        original state
        :param state: a tensor input for the net
        """
        raise Exception('You should implement this function')

    def extract_from_experience(self, experience):
        """
        You should extract  (states, actions, rewards, next_states)
        from what you have stored in memory.
        :param : the experience you have putted in
        :return: a tuple containing (states, actions, rewards, next_states)
        """
        # extract elements
        raise Exception('You should implement this function')


class Example(DQN):
    """
    An Example of using the DQN to implement a simple CartPole game.
    """
    def __init__(self,
                 game,
                 state_n,
                 action_n,
                 batch_size=100,
                 max_memory_size=1000,
                 learning_rate=0.0001,
                 switch_freq=800,
                 double_q_leaning=True):
        DQN.__init__(self,
                     game_name=game,
                     state_n=state_n,
                     action_n=action_n,
                     batch_size=batch_size,
                     max_memory_size=max_memory_size,
                     learning_rate=learning_rate,
                     swtich_freq=switch_freq,
                     double_q_leaning=double_q_leaning)
        self.state_n = state_n
        self.action_n = action_n

    def get_tensor_shape(self):
        return (None, self.state_n), (None, self.action_n), (None, )

    def get_net(self, state):
        Z = tf.contrib.layers.fully_connected(state, 128)
        Z = tf.contrib.layers.fully_connected(Z, 128)
        Z = tf.contrib.layers.fully_connected(Z, 128)

        output = tf.contrib.layers.fully_connected(Z, self.action_n, activation_fn=None)
        # Dualing Q net
        # advantage, value = tf.split(Z, num_or_size_splits=2, axis=1)
        #
        # action_adv = tf.contrib.layers.fully_connected(advantage, self.action_n)
        # state_value = tf.contrib.layers.fully_connected(value, 1)
        # output = state_value + (action_adv - tf.reduce_mean(action_adv, keep_dims=True))

        return output

    def extract_from_experience(self, experience):
        # extract elements
        states = self.memory.get_array_from(experience[:, 0])
        actions = self.memory.get_array_from(experience[:, 1])
        rewards = experience[:, 2]
        next_states = self.memory.get_array_from(experience[:, 3])
        return states, actions, rewards, next_states


def train_structure(NAME='CartPole-v0',
                    batch_size=100,
                    max_memory_size=1000,
                    learning_rate=0.0001,
                    switch_freq=800,
                    max_episode=1000,
                    max_record_reward_num=50,
                    save_model_freq=200,
                    log_dir="D:\\NJU\\Games\\log\\Q_learning_in_cartpole\\log",
                    finish_reward=195,
                    reward_fn=lambda state, next_state, done, reward, total_reward : total_reward,
                    action_fn=lambda env, action_matrix : np.argmax(action_matrix),
                    state_n=None,
                    action_n=None):
    env = gym.make(NAME)
    # initialize the model
    if state_n is None:
        state_n = env.observation_space.shape[0]
    if action_n is None:
        action_n = env.action_space.n

    model = Example(NAME,
                    state_n,
                    action_n,
                    batch_size=batch_size,
                    max_memory_size=max_memory_size,
                    learning_rate=learning_rate,
                    switch_freq=switch_freq)
    model.start_log(log_dir)
    global_step = 0
    last_total_rewards = RewardQueue(max_record_reward_num)
    finish = False
    for episode in range(1, max_episode):
        state = env.reset()
        done = False
        total_reward = 0
        sum_loss = 0
        step = 0
        while not done:
            global_step += 1
            env.render()
            # get action from the model which is one hot of action [0 0 0 1 0 0 0]
            action_matrix = model.get_action(state, global_step)

            next_state, reward, done, _ = env.step(action_fn(env, action_matrix))

            total_reward += reward

            reward = reward_fn(state, next_state, done, reward, total_reward)

            if done:
                next_state = np.zeros_like(state)

            # add memory to the model for later learning
            model.add_memory([state, action_matrix, reward, next_state])
            loss = model.train(global_step, finish)
            sum_loss += loss
            state = next_state
            step += 1
        print('episode : %d, total reward: %f, loss : %f, finished : %r' % (episode, total_reward, sum_loss / step, finish))

        last_total_rewards.add_reward(total_reward)

        if episode % save_model_freq == 0:
            print('save model...')
            model.save_model(global_step)

        if last_total_rewards.is_enough() and last_total_rewards.get_average() > finish_reward and not finish:
            print('Finished leanring....')
            model.save_model(global_step)
            break
    model.save_model(global_step)
    env.close()
    model.close()


def start_cart_pole():
    train_structure(NAME='CartPole-v0')


def start_mountain_car():
    def reward_fun(state, next_state, done, reward, total_reward):
        # reward *= 100
        #
        # reward += total_reward + 0.5 + next_state[0]
        reward = total_reward
        if done and total_reward > -200:
            reward = 500
        return reward

    NAME = 'MountainCar-v0'
    train_structure(NAME=NAME,
                    batch_size=1024,
                    max_memory_size=10000,
                    learning_rate=0.0001,
                    switch_freq=8000,
                    log_dir="D:\\NJU\\Games\\log\\Q_learning_in_cartpole",
                    max_record_reward_num=50,
                    max_episode=2000,
                    save_model_freq=200,
                    finish_reward=-150,
                    reward_fn=reward_fun)

def pendulum():
    def reward_fun(state, next_state, done, reward, total_reward):
        return reward*10

    action_n = 100

    def action_fn(env, action_matrix):
        return de_discretize_action(np.argmax(action_matrix), env.action_space.low, env.action_space.high, action_n)

    NAME = 'Pendulum-v0'
    # initialize the model
    train_structure(NAME=NAME,
                    action_n = 100,
                    batch_size=256,
                    max_memory_size=1000,
                    learning_rate=0.0001,
                    switch_freq=1600,
                    log_dir="D:\\NJU\\Games\\log\\Q_learning_in_cartpole",
                    max_record_reward_num=50,
                    max_episode=4000,
                    save_model_freq=200,
                    finish_reward=-200,
                    reward_fn=reward_fun,
                    action_fn=action_fn)


if __name__ == '__main__':
    #  start_cart_pole()
    pendulum()
    # start_mountain_car()


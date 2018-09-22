import gym
import tensorflow as tf
import random
import numpy as np
from rl_utils import Memory

class DQN:

    def __init__(self):
        self.switch_freq = 800
        self.batch_size = 256
        self.max_memory_size = 1000

        self.feature_n = 4
        self.action_n = 2

        self.gamma = 0.8
        self.learning_rate = 1e-4
        self.start_eps = 1.0
        self.end_eps = 1e-5
        self.eps_drop = 0.95
        self.annel_eps_drop = 10
        self.epsilon = self.start_eps

        self.eps = self.start_eps
        self.memory = Memory(self.batch_size, self.max_memory_size)

        self._init_ops()

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.initialize_all_variables())

        self.load_model()
        self._init_memory()

    # same
    def _init_ops(self):
        self.action = tf.placeholder(tf.float32, [None, self.action_n])
        self.reward = tf.placeholder(tf.float32, [None, ])

        self.state, self.eval_net, e_params = self._get_net('eval_net')
        self.next_state, self.target_net, t_params = self._get_net('target_net')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.q_value = tf.reduce_sum(self.eval_net * self.action, reduction_indices=1)
        self.td_target = self.reward + self.gamma * tf.reduce_max(self.target_net, reduction_indices=1)
        tf.stop_gradient(self.td_target)
        self.loss = tf.reduce_mean(tf.square(self.q_value - self.td_target))
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    # same
    def _get_net(self, scope):
        with tf.variable_scope(scope):
            state = tf.placeholder(tf.float32, [None, self.feature_n])
            Z = tf.contrib.layers.fully_connected(state, 128)
            Z = tf.contrib.layers.fully_connected(Z, 128)
            Z = tf.contrib.layers.fully_connected(Z, 128)
            Z = tf.contrib.layers.fully_connected(Z, self.action_n, activation_fn=None)
        return state, Z, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

    def load_model(self):
        # saving and loading networks
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("checkpoints-cartpole")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def save_model(self, global_step):
        self.saver.save(self.sess, 'checkpoints-cartpole/' + GAME + '-dqn', global_step=global_step)

    # same
    def get_action(self, feed, global_step):
        # get experience
        if global_step % self.annel_eps_drop == 0 and self.eps > self.end_eps:
            self.eps *= self.eps_drop

        if random.random() <= self.eps:
            # action_index = env.action_space.sample()
            action_index = random.randrange(self.action_n)
        else:
            act_values = self.sess.run(self.eval_net, feed_dict=feed)
            action_index = np.argmax(act_values)

        action = np.zeros(self.action_n)
        action[action_index] = 1
        return action

    def train(self, global_step, learning_finished):
        if not self.memory.is_enough():
            return 0

        experience = self.memory.sample()

        # extract elements
        states = self.memory.get_array_from(experience[:, 0])
        actions = self.memory.get_array_from(experience[:, 1])
        rewards = experience[:, 2]
        next_states = self.memory.get_array_from(experience[:, 3])

        feed = {
            self.state: states, self.action: actions,
            self.reward: rewards, self.next_state: next_states}

        if not learning_finished:  # If not solved, we train and get the step loss
            step_loss_value, _ = self.sess.run([self.loss, self.opt], feed_dict=feed)

            if global_step % self.switch_freq == 0:
                print("========= SWITCH ===========")
                self.sess.run(self.replace_target_op)

        else:  # If solved, we just get the step loss
            step_loss_value = self.sess.run(self.loss, feed_dict=feed)
        return step_loss_value

    def add_momery(self, memory):
        self.memory.add(memory)


def train(env):
    # Define placeholders to catch inputs and add OPTIONS
    agent = DQN()

    # Some initial local variables
    global_step = 0
    learning_finished = False
    # Score cache
    score_queue = []
    # The episode loop
    for i_episode in range(MAX_EXPERIENCE):

        observation = env.reset()
        done = False
        score = 0
        sum_loss_value = 0

        # The step loop
        while not done:
            global_step += 1

            env.render()

            memory = []
            memory.append(observation)

            action = agent.get_action({agent.state: np.reshape(observation, (1, -1))}, global_step)

            memory.append(action)
            observation, reward, done, _ = env.step(np.argmax(action))

            score += reward
            reward = score  # Reward will be the accumulative score

            if done and score < 200:
                reward = -500  # If it fails, punish hard
                observation = np.zeros_like(observation)

            memory.append(reward)
            memory.append(observation)
            agent.add_momery(memory)

            step_loss_value = agent.train(global_step, learning_finished)
            # Use sum to calculate average loss of this episode
            sum_loss_value += step_loss_value

        print("====== Episode {} ended with score = {}, avg_loss = {} ======".format(i_episode + 1, score,
                                                                                     sum_loss_value / score))

        score_queue.append(score)
        if len(score_queue) > MAX_SCORE_QUEUE_SIZE:
            score_queue.pop(0)
            if np.mean(score_queue) > 195:  # The threshold of being solved
                learning_finished = True
            else:
                learning_finished = False

        if learning_finished:
            print("Testing !!!")
        # save progress every 100 episodes
        if learning_finished and i_episode % 100 == 0:
            agent.save_model(global_step)


MAX_SCORE_QUEUE_SIZE = 50  # number of episode scores to calculate average performance
GAME = 'CartPole-v0'  # name of game
MAX_EXPERIENCE = 3000

if __name__ == "__main__":
    env = gym.make(GAME)
    # env.monitor.start(OUT_DIR, force=True)
    train(env)
    # env.monitor.close()

import tensorflow as tf
import numpy as np
import gym
from rl_utils import *

class PolicyGradient:
    def __init__(self,
                 name,
                 state_n,
                 action_n,
                 learning_rate=0.01):

        self.name = name

        self.enough = False
        self.trajectory_num = 1

        self.current_states=[]
        self.current_actions=[]
        self.current_rewards=[]

        self.state_n = state_n
        self.action_n = action_n
        self.gamma = 0.95
        self.current_gamma = 1

        self.state_shape, self.action_shape = self.get_shape()

        self.pl_state = tf.placeholder(tf.float32, self.state_shape, 'pl_state')
        self.pl_action = tf.placeholder(tf.float32, self.action_shape, 'pl_action')
        self.pl_trajectory_reward = tf.placeholder(tf.float32, (None,), 'trajectory_reward')

        X = tf.contrib.layers.fully_connected(self.pl_state, 100)
        X = tf.contrib.layers.fully_connected(X, 50)
        X = tf.contrib.layers.fully_connected(X, self.action_n, activation_fn=None)
        self.policy_net = tf.nn.softmax(X)
        self.prediction = tf.argmax(self.policy_net)

        # this is where my first version of PG doesn't work
        self.loss = tf.losses.softmax_cross_entropy(self.pl_action, self.policy_net, weights=self.pl_trajectory_reward)
        self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        self._load_model()

    def get_shape(self):
        return (None, self.state_n), (None, self.action_n)

    def get_action(self, state):
        feed_dict = {self.pl_state: np.reshape(state, (1, self.state_n))}
        action_matrix = self.sess.run(self.policy_net, feed_dict=feed_dict)
        idx = choose_action(np.reshape(action_matrix, (self.action_n,)))
        action = np.zeros((self.action_n, ))
        action[idx] = 1
        return action

    def _load_model(self):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state('checkpoints_' + self.name)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded model:", checkpoint.model_checkpoint_path)
        else:
            print("No model, start a new training process...")

    def save_model(self,):
        self.saver.save(self.sess, 'checkpoints_' + self.name + '/checkpoints')

    def add_memory(self, state, action, reward, next_state, done):
        self.current_states.append(state)
        self.current_actions.append(action)
        self.current_rewards.append(reward)

        if done:
            last_reward = 0
            for i in reversed(range(len(self.current_rewards))):
                last_reward = self.current_rewards[i] + self.gamma * last_reward
                self.current_rewards[i] = last_reward

            rewards = np.array(self.current_rewards)
            mean = np.mean(rewards)
            std = np.std(rewards)
            self.current_rewards = (rewards - mean) / std
            self.enough = True

    def is_enough(self):
        return self.enough

    def run(self):
        feed_dict = {self.pl_state : self.current_states,
                     self.pl_action : self.current_actions,
                     self.pl_trajectory_reward : self.current_rewards}

        loss, _ = self.sess.run([self.loss, self.opt], feed_dict=feed_dict)

        self.current_states = []
        self.current_actions = []
        self.current_rewards = []
        self.enough = False

        return loss

    def close(self):
        self.sess.close()


def train_structure(env,
                    model,
                    max_episode=10000,
                    max_record_reward_num=50,
                    action_fn=lambda env, action_matrix : np.argmax(action_matrix)):
    global_step = 0
    last_total_rewards = RewardQueue(max_record_reward_num)
    finish = False
    for episode in range(1, max_episode):
        state = env.reset()
        done = False
        total_reward = 0
        step = 0
        while not done:
            global_step += 1
            env.render()
            # get action from the model which is one hot of action [0 0 0 1 0 0 0]
            action_matrix = model.get_action(state)

            next_state, reward, done, _ = env.step(action_fn(env, action_matrix))

            total_reward += reward

            if done:
                next_state = np.zeros_like(state)

            # add memory to the model for later learning
            model.add_memory(state, action_matrix, reward, next_state, done)
            state = next_state
            step += 1

        if model.is_enough():
            loss = model.run()
            print('episode : %d, total reward: %f, loss : %f, finished : %r' % (episode, total_reward, loss, finish))

        if done and episode % 100 == 0:
            print("Save model...")
            model.save_model()

        last_total_rewards.add_reward(total_reward)


def start_mountain_car():
    NAME = 'MountainCar-v0'
    env = gym.make(NAME)
    env = env.unwrapped
    model = PolicyGradient(NAME, action_n=env.action_space.n, state_n=env.observation_space.shape[0])
    try:
        train_structure(env, model)
    finally:
        env.close()
        model.close()

def start_cart_pole():
    NAME = 'CartPole-v0'
    env = gym.make(NAME)
    env = env.unwrapped
    model = PolicyGradient(NAME, action_n=env.action_space.n, state_n=env.observation_space.shape[0])
    try:
        train_structure(env, model)
    finally:
        env.close()
        model.close()


if __name__ == '__main__':
    start_cart_pole()
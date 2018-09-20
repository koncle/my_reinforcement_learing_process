import gym
import tensorflow as tf
import numpy as np
from tensorboardX import SummaryWriter
from Models import *


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
    print('best reward ', best_reward, 'best parameters ', best_param)
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


def DQNSearch():
    env = gym.make('CartPole-v0')
    model = DQN()
    for _ in range(1000):
        state = env.reset()
        done = False
        total_reward = 0
        sum_loss = 0
        step = 0
        while not done:
            action_matrix = model.get_action(state)
            action = np.argmax(action_matrix)
            next_state, reward, done, info = env.step(action)
            env.render()

            total_reward += reward
            reward = total_reward

            if done and reward < 190:
                reward = -500

            model.add_memory([state, action_matrix, next_state, reward])
            loss = model.update()
            sum_loss += loss
            state = next_state
            step += 1

        print('total reward: %f, loss : %f' % (total_reward, sum_loss / step))
    env.close()
    model.close()

if __name__ == '__main__':
    DQNSearch()

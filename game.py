import gym
from Models import *

"""
Use DQN to teach the agent to learn the cartpole game.
I followed this [[tutorial][http://kvfrans.com/simple-algoritms-for-solving-cartpole]]
and implement three methods:
1. randomly select actions until find the best
2. use hill-climbing method, start from a point and search around,
   if there is a better action then move to that place(change weight),
   finally we will find the best solution
3. Use DQN to solve the problem. It is so powerful but need a lot of
carefulness to implement it. 

My painful Experience : For example, I forgot to indent a code
block which results in a long lasting failure to learning any policy 
for my agent. This really frustrates me and I can't find where is 
wrong with my code(I still believe that code block is very ok). The only 
way to address this problem is to compare my code to others' code. So I 
decide to download [[a DQN repository][https://github.com/lbbc1117/ClassicControlDQN]] 
from github which works well on cartpole game. I rewrite the code in this repo and 
rewrite my code simultaneously. After changing a code block in his code, 
I have to test whether the altered code still works well. Finally, I 
rewrite all the code in his repo, but found nothing strange in my code.
Thus I have to view his and mine code line by line, and found the error 
in the function : get_action() which always return a random action. That's
why the game always have a reward between 8 and 12. 
"""

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
    NAME =  'CartPole-v0'
    env = gym.make(NAME)
    model = DQN(NAME)
    global_step = 0
    max_record_num = 50
    last_total_rewards = []
    finish = False
    for episode in range(1000):
        state = env.reset()
        done = False
        total_reward = 0
        sum_loss = 0
        step = 0
        while not done:
            global_step += 1
            env.render()

            action_matrix = model.get_action(state, global_step)

            next_state, reward, done, _ = env.step(np.argmax(action_matrix))

            total_reward += reward
            reward = total_reward

            if done and reward < 200:
                reward = -500
                next_state = np.zeros_like(state)

            model.add_memory([state, action_matrix, reward, next_state])
            loss = model.train(global_step, finish)
            sum_loss += loss
            state = next_state
            step += 1
        print('episode : %d, total reward: %f, loss : %f, finished : %r' % (episode, total_reward, sum_loss / step, finish))
        last_total_rewards.append(total_reward)
        if len(last_total_rewards) > max_record_num:
            last_total_rewards.pop(0)

        if np.mean(last_total_rewards) > 195 and not finish:
            print('Finished leanring....')
            model.save_model(global_step)
            finish = True
    env.close()
    model.close()

if __name__ == '__main__':
    # RandomSearch()
    # HillCliming()
    DQNSearch()

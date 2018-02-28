## 方法

from TSP_Burma14 import ENV
from D3QN import DQN
import numpy as np
import matplotlib.pyplot as plt

env = ENV()

# env.location_display()

RL = DQN(n_actions=env.action_dim,
         n_features=env.state_dim,
         learning_rate=0.001,
         gamma=0.9,
         e_greedy_end=0.2,
         memory_size=3000,
         e_liner_times=10000,
         batch_size=64,
         output_graph=False,
         double=True,
         dueling=True,
         units=10,
         train=False,
         # train=True
         )

# train part
if RL.train:
    # if True:
    step = 0
    ep_reward = 0
    episodes = 30000
    for episode in range(episodes):
        ep_reward = 0
        distance_all = 0
        step = 0
        observation = env.reset()  # initial observation
        step_his = []
        while True:
            action = RL.choose_action(observation, env)  # RL choose action based on observation
            step_his.append(action)
            # action_index = [i for i in range(env.action_dim) if env.state[i] != -1]
            # action = np.random.choice(action_index)
            observation_, reward, done, info = env.step(action)  # RL get next observation and reward
            ep_reward += reward
            distance_all += info["distance"]
            RL.store_transition(observation, action, reward, observation_)  # store memory

            if RL.memory_counter > RL.memory_size:
            # if RL.memory_counter > RL.memory_size:
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1
        if episode % 50 == 0:
            print('Episode:', episode + 1, '/', episodes, 'step:', step, ' ep_reward: %.4f' % ep_reward,
                  'distance_all: %.4f' % distance_all, 'epsilon: %.3f' % RL.epsilon)
            print('action', step_his)

    # save net
    RL.net_save()
    # end of game
    print('train over')
else:
    trajectory_record = np.zeros([500, 2])
    trajectory_record[0, 0] = env.city_location[0][0]
    trajectory_record[0, 1] = env.city_location[0][1]
    distance = 0
    observation = env.reset()  # initial observation
    step = 0
    step_his = []
    for step in range(500):
        action = RL.choose_action(observation, env)  # RL choose action based on observation
        if step == 3:
            action = 4
        step_his.append(action)
        trajectory_record[step + 1, 0] = env.city_location[action][0]
        trajectory_record[step + 1, 1] = env.city_location[action][1]
        observation_, reward, done, info = env.step(action)  # RL get next observation and reward
        distance += info["distance"]
        print('reward', reward)

        # swap observation
        observation = observation_

        for i in range(len(env.city_location)):
            plt.scatter(env.city_location[i][0], env.city_location[i][1])
            plt.text(env.city_location[i][0], env.city_location[i][1], str(i), size=15, alpha=0.2)
        plt.plot(trajectory_record[:step + 2, 0], trajectory_record[:step + 2, 1])
        plt.show()
        plt.pause(0.5)

        # break while loop when end of this episode
        if done:
            print(step)
            print(distance)
            print('action', step_his)
            break

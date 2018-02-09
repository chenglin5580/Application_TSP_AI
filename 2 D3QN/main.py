

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
         e_greedy_end=0.1,
         memory_size=3000,
         e_liner_times=10000,
         batch_size=50,
         output_graph=False,
         double=True,
         dueling=True,
         train=False
         )


# train part
if RL.train:
# if True:
    step = 0
    ep_reward = 0
    episodes = 10000
    for episode in range(episodes):
        ep_reward = 0
        step = 0
        observation = env.reset()  # initial observation
        while True:
            action = RL.choose_action(observation)               # RL choose action based on observation
            # action_index = [i for i in range(env.action_dim) if env.state[i] != -1]
            # action = np.random.choice(action_index)
            observation_, reward, done, info = env.step(action)  # RL get next observation and reward
            ep_reward += reward
            RL.store_transition(observation, action, reward, observation_)  # store memory

            if RL.memory_counter > RL.memory_size:
                 RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1
        print('Episode:', episode + 1, '/', episodes, 'step:', step,  ' ep_reward: %.4f' % ep_reward, 'epsilon: %.3f' % RL.epsilon)
    # save net
    RL.net_save()
    # end of game
    print('train over')
else:
    trajectory_record = np.zeros([500, 2])
    trajectory_record[0, 0] = env.city_location[0][0]
    trajectory_record[0, 1] = env.city_location[0][1]
    ep_reward = 0
    observation = env.reset()  # initial observation
    step = 0
    for step in range(500):
        action = RL.choose_action(observation, env)               # RL choose action based on observation
        trajectory_record[step + 1, 0] = env.city_location[action][0]
        trajectory_record[step + 1, 1] = env.city_location[action][1]
        observation_, reward, done, info = env.step(action)  # RL get next observation and reward
        ep_reward += reward
        print('reward', reward)

        # swap observation
        observation = observation_

        for i in range(len(env.city_location)):
            plt.scatter(env.city_location[i][0], env.city_location[i][1])
        plt.plot(trajectory_record[:step + 2, 0], trajectory_record[:step + 2, 1])
        plt.show()
        plt.pause(1)

        # break while loop when end of this episode
        if done:
            print(step)
            break



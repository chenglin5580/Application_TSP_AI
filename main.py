from TSP_Burma14 import ENV

import numpy as np

env = ENV()

# env.location_display()

# RL = DQN(n_actions=env.action_dim,
#          n_features=env.state_dim,
#          learning_rate=0.01,
#          gamma=0.9,
#          e_greedy_end=0.1,
#          memory_size=3000,
#          e_liner_times=10000,
#          batch_size=256,
#          output_graph=False,
#          double=True,
#          dueling=True,
#          train=False
#          )


# train part
# if RL.train:
if True:
    step = 0
    ep_reward = 0
    episodes = 300
    for episode in range(episodes):
        ep_reward = 0
        observation = env.reset()  # initial observation
        while True:
            # action = RL.choose_action(observation)               # RL choose action based on observation
            action_index = [i for i in range(env.action_dim) if env.state[i] != -1]
            action = np.random.choice(action_index)
            observation_, reward, done, info = env.step(action)  # RL get next observation and reward
            ep_reward += reward
            # RL.store_transition(observation, action, reward, observation_)  # store memory

            # if RL.memory_counter > RL.memory_size:
            #     RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1
        print('Episode:', episode + 1, '/', episodes, ' ep_reward: %.4f' % ep_reward, 'epsilon: %.3f' % RL.epsilon)
    # save net
    # RL.net_save()
    # end of game
    print('train over')
else:
    pass



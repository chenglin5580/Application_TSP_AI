'''
TSP 缅甸 Burma 14 城市
'''

import matplotlib.pyplot as plt
import numpy as np

class ENV(object):
    def __init__(self):
        self.city_location = [[16.47, 96.10],
                              [16.47, 94.44],
                              [20.09, 92.54],
                              [22.39, 93.37],
                              [25.23, 97.24],
                              [22.00, 96.05],
                              [20.47, 97.02],
                              [17.20, 96.29],
                              [16.30, 97.38],
                              [14.05, 98.12],
                              [16.53, 97.38],
                              [21.52, 95.59],
                              [19.42, 97.13],
                              [20.09, 94.55],
                              ]
        self.action_dim = len(self.city_location) - 1
        self.state_dim = self.action_dim
        self.state = self.reset()
        self.action_old = int(0)


    def location_display(self):

        plt.figure(1)
        for i in range(len(self.city_location)):
            plt.scatter(self.city_location[i][0], self.city_location[i][1])

        plt.show()

    def reset(self):
        state_ini = -1 * np.ones([self.state_dim])
        state_ini[0] = -1
        return state_ini.copy()

    def render(self):
        pass


    def step(self, action):

        self.state[int(action)] = -1

        reward = -1
        reward -= np.square(self.city_location[action] - self.city_location[self.action_old])

        delta_sum = np.sum(self.state - -1 * np.ones([self.state_dim]))
        if delta_sum == 0:
            reward -= np.square(self.city_location[action] - self.city_location[0])
            done = True
        else:
            done = False

        return self.state.copy(), reward, done, []










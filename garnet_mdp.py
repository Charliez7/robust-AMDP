import os.path

import numpy as np
import gym
import xlsxwriter
from gym.spaces import Discrete
def generate_dirichlet_rounded(alpha, n, decimals=2):
   
    vector = np.random.dirichlet(alpha) 
    rounded_vector = np.round(vector, decimals)
    rounded_vector[-1] = 1.0 - np.sum(rounded_vector[:-1])
    rounded_vector[-1] = round(rounded_vector[-1], decimals)
    return rounded_vector

def initial_distribution(n):
    return generate_dirichlet_rounded([1] * n, n, decimals=3)

def initial_policy(m,n):
    return np.array([generate_dirichlet_rounded([1] * n, n, decimals=3) for _ in range(m)])

class GarnetMDP(gym.Env):
    def __init__(self, n_state, n_action,p_init, nb):
        self.state_num = n_state
        self.action_num = n_action
        self.state = np.random.choice(self.state_num, p=p_init)
        self.transitions = None
        self.rewards = None
        self.p_init=p_init
        self.action_space = Discrete(self.action_num)
        self.observation_space = Discrete(self.state_num)
        self.nb = nb
        self.branch_matrix = None

    def reset(self):
        self.state=np.random.choice(self.state_num, p=self.p_init)
        self.build_transitions()
        self.build_reward()

    def get_transition_prob(self, state, action, next_state):
        if next_state in self.transitions[state][action]:
            return self.transitions[state][action][next_state]
        else:
            return 0

    def build_random(self, state_num):
        temp = np.random.rand(state_num)
        temp = temp / np.sum(temp)
        return temp

    def build_transitions(self):
        transitions = np.zeros((self.state_num, self.action_num, self.state_num))
        branch_matrix = np.zeros((self.state_num, self.action_num, self.state_num))
        for state in range(self.state_num):
            # print(self.states)
            for action in range(self.action_num):
                next_states = np.random.choice(self.state_num, size=self.nb, replace=False)
                random_p_next_states = self.build_random(len(next_states))
                flag = 0
                for next_state in next_states:
                    transitions[state - 1][action - 1][next_state - 1] = random_p_next_states[flag]
                    flag += 1
        self.transitions=transitions
        branch_matrix[transitions == 0] = 1
        self.branch_matrix=branch_matrix
        return None

    def build_reward(self):
        rewards = np.zeros((self.state_num, self.action_num, self.state_num))
        for state in range(self.state_num):
            # print(self.states)
            for action in range(self.action_num):
                # temp_r = np.random.uniform(0, 10)
                for next_state in range(self.state_num):
                    rewards[state - 1][action - 1][next_state - 1] = np.random.uniform(0, 10)
        self.rewards=rewards
        return None


    def step(self, action):
        next_state = np.random.choice(self.state_num, p=self.transitions[self.state, action])
        reward = self.rewards[self.state, action, next_state]
        self.state = next_state
        return next_state, reward

s_num=5
a_num=3
p_init=initial_distribution(s_num)
env = GarnetMDP(s_num, a_num, p_init, 4)
env.reset()
transition_matrix = env.transitions
branch_matrix = env.branch_matrix
cost = env.rewards
np.save('transition_matrix.npy',transition_matrix)
np.save('branch_matrix.npy',branch_matrix)
np.save('cost.npy',cost)
np.save('p_init.npy',p_init)


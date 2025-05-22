import random
import gym


class ic_env(gym.Env):
    # Initilizes object of Class Environment
    def __init__(self, max_inventory):
        self.fixed_cost = 0
        self.max_inventory = max_inventory
        self.state = max_inventory
        # self.lost_reveune_factor = 1
        self.reward = 0.0

    def reset(self):
        """ Returns the initial state"""
        self.state = self.max_inventory
        self.reward = 0.0
        return self.max_inventory

    # Order cost: 1 per item
    def order_cost(self, x):
        return 1 * x

    def holding_cost(self, x):
        """Holding cost: When input is negative = backorders
             Holding cost=backlog cost: 1 per item"""
        if x < self.max_inventory:
            return 1 * (self.max_inventory - x)
        else:
            return 1 * (x - self.max_inventory)

    def step(self, state, action, demand):
        next_state = state + action - demand

        if next_state <= 0:
            demand = state + action  # If demand excess the maximal possible supply quantity, it will reduce to the maximal possible supply quantity
            next_state = 0

        if next_state > 2 * self.max_inventory:
            next_state = 2 * self.max_inventory  # control the maximal inventory
        reward = self.order_cost(action) + self.holding_cost(next_state)
        self.reward = self.reward+ reward
        self.state = next_state

        return reward

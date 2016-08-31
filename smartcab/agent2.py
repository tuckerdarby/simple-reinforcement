import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""
    def __init__(self, env):
        super(LearningAgent, self).__init__(env)
        # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize
        #parameters
        self.learner = 0.9
        self.discount = 0.7
        self.q = 0
        #state/action
        self.action = None
        self.actions = [None, 'forward', 'left', 'right']  # really just the keys to rewards
        self.rewards = {}
        self.init_states()
        self.runs = 0
        # Measurements
        self.trips = []
        # - must reset
        self.errors = []
        self.moves = 0
        self.acceptable = 0
        self.total_reward = 0

    def init_states(self):
        # Inputs: light, oncoming, right, left, direction
        lights = ['green', 'red']
        #traffic = [True, False]
        oncomings = ['forward', 'left', 'right', None]
        cross = [True, False]
        #rights = ['forward', 'left', 'right', None]
        #lefts = ['forward', 'left', 'right', None]
        directions = ['forward', 'left', 'right']
        states = list(product(lights, oncomings, cross, directions))
        # reward tables
        for action in self.actions:
            self.rewards[action] = {}
            for state in states:
                self.rewards[action][state] = random.uniform(0, 10)

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.log_trial()
        self.total_reward = 0
        self.acceptable = 0
        self.moves = 0
        self.errors = []

    def update(self, t):
        self.define_state()
        old_state = self.state
        # choose action
        action, estimate = self.best_action()
        reward = self.env.act(self, action)
        # Get next state
        self.define_state()
        utility = reward + self.discount * estimate
        self.rewards[action][old_state] += self.learner * (utility - self.q)
        self.q = self.rewards[action][self.state]
        self.log_move(reward)

    def best_action(self):
        chosen = None
        estimate = -100
        #greedy choice
        for action in self.actions:
            if self.rewards[action][self.state] >= estimate:
                chosen = action
                estimate = self.rewards[action][self.state]

        #randomize choice
        randoming = False
        if randoming:
            rand = random.uniform(0, 1)
            if rand > (min(0.8, self.runs) * 0.1):
                random_act = random.randint(0, len(self.actions)-1)
                chosen = self.actions[random_act]
                estimate = self.rewards[chosen][self.state]

        return chosen, estimate

    def define_state(self):
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        #traffic = (inputs['oncoming'] != None or inputs['right'] != None or inputs['left'] != None)
        cross = (inputs['right'] != None or inputs['left'] != None)
        self.state = (inputs['light'], inputs['oncoming'], cross, self.next_waypoint)
        #self.state = (inputs['light'], traffic, self.next_waypoint)
        #self.state = (inputs['light'], inputs['oncoming'], inputs['right'], inputs['left'], self.next_waypoint)

    def log_move(self, reward):
        if reward < 0:
            self.errors.append(reward)
        else:
            self.acceptable += 1
        self.total_reward += reward
        self.moves += 1

    def log_trial(self):
        trip = {
            'reward': self.total_reward,
            'moves': self.moves,
            'errors': sum(self.errors),
            'acceptable': self.acceptable
        }
        self.runs += 1
        self.trips.append(trip)


def run():
    """Run the agent for a finite number of trials."""
    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # specify agent to track
    sim = Simulator(e, update_delay=0.0, display=False)
    sim.run(n_trials=1000)  # run for a specified number of trials

    #data crunch
    visualize_data(['reward', 'moves', 'errors', 'acceptable'], a.trips)


def visualize_data(names, data):
    fig = sns.plt.figure()
    num = len(names)
    for i in range(num):
        ax = fig.add_subplot(num, 1, 1 + i)
        ax.set_title(names[i])
        values = map(lambda x: x[names[i]], data)
        ax.plot(values)
    sns.plt.show()


if __name__ == '__main__':
    run()

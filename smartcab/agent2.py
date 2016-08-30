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
        # TODO: Initializ
        # e any additional variables here
        self.learner = 0.5
        self.discount = 0.5
        self.trips = []
        self.action = None
        self.actions = [None, 'forward', 'left', 'right']  # really just the keys to rewards
        self.rewards = {}
        self.init_states()
        self.q = 0
        # Measurements
        self.total_reward = 0
        self.actions_taken = 0

    def init_states(self):
        # Inputs: light, oncoming, right, left, direction
        lights = ['green', 'red']
        oncomings = ['forward', 'left', 'right', None]
        rights = ['forward', 'left', 'right', None]
        lefts = ['forward', 'left', 'right', None]
        directions = ['forward', 'left', 'right']
        states = list(product(lights, oncomings, rights, lefts, directions))
        # reward tables
        for action in self.actions:
            self.rewards[action] = {}
            for state in states:
                self.rewards[action][state] = random.uniform(0, 0.2)

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.trips.append({'reward': self.total_reward, 'actions': self.actions_taken})
        self.total_reward = 0
        self.actions_taken = 0

    def update(self, t):
        self.define_state()
        oldState = self.state
        # choose action
        action, estimate = self.best_action()
        reward = self.env.act(self, action)
        # Get next state
        self.define_state()
        utility = reward + self.discount * estimate
        self.rewards[action][oldState] += self.learner * (utility - self.q)
        self.q = self.rewards[action][self.state]
        self.total_reward += reward
        # TODO: Learn policy based on state, action, reward
        #print("LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}"
         #     .format(deadline, inputs, action, reward))  # [debug]

    def best_action(self):
        chosen = None
        estimate = -100
        for action in self.actions:
            if self.rewards[action][self.state] > estimate:
                chosen = action
                estimate = self.rewards[action][self.state]
        return chosen, estimate

    def define_state(self):
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        self.state = (inputs['light'], inputs['oncoming'], inputs['right'], inputs['left'], self.next_waypoint)


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    trips = a.trips
    totals = map(lambda x: x['reward'], trips)
    moves = map(lambda x: x['actions'], trips)
    fixed = map(lambda x: x['reward']/float(max(1, x['actions'])), trips)
    print sum(totals) / len(totals)
    #sns.distplot(x='actions', y='reward', data=trips)
    #sns.regplot(x=totals, y=moves)
    sns.barplot(x=range(len(totals)), y=totals)
    sns.plt.show()


if __name__ == '__main__':
    run()

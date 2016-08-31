import random
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
        self.learner = 0.8
        self.discount = 0.9
        self.q = 0
        #state/action
        self.actions = [None, 'forward', 'left', 'right']
        self.rewards = {}
        self.init_states()
        self.runs = 0
        # Measurements
        self.trips = [] #contains each trip data
        self.errors = [] #magnitude of errors per trip
        self.moves = 0 #number of moves per trip
        self.correct = 0 #number of moves in the correct direction
        self.changes = 0 #number of times position changed
        self.acceptable = 0 #count of non-negative moves per trip
        self.total_reward = 0 #total reward (positive and negative)

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
                self.rewards[action][state] = random.uniform(-1, 0)

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.log_trial()
        self.total_reward = 0
        self.acceptable = 0
        self.moves = 0
        self.correct = 0
        self.changes = 0
        self.errors = []

    def update(self, t):
        self.define_state()
        old_state = self.state
        # choose action
        action, estimate = self.best_action()
        way = self.next_waypoint
        reward = self.env.act(self, action)
        # Get next state
        self.define_state()
        utility = reward + self.discount * estimate
        if self.location == self.planner.destination:
            self.rewards[action][old_state] = reward
        else:
            self.rewards[action][old_state] += self.learner * utility# * (utility - self.q)
        self.q = self.rewards[action][self.state]
        self.log_move(reward, action, way)

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
        self.location = self.env.agent_states[self]['location']
        #self.state = (inputs['light'], traffic, self.next_waypoint)
        #self.state = (inputs['light'], inputs['oncoming'], inputs['right'], inputs['left'], self.next_waypoint)

    def log_move(self, reward, action, direction):
        if reward < 0:
            self.errors.append(reward)
        else:
            self.acceptable += 1
            if direction == action:
                self.correct += 1
        if action != None:
            self.changes += 1
        self.total_reward += reward
        self.moves += 1

    def log_trial(self):
        trip = {
            'reward': self.total_reward,
            'moves': self.moves,
            'errors': sum(self.errors),
            'acceptable': self.acceptable/float(max(1, self.moves)),
            'wrong': 1.0-(self.acceptable/float(max(1, self.moves))),
            'right': (self.correct/float(max(1, self.changes))),
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
    #sim = Simulator(e, update_delay=0.5, display=True)
    sim.run(n_trials=250)  # run for a specified number of trials
    #data
    visualize_data(['reward', 'moves', 'errors', 'acceptable', 'wrong', 'right'], a.trips)


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

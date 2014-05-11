#!/opt/local/bin/python2.7

# An implementation of Q-learning and SARSA on the cliff walk environment based 
# on [Sutton & Barto, 1998]'s exmaple 6.6.  
#
# Written by:
# github.com/jvs--          May 2014


import numpy as np
import matplotlib.pyplot as plt
import random
import pprint 

################################################################################
###                           LEARNING METHODS                               ###
################################################################################

def Q_learning(agent, env, episodes, alpha=0.1, gamma=0.9):
    stats = [] # for book keeping how much reward was gained in each episode
    
    Q = init_Q(env.states) # Initialize Q arbitrarily 
    for episode in xrange(episodes):
        accumulated_reward = 0 # for book keeping of the reward
        s = env.start
        while not s in env.goals:
            # Choose a from s using Q, take action, observe reward, next state
            (a, s_, reward) = agent.take_action(env, s, Q)
            Q[s][a] = Q[s][a] + alpha * (reward + gamma * max(Q[s_].values()) - Q[s][a])
            s = s_
            accumulated_reward += reward
        print "FOUND GOAL!"
        
        stats.append((episode, accumulated_reward))
    return (Q, stats)

def SARSA(agent, env, episodes, alpha=0.1, gamma=0.9):
    stats = [] # for book keeping how much reward was gained in each episode
     
    Q = init_Q(env.states) #Initialize Q arbitrarily
    for episode in xrange(episodes):
        accumulated_reward = 0 # for book keeping of the reward
        s = env.start
        #Choose a from s using policy derived from Q (eg greedy) 
        a = agent.best_action(s, Q)
        while not s in env.goals:
            #Take action a and observe r, s' 
            (a, s_, reward) = agent.take_action(env, s, Q)
            a_ = agent.best_action(s_, Q)
            #Choose a' form s' using policy derived from Q (eg greedy)
            Q[s][a] = Q[s][a] + alpha * (reward + gamma * Q[s_][a_] - Q[s][a])
            s = s_
            a = a_
            accumulated_reward += reward
        print "FOUND GOAL!"
        
        stats.append((episode, accumulated_reward))
    return (Q, stats)


### Utility functions
def init_Q(states):
    Q = {}
    for state in states:
            Q[state] = {'down': 0, 'up': 0, 'left': 0, 'right': 0} 
    return Q

def states_from_grid(grid):
    states = []
    for i, lst in enumerate(grid):
        for j, entry in enumerate(lst):
            if grid[i][j] != "#":
                states.append((i,j))
    return states


### Class definitions
class Experiment:
    """Runs agent on environment and displays reward statistics.
    
    Arguments:
    env -- the environment to run on
    agent -- the agent
    nr_episodes --- how many episodes to run"""
    def __init__(self, env, agent, nr_episodes):
        self.env = env
        self.agent = agent
        self.episodes = nr_episodes
    
    def run(self):
        # Let the agent learn on the provided environment
        (Q, stats) = agent.learn(self.env, self.episodes)
        
        # Print the learned Q function and reward statistics 
        # Uncomment if you don't wanna see the clutter
        pp = pprint.PrettyPrinter(indent=2)
        pp.pprint(stats)
        pp.pprint(Q)
        
        # Plot reward statistics
        plt.plot(stats)
        plt.show()


class Env:
    """The environment modeled as a Markov decision process. We 
    assume a perfect model of the world. Thus states, actions, transition
    function and immediate reward function are known.
    
    Arguments:
    states -- states of the mdp
    goals -- a subset of the states defining which are goals states
    actions -- possible actions
    transitions -- P(s'|s,a) a function that defines transitionprobabilities
    reward -- r(s,a,s') a function defining the immediate reward
    """
    def __init__(self, grid, states, start, terminals, cliff):
        self.grid = grid
        self.states = states
        self.start = start 
        self.goals = terminals
        self.cliff = cliff

                   
    def show(self):
        for line in self.grid:
            print line
        print ""
        
    def on_grid(self, state):
        # ** In this implementation we should never even get into a stituation
        # where requested states are outside the dimensions of the grid
        #print "Is ", str(state), " on the grid?"
        assert ( (state[0] < len(grid) and state[1] < len(grid[0]) ) and 
                 (state[0] >= 0 and state[1] >= 0) )
                
        # Since we start from inside the grid walls define the boundaries of the
        # world and are not walkable. 
        if self.grid[state[0]][state[1]] == "#":
            #print "no."
            return False
        else:
            #print "yes."
            return True
        
    def R(self, state):
        """Reward function for the cliff environment."""
        if state in self.goals:
            return 100
        elif state in cliff:
            return -100
        else:
            return -1
    
    def T(self, state, action):
        """A transition-function for the cliff environment. Substitute your 
        own for other environments and tasks."""
        # Otherwise same as normal grid transitions
        if not self.on_grid(state):
            log = str(state) + " is not a valid position on the grid."
            raise Exception(log)
            
        if state in cliff:
             # Reset to start state when fallen into cliff
            return self.start 
            
        direction = action
        if direction == "up":
            newpos = (state[0]-1, state[1])
        elif direction == "down":
            newpos = (state[0]+1, state[1])
        elif direction == "left":
            newpos = (state[0], state[1]-1)
        elif direction == "right":
            newpos = (state[0], state[1]+1)
        else:
            log = str(direction) + " is not a valid direction."
            raise Exception(log) 
        
        # Show attempted move
        #print "Try moving from ", str(state), direction," to ", newpos 
        
        if self.on_grid(newpos):
            new_state = newpos
        else:
            new_state = state # if invalid just stay were you are 

        return new_state

class Agent:
    """An agent with eps greedy action selection and learns using a supplied 
    learning method
    
    Arguments:
    actions -- possible actions the agent can take
    learning_method -- learning method (eg. Q-learning)
    """
    def __init__(self, actions, learning_method):
        self.actions = actions
        self.state = ()
        self.Q = []
        self.learning_method = learning_method
        
    def learn(self, env, episodes):
        return self.learning_method(self, env, episodes)
                   
    def take_action(self, env, state, Q):
        # Given the current state select an action according to Q
        # and try to transition using this action and observe the new state
        #action = self.select_random() # random action selection
        action = self.select_epsilon_greedy(state, Q) # greedy action selection
        new_state = env.T(state, action)
        print action, " ", new_state
        self.state = new_state
        #return the new state and reward receives from the environment
        return (action, new_state, env.R(new_state))
    
    ###  Action selection methods ###
    def select_epsilon_greedy(self, state, Q, eps=0.1):
        if random.random() < eps:
            action = random.choice(self.actions) # With prob. eps choose random
        else:
            action = self.best_action(state, Q) # Otherwise choose best action
        return action
        
    def select_random(self):
        return random.choice(self.actions) # random action selection for testing

    def best_action(self, state, Q):
        best = 0
        best_action = random.choice(self.actions) # init randomly
        for action in Q[state]:
            curr =  Q[state][action]
            if curr > best:
                best = curr
                best_action = action
        return best_action


if __name__ == '__main__':
    ### Set up environment, agent and experiment to run###
    episodes = 500 # Number of episodes to run experiment 
    actions = ['up', 'down', 'left', 'right'] # Possible actions to take
    
    # Defining a cliff world
    # This grid is moslty for visualisation on commandline except for the 
    # walls '#' which are actually used to test world boundaries but they could 
    # as well be defines down below as an extra variable like goals and cliff.       
    grid = [['#','#','#','#','#','#','#','#','#','#','#','#','#','#'],
            ['#',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ','#'],
            ['#',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ','#'],
            ['#',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ','#'],
            ['#','A','U','U','U','U','U','U','U','U','U','U','G','#'],
            ['#','#','#','#','#','#','#','#','#','#','#','#','#','#']]
            
    states = states_from_grid(grid)
    goals = [(4, 12)]
    start = (4,1)
    cliff = [(4,2), (4,3), (4,4), (4,5), (4,6), (4,7), (4,8), (4,9), (4, 10), (4,11)]
    
    # Init environment 
    env = Env(grid, states, start, goals, cliff)
    env.show() # Let's take a look at our world
    
    ### Init agent with Q_learning as learning strategy and run experiment ###
    agent = Agent(actions, Q_learning) 
    experiment = Experiment(env, agent, episodes)
    experiment.run()
    
    ### Set up agent with SARSA as learning strategy and run experiment ###
    agent = Agent(actions, SARSA)  
    experiment = Experiment(env, agent, episodes)
    experiment.run()

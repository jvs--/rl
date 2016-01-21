#!/opt/local/bin/python2.7

# An implementation of Temporal Difference control algorithms SARSA and 
# Q-learning to illustrate model-free reinforcment learning based on 
# [Sutton & Barto, 1998]'s cliff walk example (Example 6.6).
#
# Written by:
# github.com/jvs--          May 2014


import numpy as np
import matplotlib.pyplot as plt
import random


class Env:
    """The environment modeled as a Markov decision process. This means we 
    assume a perfect model of the world. Including states, actions, transition
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
    def __init__(self, actions):
        self.actions = actions
        self.state = ()
        self.Q = []
        self.pi = []
           
    def take_action(self, env, state, Q):
        # given the current state we are in select an action according to Q
        # and try to transition using this action and observe the new state
        #action = self.select_random()
        action = self.select_greedy(state, Q)
        new_state = env.T(state, action)
        print action, " ", new_state
        self.state = new_state
        #return the new state and reward receives from the environment
        return (action, new_state, env.R(new_state))
    
    def select_random(self):
        # random action selection for now
        return random.choice(self.actions)
    
    def select_greedy(self, state, Q):
        best = 0
        best_action = random.choice(self.actions) # random as long as you don't know better
        for action in Q[state]:
            #print "foo ", action
            curr =  Q[state][action]
            #print "bar ", curr
            if curr > best:
                best = curr
                best_action = action
        return best_action
                
class Experiment:
    def __init__(self, env, agent, nr_episodes, learning_method):
        self.env = env
        self.agent = agent
        self.episodes = nr_episodes
        self.learning_method = learning_method
    
    def run(self):
        #Run the experiment and display results 
        (Q, stats) = self.learning_method(self.env, self.agent, self.episodes)
        print Q
        print stats
        plt.plot(stats)
        plt.show()
        #show_gridrun(Q)
    
# Obsolete?? 
def states_from_grid(grid):
    states = []
    for i, lst in enumerate(grid):
        for j, entry in enumerate(lst):
            if grid[i][j] != "#":
                states.append((i,j))
    return states

         
### LEARING METHODS AHEAD ###---------------------------------------------------

def TD0(episodes, V, pi, alpha=0.1, gamma=0.9):
    #Initialize V(s) arbitrarily, pi..
    for episode in episodes:
        #init s
        while True:
            action = pi[s]
            #take action
            (reward, s_) = agent.take_action(action)
            #observe reward r and next state s'
            # Update value function
            V[s] = V[s] + alpha * (reward + gamma * V[s_] - V[s])
            s = s_
            if s in goals:
                return V
                
                
def SARSA(env, agent, episodes, alpha=0.1, gamma=0.9):
    stats = [] # for book keeping how much reward was gained in each episode
     
    Q = init_Q(env.states) #Initialize Q arbitrarily
    for episode in xrange(episodes):
        accumulated_reward = 0 # for book keeping of the reward
        s = env.start
        a = agent.select_greedy(s, Q)#Choose a from s using policy derived from Q (eg greedy) 
        while not s in env.goals:
            #Take action a and observe r, s' 
            (a, s_, reward) = agent.take_action(env, s, Q)
            a_ = agent.select_greedy(s_, Q)
            #Choose a' form s' using policy derived from Q (eg greedy)
            Q[s][a] = Q[s][a] + alpha * (reward + gamma * Q[s_][a_] - Q[s][a])
            s = s_
            a = a_
            accumulated_reward += reward
        print "FOUND GOAL!"
        stats.append((episode, accumulated_reward))
    return (Q, stats)

def Q_learning(env, agent, episodes, alpha=0.1, gamma=0.9):
    stats = [] # for book keeping how much reward was gained in each episode
    
    Q = init_Q(env.states) #Initialize Q arbitrarily 
    for episode in xrange(episodes):
        accumulated_reward = 0 # for book keeping of the reward
        s = env.start
        while not s in env.goals:
            #Choose a from s using Q, take action, observe reward, next state
            (a, s_, reward) = agent.take_action(env, s, Q) 
            #print "Q ", Q[s][a]
            #print "Q[s'] ", Q[s_]
            #print "max Q[s'] ", max(Q[s_].values())  
            Q[s][a] = Q[s][a] + alpha * (reward + gamma * max(Q[s_].values()) - Q[s][a])
            s = s_
            accumulated_reward += reward
        print "FOUND GOAL!"
        stats.append((episode, accumulated_reward))
    return (Q, stats)
    
    
def init_Q(states):
    Q = {}
    for state in states:
            Q[state] = {'down': 0, 'up': 0, 'left': 0, 'right': 0} 
    return Q
    
# Action selection methods
def epsilon_greedy(Q, state, eps,):
    actions = ['up', 'down', 'left', 'right']
    if random.random() < eps:
        action = random.choice(actions) # With prob. eps choose random
    else:
        action = np.argmax(Q) # Otherwise choose the best action
    return action
             
if __name__ == '__main__':
    ### Set up environment ###
    
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

    env = Env(grid, states, start, goals, cliff)
    env.show() # Let's take a look at our world
    
    ### Set up agent with possible actions and a learning_method ###
    actions = ['up', 'down', 'left', 'right']
    agent = Agent(actions) 
    
    ### Run experiment with this agent in the cliff environment ###
    episodes = 500
    experiment = Experiment(env, agent, episodes, SARSA)
    experiment.run()
    
    episodes = 500
    experiment = Experiment(env, agent, episodes, Q_learning)
    experiment.run()

#!/opt/local/bin/python2.7

# An implementation of policy iteration and value iteration running on a
# markov decission process representing the grid world form example 4.1
# in [Sutton & Barto, 1998] chapter 4. The implementation is direclty based 
# on Figure 4.3 and 4.5
#
# Written by:
# github.com/jvs--          April 2014
# 
#
# Policy iteration
# Value interation 

import numpy as np
import matplotlib.pyplot as plt
import random

    
### Utility functions
def states_from_grid(grid):
    states = []
    for i, lst in enumerate(grid):
        for j, entry in enumerate(lst):
            if grid[i][j] != "#":
                states.append((i,j))
    return states


### Class definitions 
class Env:
    """The environment modeled as a Markov decision process. This means we 
    assume a perfect model of the world. Including states, actions, transition
    function and immediate reward function are known.
    
    Arguments:
    states -- states of the mdp
    goals -- a subset of the states defining which are goals states
    actions -- possible actions
    transitions -- P(s'|s,a) a function that defines transitionsprobabilities
    reward -- r(s,a,s') a function defining the immediate reward
    """
    def __init__(self, grid, states, goals, actions):
        self.states = states
        self.actions = actions # possible actions
        self.goals = goals
        #self.transitions = transitions# P(s'|s,a)
        #self.reward = reward # r(s,a,s')
        self.grid = grid
    
        
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
        """A reward function for this particular environment. Substitute your own for 
        other environments and tasks."""
        if state in [(3,3)]:
            return 100
        else:
            return -1
    
    def T(self, state, action):
        """A transitionsfuntion for this particular environment. Substitute your own
        for other environments and tasks."""
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
            log = direction + "is not a valid direction."
            raise Exception(log) 
        
        print "Try moving", direction,"..." # Show attempted move
        print state
        print direction
        print newpos
        
        if self.on_grid(newpos):
            new_state = newpos
        else:
            new_state = state # if invalid just stay were here you are
        
        return [(1.0, new_state)]
    
    def show(self):
        for line in self.grid:
            print line
        print ""
        
        
        

class Agent:
    def __init__(self, pi, actions, goals, transitions, reward):
        self.state = start_state
        self.actions = actions
        self.V = V
        self.pi = policy
        
    def take_action(self, state):
        # given a state received by the environment select an action according
        # your knowledge of the world (policy)
        pass
        # select action to take
        # 
    def solve(self):
        pass
        #return (value_function, policy)
                
class Experiment:
    def __init__(self, env, agent, nr_episodes, start_state):
        self.env = env
        self.agent = agent
        nr_episodes = nr_episodes
        self.agent.state = start_state  # place the agent at its initial spot 
    
    def run():
        for i in xrange(self.nr_episodes):
            agent.take_action()
            #env.

### RL methods for the agent to solve the problem with
### Policy Iteration
def policy_iteration(mdp):
    states, actions = mdp.states, mdp.actions
    # Initalize valuefunction arbitarily (here each state is mapped to 0)
    V = dict([(s, 0) for s in states])
    # Initialize policy arbitrarily (here random action choice)
    pi = dict([(s, random.choice(actions)) for s in states])
    #print pi
    #print V
    
    ### As long as policy is not stable continue to iterate between 
    ### evaluation and improvement of the policy
    stable = False
    while not stable:
        V = policy_evaluation(mdp, V, pi)
        #print V
        updated_pi = policy_improvement(mdp, V, pi)
        #print updated_pi
        
        if updated_pi == pi:
            stable = True
        pi = updated_pi # swap old policy for the new one
        
    return (V, pi)

def policy_evaluation(mdp, V, pi, theta=0.01, gamma=0.9):  
    states, reward, transitions= mdp.states, mdp.R, mdp.T 
    ### Policy evaluation part##
    #delta = 0
    while True:
        delta = 0 
        for s in states:
            v = V[s]
            print "reward ", reward(s)
            # Note that in [Sutton and Barto] reward(s) is inside the sum
            V[s] = reward(s) + gamma * sum(p * V[s_] for (p, s_) in transitions(s, pi[s]))### Update rule
            print "v ", v
            print "V(s) ", V[s]
            delta = max(delta, abs(v - V[s]))
            print "delta ", delta
            #i = i + 1
            #if i > 10:
            #    return V
            if delta < theta:
                print "eval done"
                return V

def policy_improvement(mdp, V, pi, gamma=0.9):
    states, reward, transitions, actions = mdp.states, mdp.R, mdp.T, ['up', 'down', 'left', 'right']
    ### Policy improvement part##
    for s in states:
        action_values = {}
        for a in actions:
            action_values[a] = sum(p * V[s_] for (p, s_) in transitions(s, a))
        print "action_values ", action_values
        # select the action that has yields the biggest reward acording to V
        pi[s] = max(action_values, key=action_values.get)
        print "pi[s]", pi[s]
    print "improve done"
    return pi

### Value Iteration
def value_iteration(env, gamma=0.9):
    actions = ['up', 'down', 'left', 'right']
    # Initalize valuefunction arbitarily (here each state is mapped to 0)
    V = dict([(s, 0) for s in states])
    delta = 0
    theta = 0.001
    # Perform value iteration
    while True:
        for s in states:
            v = V[s]
            V[s] =  env.R(s) + gamma * max(
            [sum([p * V[s_] for (p, s_) in env.T(s, a)]) for a in actions])
            delta = max(delta, abs(v - V[s]))
            print "delta: ", delta
        if delta < theta:
            break
    # Select best policy 
    # for arg max r + gama * doesn't matter as it's the same for all s
    pi[s] = np.argmax(
    [sum([p * V[s_] for (p, s_) in env.T(s, a)]) for a in actions])    
    
    return (value_function, policy)

def best_action(V, env):
    stuff = {}
    for a in actions:
        stuff[a]
        s_ = env.T(s, a)
        #(env.R(s_) + gamma V[s_])
    
       
if __name__ == "__main__":
    # Setting up an environement 
    grid = [['#','#','#','#','#','#'],
            ['#',' ',' ',' ',' ','#'],
            ['#',' ','#','#',' ','#'],
            ['#',' ',' ',' ',' ','#'],
            ['#',' ',' ',' ',' ','#'],
            ['#','#','#','#','#','#']]
    states = states_from_grid(grid)
    goals = [(0,0), (3,3)] 
    actions = ['up', 'down', 'left', 'right']
    #print "grid ", grid[-1][0]
    #print states
    
    env = Env(grid, states, goals, actions)
    (V, pi) = policy_iteration(env)
    #(V, pi) = value_iteration(env)
    # print V
    for entry in V:
        grid[entry[0]][entry[1]] = V[entry]
    env.grid = grid
    env.show()
    # print pi
    for entry in pi:
        grid[entry[0]][entry[1]] = pi[entry]
    env.grid = grid
    env.show()


    

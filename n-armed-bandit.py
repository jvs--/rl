#!/opt/local/bin/python2.7

# Implementation of action selection methods on an n-armed bandit based on
# [Sutton & Barto, 1998] Chapter 2.
# This code reproduces Fig 2.1, 2.4 and 2.5 from the textbook.
#
# Methods included here:
# 1. epsilon-greedy 
# 2. optimistic initial values
# 3. softmax

import numpy as np
import matplotlib.pyplot as plt
import random 


class NarmedBandit:
    """Am n-armed bandit with a random distribution of reward for each arm"""
    def __init__(self, nr_arms,):
        self.nr_arms = nr_arms # number of arms
        # True reward 
        self.reward = np.random.multivariate_normal(np.zeros((nr_arms)), 
            np.eye(nr_arms))
        self.bestaction = np.argmax(self.reward)

    def choose_arm(self, x):
        # Returns a reward for an arm based on the true reward but with noise
        sigma = 1
        received_reward = self.reward[x] + sigma * np.random.normal(0,1)
        return received_reward

# Action selection methods
def epsilon_greedy(Q, nr_actions, eps,):
    if random.random() < eps:
        action = random.randint(0, nr_actions - 1) # With prob. eps choose random
    else:
        action = np.argmax(Q) # Otherwise choose the best action
        #print "Action choosen ", action
    return action

def softmax(Q, nr_actions, temp):   
    # Calculate softmax probabilities using Gibbs
    eqt = np.exp(Q/temp)
    prob = eqt / sum(eqt)
    
    #print "Q ", Q
    #print "Probability of actions: ", prob
    
    action = np.argmax(Q) # Choose best action
    
    # Choose action acording to their probability
    r = np.random.rand()
    for index, p in enumerate(prob):
        if (r - p) <= 0:
            action = index
            
    return action

def reinforcement_comparison(Q):
    pass

def test_suit(nr_bandits, nr_arms, nr_plays, selection_method, eps, opt):
    # Initialize 
    bandits = []
    Q_star = []
    # Generate bandits
    for i in xrange(nr_bandits):
        bandits.append(NarmedBandit(nr_arms))   
        Q_star.append(bandits[i].reward)
    Q_t0 = np.zeros(np.shape(Q_star)) # Initially reward function all 0
    Q_t = Q_t0
    # Initialize with optimistic reward values if called for
    if opt:
        for i in xrange(len(Q_t)):
            for j in xrange(len(Q_t[1])):
                Q_t[i][j] = 5
    k = np.zeros(nr_arms) # for keeping track of how often an arm was selected
    mean_rewards = [] 
    mean_maxaction = []

    # Run simulations
    for pi in xrange(nr_plays):
        rewards_ith_play = []
        ith_play_maxactionselected = []
        for bi in xrange(nr_bandits):
            # Select an action a
            a = selection_method(Q_t[bi], nr_arms, eps)
            if a == bandits[bi].bestaction:
                ith_play_maxactionselected.append(1.0)
            else:
                ith_play_maxactionselected.append(0.0)

            k[a] += 1 # Keep trac of how often you have choosen this action 
            # Estimate reward = all rewards received / times received
            # Incremental version (from equation 2.4)
            received_reward = bandits[bi].choose_arm(a)
            # Update rule
            Q_t[bi][a] = Q_t[bi][a] + 0.1 * (received_reward - Q_t[bi][a])
            
            # TODO:
            # Some issues here with the update rule
            # Not sure what causes problems with this version
            #Q_t[bi][a] = Q_t[bi][a] + 1./(k[a]+1) * (received_reward - Q_t[bi][a])

            rewards_ith_play.append(received_reward)
        #print rewards_ith_play
        ith_play_mean = np.mean(rewards_ith_play)
        mean_maxaction.append(np.mean(ith_play_maxactionselected))
        #print mean_maxaction
        #print "mean of play nr", pi, ": ", mean
        mean_rewards.append(ith_play_mean)

    return (mean_rewards, mean_maxaction)

######################################################
###### Running experiments and plotting results ######
######################################################

def run_optimistic():
    # Run simulations
    print "Running... GREEDY OPTIMISTICALLY"
    avg1, perc1 = test_suit(500, 10, 1000, epsilon_greedy, 0, 1) 
    print "Running... EPSILON-GREEDY WITH EPS = 0.1"
    avg2, perc2 = test_suit(500, 10, 1000, epsilon_greedy, 0.1, 0) 
    data1 = [avg1, avg2]
    data2 = [perc1, perc2]

    # Plot results
    list_label = ["optimistic", "greedy"]
    # Plotting average reward
    x = np.linspace(0, 1000, 1000)
    for i in xrange(len(data1)):
        plt.plot(x, data1[i])
    plt.xlabel('Plays')
    plt.ylabel('Average Rewards')
    plt.legend(list_label)
    plt.show()
    
    # Plotting percentage optimal action 
    for i in xrange(len(data2)):
        plt.plot(x, data2[i]);

    plt.xlabel('Plays')
    plt.ylabel('Optimal Actions (%)')
    plt.legend(list_label);
    plt.show()
    
def run_greedy():
    # Run simulations
    print "Running... EPSILON-GREEDY WITH EPS = 0.1"
    avg1, perc1 = test_suit(2000, 10, 1000, epsilon_greedy, 0.1, 0) 
    print "Running... EPSILON-GREEDY WITH EPS = 0.01"
    avg2, perc2 = test_suit(2000, 10, 1000, epsilon_greedy, 0.01, 0) 
    print "Running... EPSILON-GREEDY WITH EPS = 0.0"
    avg3, perc3 = test_suit(2000, 10, 1000, epsilon_greedy, 0.0, 0) 
    
    averages = [avg1, avg2, avg3]
    percentages = [perc1, perc2, perc3]

    # Plot results
    plot_label = "eps="
    list_label = []
    eps = [0.1, 0.01, 0]
    for i in eps:
        list_label.append(plot_label + str(i))

    # Plotting for average reward
    x = np.linspace(0, 1000, 1000)
    for i in xrange(len(eps)):
        plt.plot(x, averages[i])

    plt.xlabel('Plays')
    plt.ylabel('Average Rewards')
    plt.legend(list_label)
    plt.show()
    
    # Plotting percentage optimal action 
    for i in xrange(len(percentages)):
        plt.plot(x, percentages[i]);

    plt.xlabel('Plays')
    plt.ylabel('Optimal Actions (%)')
    plt.legend(list_label);
    plt.show()
    
def run_softmax():
    # Try runing softmax with different temperatures and plot
    print "Running... SOFTMAX"
    avg1, perc1 = test_suit(2000, 10, 1000, softmax, 0.3, 0)
    avg2, perc2 = test_suit(2000, 10, 1000, softmax, 0.5, 0)
    avg3, perc3 = test_suit(2000, 10, 1000, softmax, 0.1, 0)
    avg4, perc4 = test_suit(2000, 10, 1000, softmax, 0.01, 0)
    avg5, perc5 = test_suit(2000, 10, 1000, softmax, 1, 0)
    avg6, perc6 = test_suit(2000, 10, 1000, softmax, 2, 0)
    
    
    averages = [avg1, avg2, avg3, avg4, avg5, avg6]
    percentages = [perc1, perc2, perc3, perc4, perc5, perc6]
    
    # Plot results
    plot_label = "temp="
    list_label = []
    temp = [0.3, 0.5, 0.1, 0.01, 1, 2]
    for i in temp:
        list_label.append(plot_label + str(i))

    # Plotting for average reward
    x = np.linspace(0, 1000, 1000)
    for i in xrange(len(averages)):
        plt.plot(x, averages[i])

    plt.xlabel('Plays')
    plt.ylabel('Average Rewards')
    plt.legend(list_label)
    plt.show()
    
    # Plotting percentage optimal action 
    for i in xrange(len(percentages)):
        plt.plot(x, percentages[i]);

    plt.xlabel('Plays')
    plt.ylabel('Optimal Actions (%)')
    plt.legend(list_label);
    plt.show()
    

if __name__ == "__main__":
    # Run simulations and plot results 
    run_greedy()
    #run_optimistic()
    #run_softmax()

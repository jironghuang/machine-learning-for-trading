"""
An algorithm for a QLearner  (c) 2017 Arjun Joshua
"""

import numpy as np
import random as rand

class QLearner(object):


    def __init__(self, \

                 
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        # initialize Q-table with random numbers between -1 and 1
        Q = np.random.uniform(-1, 1, (num_states, num_actions))
        # initialize counting transition matrix for dyna-Q part of learner
        Tc = 0.0001 * np.ones((num_states, num_actions, num_states))
        # initialize dyna-Q's transition matrix model
        T = np.zeros((num_states, num_actions, num_states))
        # initialize dyna-Q's reward function model
        R = np.zeros((num_states, num_actions)) # reward matrix
        self.Q = Q
        self.Tc = Tc
        self.T = T
        self.R = R
        self.verbose = verbose
        self.num_actions = num_actions # number of possible actions
        self.num_states = num_states # number of possible states
        self.s = 0 # state
        self.a = 0 # action
        self.alpha = alpha # learning rate in the range 0 - 1
        self.gamma = gamma # inverse discount rate in the range 0 - 1. Low gamma means future returns are significantly devalued
        self.rar = rar # random action rate, the probability to take a random action as opposed to consulting the Q-table policy
        self.radr = radr # random action decay rate. 
        self.dyna = dyna # number of iterations of the dyna loop

    def querysetstate(self, s):
        """
        @summary: Take in a state and return an action without updating the 
                  Q-table. Used to get the very first action when there is
                  no reward information available
        @param s: integer, the new state
        @returns: integer a, the selected action
        """
 
        
        if rand.random() < self.rar: # with probability rar, execute next statement
            a = rand.randrange(self.num_actions) # pick a random action
        else:
            a = self.Q[s, :].argmax() # select action that maximizes reward
        
        if self.verbose: print "s =", s,"a =",a
        
        self.s = s
        self.a = a
        return a

    def query(self,s_prime,r):
        """
        @summary: Update the Q table given s', r (and s, a from the previous
                  interaction with the environment) and return the next 
                  action, a'. If dyna>0, implement dyna-Q
        @param s_prime: integer, the new state
        @param r: int/float, reward for selecting the previous action a
        @returns: The next action, a'
        """
        
        
        # Import internal attributes of QLearner object into method.
        # This speeds up computation rather than referring to internal 
        # attributes via self.attribute. At the end of this method, the 
        # modified attributes are exported into the QLearner objects own
        # attributes
        num_states = self.num_states
        num_actions = self.num_actions
        s = self.s
        a = self.a
        alpha = self.alpha
        gamma = self.gamma
        Tc = self.Tc
        T = self.T
        R = self.R
        Q = self.Q
        
        # update policy with a blend of existing policy Q[s,a], present 
        # reward r and expected discounted future reward Q[s',a']. Blend 
        # controlled by alpha
        Q[s, a] = (1 - alpha) * Q[s, a] + alpha * (r + \
            gamma * Q[s_prime, Q[s_prime, :].argmax()])
            
        if rand.random() < self.rar: # with probability rar, execute next statement
            a_prime = rand.randrange(num_actions) # pick a random action
        else: # Note! ndarray.argmax() turns out to be noticeably faster than np.argmax(ndarray) 
            a_prime = Q[s_prime, :].argmax() # select action that maximizes reward
        
        if self.verbose: print "s =", s_prime,"a =",a_prime,"r =",r
        
        if self.dyna > 0: # execute dyna-Q
            Tc[s, a, s_prime] += 1 # increment counting transition matrix
            norm = np.sum(Tc[s, a, :])  #  time-consuming step
            T[s, a, :] = Tc[s, a, :] / norm # update transition matrix elements, time-consuming step
            R[s, a] = (1 - alpha) * R[s, a] + alpha * r # update reward matrix, time-consuming step
            count = 0
            while count < self.dyna:
                s_sim = rand.randrange(num_states) # randomly choose state
                a_sim = rand.randrange(num_actions) # randomly choose action
                sprime_sim = T[s_sim, a_sim, :].argmax() # select next action, a' from transition matrix
                r_sim = R[s_sim, a_sim] # simulate reward from reward matrix
                # update Q-table policy as above, except now with simulated s, a, s' and r
                Q[s_sim, a_sim] = (1 - alpha) * Q[s_sim, a_sim] + \
                alpha * (r_sim + \
                    gamma * Q[sprime_sim, Q[sprime_sim, :].argmax()])
                count += 1
            
        # Export updated attributes to QLearner object.
        self.rar *= self.radr # reduce rar with decay rate, radr, for next real interaction with environment
        self.s = s_prime
        self.r = r
        self.a = a_prime
        self.Q = Q
        self.T = T
        self.R = R
        self.Tc = Tc
        return a_prime

if __name__=="__main__":
    print "QLearner.py implements a class designed to be called, not to be \
ran!"

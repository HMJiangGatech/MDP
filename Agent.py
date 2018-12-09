import numpy as np
import random
import math

class Agent:
    def __init__(self):
        pass

    def act(self, s):
        raise NotImplementedError("Function need to be implemented by sub class")

    def observe(self, s_old, a, r, s_new):
        raise NotImplementedError("Function need to be implemented by sub class")

    def reset(self):
        raise NotImplementedError("Function need to be implemented by sub class")

class RandomAgent(Agent):
    def __init__(self, num_act):
        self.num_act = num_act

    def act(self, s):
        return 1
        return random.randint(0,self.num_act-1)

    def observe(self, s_old, a, r, s_new):
        pass

    def reset(self):
        pass

class UCB1(Agent):
    def __init__(self, num_act, c=0.2):
        self.num_act = num_act
        self.mu = np.zeros(num_act)
        self.T = np.zeros(num_act, dtype=np.int)
        self.t = 0
        self.c = c

    def act(self, s):
        if self.t < self.num_act:
            return self.t

        return (self.mu + self.c*(math.log(self.t+1)/self.T)**0.5).argmax()


    def observe(self, s_old, a, r, s_new):
        self.mu[a] = self.mu[a]*(self.T[a]/(self.T[a]+1))+r/(self.T[a]+1)
        self.T[a] += 1
        self.t += 1

    def reset(self):
        pass

class Greedy(Agent):
    def __init__(self, num_act, eps=1.0, decay_steps = 10):
        self.num_act = num_act
        self.mu = np.zeros(num_act)
        self.T = np.zeros(num_act, dtype=np.int)
        self.t = 0
        self.eps = eps
        self.decay_steps = max(decay_steps, self.num_act*2)

    def act(self, s):
        if self.t < self.num_act:
            return self.t

        eps = self.eps * (self.decay_steps-self.t)/self.decay_steps
        if random.uniform(0,1) < eps:
            return random.randint(0,self.num_act-1)
        else:
            return self.mu.argmax()


    def observe(self, s_old, a, r, s_new):
        self.mu[a] = self.mu[a]*(self.T[a]/(self.T[a]+1))+r/(self.T[a]+1)
        self.T[a] += 1
        self.t += 1

    def reset(self):
        pass

class QLearning(Agent):
    """
    Tabula Q Learning Agent:
    gamma, discount reward coefficient
    """
    def __init__(self, num_state, num_act, lr=0.1, eps=1.0, decay_steps = 10, gamma = 0.95):
        self.num_act = num_act
        self.Q = np.zeros((num_state,num_act))
        self.eps = eps
        self.decay_steps = max(decay_steps, self.num_act*2)
        self.lr = lr
        self.gamma = gamma
        self.t = 0

    def act(self, s):
        eps = self.eps * (self.decay_steps-self.t)/self.decay_steps
        if random.uniform(0,1) < eps:
            return random.randint(0,self.num_act-1)
        else:
            return self.Q[s,:].argmax()


    def observe(self, s_old, a, r, s_new):
        self.Q[s_old,a] = (1-self.lr)*self.Q[s_old,a] + self.lr*(r+self.gamma*self.Q[s_new,:].max())
        self.t += 1


    def reset(self):
        pass

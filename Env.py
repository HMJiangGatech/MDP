import numpy as np
import random

class Environment:
    def __init__(self):
        pass

    def step(self,action):
        raise NotImplementedError("Function need to be implemented by sub class")

    def reset(self):
        raise NotImplementedError("Function need to be implemented by sub class")

class MDP(Environment):
    def __init__(self, num_state=None, num_act=None, trans_mat=None, reward_coef=None):
        self.num_state = random.randint(5,10) if num_state is None else num_state
        self.num_act = random.randint(5,10) if num_act is None else num_act
        if trans_mat is None:
            self.trans_mat = np.zeros((self.num_state,self.num_act,self.num_state))
            for i in range(self.num_state):
                for j in range(self.num_act):
                    self.trans_mat[i,j,:] = np.random.dirichlet([5]*self.num_state)
        else:
            self.trans_mat = trans_mat
        if reward_coef is None:
            self.reward_coef = np.random.normal(1,1,(self.num_state,self.num_state))
        else:
            self.reward_coef = reward_coef

        self.curr_state = random.randint(0,self.num_state-1)


    def step(self,action):
        new_state = np.random.choice(self.num_state,p=self.trans_mat[self.curr_state,action,:])
        reward = np.random.normal(self.reward_coef[self.curr_state,new_state],1)
        self.curr_state = new_state
        return new_state, reward

    def reset(self):
        self.curr_state = random.randint(0,self.num_state-1)

class MAB(Environment):
    """
    MULTI-ARMED BANDITS, Bernoulli Distribution
    """
    def __init__(self, num_arm=None, prob=None):
        if num_arm is None:
            self.num_arm = random.randint(5,50)
            self.prob = [random.uniform(0,1) for _ in range(self.num_arm)]
        else:
            self.num_arm = num_arm
            if prob is None:
                self.prob = [random.uniform(0,1) for _ in range(self.num_arm)]
            else:
                self.prob = prob
        self.num_act = self.num_arm

    def step(self,action):
        return None, 1 if random.uniform(0,1) <= self.prob[action] else 0

    def reset(self):
        return None

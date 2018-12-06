## Define Args
import argparse
parser = argparse.ArgumentParser(description='PyTorch MDP')


###### Environment Setting ######
parser.add_argument('-E', '--Environment', default='MAB_20', type=str)
parser.add_argument('--num_act', default=None, type=int, help='number of actions for MAB')
parser.add_argument('-N', default=1, type=int, help='number of episode')
parser.add_argument('--max_step', default=400, type=int, help='maximun number of steps for a episode')

###### Agent Setting ############
parser.add_argument('-A', '--Agent', default='RandomAgent', type=str)
###UBC1 Agent
parser.add_argument('--ucb1_c', default=0.2, type=float, help='tuning parameter of UCB1 agent')
###Greedy Agent
parser.add_argument('--greedy_eps', default=1.0, type=float, help='explore probability of Greedy agent')
parser.add_argument('--greedy_decay', default=350, type=int, help='decay steps of Greedy agent')

import Env
import Agent
import numpy as np
import time
import matplotlib.pyplot as plt

def buildenv(args):

    # Build Env
    if args.Environment == "MAB":
        env = Env.MAB(args.num_act)
    elif args.Environment.startswith("MAB_"):
        num_arms = [int(s) for s in args.Environment.split("_") if s.isdigit()]
        num_arms = num_arms[0]
        np.random.seed(num_arms)
        env = Env.MAB(num_arms, np.random.uniform(0,1,num_arms))
        np.random.seed(int(time.time()))
    elif args.Environment == "MDP":
        env = Env.MDP()
    elif args.Environment.startswith("MDP_"):
        nums = [int(s) for s in args.Environment.split("_") if s.isdigit()]
        np.random.seed(nums[0]*nums[1])
        env = Env.MDP(nums[0],nums[1])
        np.random.seed(int(time.time()))

    # Build Agent
    if args.Agent == "RandomAgent":
        agent = Agent.RandomAgent(env.num_act)
    elif args.Agent == "UCB1":
        agent = Agent.UCB1(env.num_act, args.ucb1_c)
    elif args.Agent == "Greedy":
        agent = Agent.Greedy(env.num_act, args.greedy_eps, args.greedy_decay)
    return env, agent


def main(args):
    env, agent = buildenv(args)
    all_reward = 0
    all_rewards = []
    for iter_ep in range(args.N):
        state = env.reset()
        agent.reset()
        episode_reward = 0
        for iter_step in range(args.max_step):
            action = agent.act(state)
            new_state, reward = env.step(action)
            episode_reward += reward
            all_rewards += [episode_reward]
            agent.observe(state, action, reward, new_state)
            state = new_state
        all_reward += episode_reward
    print("Total Reward: \t", all_reward)
    plt.plot(all_rewards)
    plt.show()



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

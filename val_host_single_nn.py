import os

import gymnasium as gym
from my_network import HostSingleNN
import numpy as np
import functools
from random import Random
from multiprocessing import Pool
import pickle
import matplotlib.pyplot as plt
import networkx as nx


def eval(x, render=False, episodes=100):
    cumulative_rewards = []
    task = gym.make("CartPole-v1", render_mode="human")
    agent = HostSingleNN([4, 4, 2], [4, 4, 1]) # cartPole
    #agent = HostNN([8, 4, 4], [2, 2, 1]) # lunarLander
    
    # load the weights
    agent.set_weights(x)   

    for i in range(episodes):
        cumulative_rewards.append(0)

        done = False
        obs, info = task.reset(seed=i, options={})
        counter = 0
        while not done:
            output = agent.activate(obs)

            obs, rew, terminated, truncated, info = task.step(np.argmax(output))
            # obs, rew, done, _ = task.step(np.argmax(output))
            cumulative_rewards[-1] += rew
            done = terminated or truncated    
        print(cumulative_rewards[-1])        
        counter += 1

    task.close()
       
    #return -sum(cumulative_rewards) / 100
    return cumulative_rewards, agent


if __name__ == "__main__":
    g = None
    
    with open("results_hostNN\host_single_nn_test_-135.59.pkl", "rb") as f:
    #with open("results_hostNN\LunarLander\host_nn_test1.pkl", "rb") as f:
        g = pickle.load(f)
    print(type(g))

    res, agent = eval(g, episodes=100)
    print(-sum(res) / 100)
    

    
import os
import gymnasium as gym
from hostNN import HostSingleNN, HostMultipleNN
from HNNhostNN import HNNHostSingleNN, HNNHostMultipleNN
import numpy as np
import functools
from random import Random
from multiprocessing import Pool
import pickle
import matplotlib.pyplot as plt
import networkx as nx

def eval(type, x, hostNodes, guestNodes, inputType, updateType = None, episodes=100):
    cumulative_rewards = []
    task = gym.make("CartPole-v1", render_mode="human")

    if type == "single":
        agent = HostSingleNN(hostNodes, guestNodes, inputType, updateType)
        agent.set_guest_weights(x) # load the weights
    elif type =="multiple":
        agent = HostMultipleNN(hostNodes, guestNodes, inputType, updateType)
        agent.set_weights(x) # load the weights
    elif type =="HNNsingle":
        agent = HNNHostSingleNN(hostNodes, guestNodes, inputType)
        agent.set_guest_weights(x)
    elif type =="HNNmultiple":
        agent = HNNHostMultipleNN(hostNodes, guestNodes, inputType)
        agent.set_guest_weights(x)
    
    for i in range(episodes):
        cumulative_rewards.append(0)

        done = False
        obs, info = task.reset(seed=i, options={})
        counter = 0
        while not done:
            output = agent.activate(obs)

            # since the guest are used to estimate an hebbian rule, we still need to update the weights in the host
            # while in the other case the weight are computed only at the beginnning
            if type == "HNNsingle" or type == "HNNmultiple":
                agent.update_weights()

            obs, rew, terminated, truncated, info = task.step(np.argmax(output))
            cumulative_rewards[-1] += rew
            done = terminated or truncated    
        print(cumulative_rewards[-1])        
        counter += 1

    task.close()
       
    #return -sum(cumulative_rewards) / 100
    return cumulative_rewards, agent


if __name__ == "__main__":
    g = None
    
    with open("pkl/host_single_HNNhostNN_-500.0.pkl", "rb") as f:
        g = pickle.load(f)
    print(type(g))
    print(g)

    if "updateType" in g:
        res, agent = eval(g["type"],      # single or multiple
                          g["candidate"], # weights
                          g["hostNodes"], 
                          g["guestNodes"], 
                          g["inputType"], 
                          g["updateType"], 
                          episodes=100)
    else:
        res, agent = eval(g["type"],      # single or multiple
                          g["candidate"], # weights
                          g["hostNodes"], 
                          g["guestNodes"], 
                          g["inputType"], 
                          episodes=100)
    
    # print the average reward
    print(-sum(res) / 100)
    

    
import os

from cma import CMAEvolutionStrategy as cmaes
import gym
from network5 import SBMD4R
import numpy as np
import functools
from random import Random
from multiprocessing import Pool
import pickle
import matplotlib.pyplot as plt
import networkx as nx
def eval(agent, render=False):

    cumulative_rewards = []
    task = gym.make("LunarLander-v2")

    for i in range(100):
        cumulative_rewards.append(0)
        done = False
        task.seed(i)
        obs = task.reset()
        # counter = 0
        while not done:
            output = agent.distance_activation(obs, i < 0)
            print((np.argmax(output), output, agent.activations))
            if render:
                task.render()
            obs, rew, done, _ = task.step(np.argmax(output))
            cumulative_rewards[-1] += rew
        # counter += 1
    return -np.mean(cumulative_rewards)

def normLL(obs):
    low = np.array(
        [
            # these are bounds for position
            # realistically the environment should have ended
            # long before we reach more than 50% outside
            -1.5,
            -1.5,
            # velocity bounds is 5x rated speed
            -5.0,
            -5.0,
            -3.14,
            -5.0,
            -0.0,
            -0.0,
        ]
    ).astype(np.float32)
    high = np.array(
        [
            # these are bounds for position
            # realistically the environment should have ended
            # long before we reach more than 50% outside
            1.5,
            1.5,
            # velocity bounds is 5x rated speed
            5.0,
            5.0,
            3.14,
            5.0,
            1.0,
            1.0,
        ]
    ).astype(np.float32)
    return np.array([2 * ((obs[i] - low[i]) / (high[i] - low[i])) - 1 for i in range(len(obs))])


def normMC(obs):
    low = [-1.2, -0.07]
    high = [0.6, 0.07]
    return np.array([((obs[i] - low[i]) / (high[i] - low[i])) for i in range(len(obs))])


def normCP(obs):
    low = [-2.4, -2, -0.2095, -1.5]
    high = [2.4, 2, 0.2095, 1.5]
    return np.array([((obs[i] - low[i]) / (high[i] - low[i])) for i in range(len(obs))])


def fromCPtoLL(obs):
    tmp = normCP(obs)
    assert len(tmp) == len(obs) and len(obs) == 4, "e che cazz "+str(obs)+"  "+str(tmp)
    low = np.array(
        [-1., -1., -5.0, -5.0, -3.14, -1.0, -0.0, -0.0]
    ).astype(np.float32)
    high = np.array(
        [1., 1., 5.0, 5.0, 3.14, 1.0, 1.0, 1.0, ]
    ).astype(np.float32)
    obs_norm = (high + low) / 2
    obs_norm[0] = (tmp[0] * (high[0] - low[0]) + low[0])
    obs_norm[2] = (tmp[1] * (high[2] - low[2]) + low[2])
    obs_norm[4] = (tmp[2] * (high[4] - low[4]) + low[4])
    obs_norm[5] = (tmp[3] * (high[5] - low[5]) + low[5])
    obs_norm[6] = 0
    obs_norm[7] = 0
    return obs_norm


def fromMCtoLL(obs):
    tmp = normMC(obs)

    low = np.array(
        [-1., -1., -5.0, -5.0, -3.14, -1.0, -0.0, -0.0]
    ).astype(np.float32)
    high = np.array(
        [1., 1., 5.0, 5.0, 3.14, 1.0, 1.0, 1.0, ]
    ).astype(np.float32)
    obs_norm = (high + low) / 2
    obs_norm[0] = -(tmp[0] * (high[0] - low[0]) + low[0])
    obs_norm[2] = -(tmp[1] * (high[2] - low[2]) + low[2])
    obs_norm[6] = 0
    obs_norm[7] = 0
    return obs_norm


def evalCP(ds):  # render=False, episodes=100, upds=20, task="", old=None, trl=False):
    x = ds[0]
    args = ds[1]
    render = args["render"]
    episodes = args["episodes"]
    ps = [args["ps"]]
    task = args["task"]
    old = args["old"]
    trl = args["trl"]
    hn = args["hn"]
    pr = args["pr"]
    seed = args["seed"]
    # print(type(hn))
    cumulative_rewards = []
    task = gym.make(task)
    agent = RNN([8, hn, 4], prune_ratio=pr, eta=0.01, seed=seed, random=False)
    agent.set_hrules(x)
    if not old is None:
        agent = old

    for i in range(episodes):

        cumulative_rewards.append(0)
        done = False
        task.seed(i)
        obs = task.reset()
        #print(trl)
        #print(task)
        while not done:
            if trl:
                obs = fromCPtoLL(obs)
            output = agent.activate(obs)

            # arr[output]+=1
            if render:
                task.render()

            if trl:
                output = np.array([output[1], output[3]])

            obs, rew, done, _ = task.step(np.argmax(output))
            cumulative_rewards[-1] += rew
            if i < max(ps):
                agent.update_weights()
            # counter += 1
    return cumulative_rewards, agent


def val_CP_procedure(ds):
    g = ds[0]
    args = ds[1]
    args["seed"] = int(ds[2])
    first_half = {}
    for k, v in args.items():
        first_half[k] = v
    first_half["episodes"] = args["ps"]
    fhf, agent = evalCP([g, first_half])

    agent.prune_weights(os.path.join("graphs", str(args["ps"]), str(args["pr"]), str(args['hn']), "cp_"+str(ds[2])+"_"))

    second = {}
    for k, v in args.items():
        second[k] = v

    second["ps"] = -1  # args["step"]
    second["episodes"] = 100 - args["ps"]
    second["old"] = agent

    shf, _ = evalCP([g, second])

    return (sum(fhf) + sum(shf)) / 100


def parallel_val_CP(candidates, keys):
    ps = int(keys.split("_")[0])
    prate = int(keys.split("_")[1])
    hn = int(keys.split("_")[2])

    args = {"episodes": 100, "ps": ps, "render": False, "task": "CartPole-v1", "trl": True, "old": None,
            "pr": prate, "hn": int(hn)}
    print(args)
    with Pool(10) as p:
        return p.map(val_CP_procedure, [[candidates[i], args, i] for i in range(len(candidates))])


def evalMC(ds):  # render=False, episodes=100, upds=20, task="", old=None, trl=False):
    x = ds[0]
    args = ds[1]
    render = args["render"]
    episodes = args["episodes"]
    ps = [args["ps"]]
    task = args["task"]
    old = args["old"]
    trl = args["trl"]
    seed = args["seed"]
    hn = args["hn"]
    pr = args["pr"]
    cumulative_rewards = []
    task = gym.make(task)
    agent = RNN([8, hn, 4], prune_ratio=pr, eta=0.01, seed=seed, random=False)
    agent.set_hrules(x)
    if not old is None:
        agent = old

    for i in range(episodes):

        cumulative_rewards.append(0)
        done = False
        task.seed(i)
        obs = task.reset()
        while not done:
            if trl:
                obs = fromMCtoLL(obs)
            output = agent.activate(obs)

            # arr[output]+=1
            if render:
                task.render()

            if trl:
                output = np.array([output[1], output[0], output[3]])

            obs, rew, done, _ = task.step(np.argmax(output))
            cumulative_rewards[-1] += rew
            if i < max(ps):
                agent.update_weights()
            # counter += 1

    return cumulative_rewards, agent


def val_MC_procedure(ds):
    g = ds[0]
    args = ds[1]
    args["seed"] = int(ds[2])
    first_half = {}

    for k, v in args.items():
        first_half[k] = v
    first_half["episodes"] = args["ps"]
    fhf, agent = evalMC([g, first_half])

    agent.prune_weights(os.path.join("graphs", str(args["ps"]), str(args["pr"]), str(args['hn']), "mc_"+str(ds[2])+"_"))

    second = {}
    for k, v in args.items():
        second[k] = v

    second["ps"] = -1  # args["step"]
    second["episodes"] = 100 - args["ps"]
    second["old"] = agent

    fhs, _ = evalMC([g, second])

    return (sum(fhf) + sum(fhs)) / 100


def parallel_val_MC(candidates, keys):
    ps = int(keys.split("_")[0])
    prate = int(keys.split("_")[1])
    hn = int(keys.split("_")[2])

    args = {"episodes": 100, "ps": ps, "render": False, "task": "MountainCar-v0", "trl": True,
            "old": None,
            "pr": prate, "hn": hn}
    with Pool(10) as p:
        return p.map(val_MC_procedure, [[candidates[i], args, i] for i in range(len(candidates))])


def load_bests(dirname):
    bests = list()
    for seedn in os.listdir(dirname):
        for fn in os.listdir(os.path.join(dirname, seedn)):
            if not "best" in fn and not "log" in fn:
                bests.append(-float(fn[:-4]))
    assert len(bests) > 4, "not enough runs!!! " + str(len(bests))
    return bests


def load_bests_genome(dirname):
    bests = list()
    for seedn in os.listdir(dirname):
        for fn in os.listdir(os.path.join(dirname, seedn)):
            if not "best" in fn and not "log" in fn:
                with open(os.path.join(dirname, seedn, fn), "rb") as f:
                    bests.append(pickle.load(f))

    assert len(bests) > 4, "not enough runs!!! " + str(len(bests))
    return bests


if __name__ == "__main__":
    x = pickle.load(open("best_319.pkl", "rb"))
    print(x)
    times = []
    from time import time
    for i in range(1000):
        a = SBMD4R([8,9,4], 0,0.001, 0, False)
        a.set_hrules(x)
        t = time()
        eval(a, False)
        times.append(time()-t)

    print((np.mean(times), np.std(times)))
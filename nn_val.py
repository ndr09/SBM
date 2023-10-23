import os

from cma import CMAEvolutionStrategy as cmaes
import gym
from network import RNN, NN, HNN
import numpy as np
import functools
from random import Random
from multiprocessing import Pool
import pickle
import matplotlib.pyplot as plt
import networkx as nx


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
    task = args["task"]
    old = args["old"]
    trl = args["trl"]
    hn = args["hn"]
    pr = args["pr"]
    cumulative_rewards = []
    task = gym.make(task)
    agent = NN([8, hn, 4])
    agent.set_weights(x)
    agent.nn_prune_weights(pr)
    if not old is None:
        agent = old

    for i in range(episodes):

        cumulative_rewards.append(0)
        done = False
        task.seed(i)
        obs = task.reset()
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
            # counter += 1
    return cumulative_rewards, agent


def val_CP_procedure(ds):
    g = ds[0]
    args = ds[1]
    args["seed"] = int(ds[2])
    first_half = {}
    for k, v in args.items():
        first_half[k] = v
    first_half["episodes"] = 100#args["ps"]

    fhf, agent = evalCP([g, first_half])
    return sum(fhf) / 100


def parallel_val_CP(candidates, keys):
    prate = int(keys.split("_")[0])
    hn = int(keys.split("_")[1])

    args = {"episodes": 100, "render": False, "task": "CartPole-v1", "trl": True, "old": None,
            "pr": prate, "hn": int(hn)}
    print(args)
    with Pool(10) as p:
        return p.map(val_CP_procedure, [[candidates[i], args, i] for i in range(len(candidates))])


def evalMC(ds):  # render=False, episodes=100, upds=20, task="", old=None, trl=False):
    x = ds[0]
    args = ds[1]
    render = args["render"]
    episodes = args["episodes"]
    task = args["task"]
    old = args["old"]
    trl = args["trl"]
    seed = args["seed"]
    hn = args["hn"]
    pr = args["pr"]
    cumulative_rewards = []
    task = gym.make(task)
    agent = NN([8, hn, 4])
    agent.set_weights(x)
    agent.nn_prune_weights(pr)
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
            # counter += 1

    return cumulative_rewards, agent


def val_MC_procedure(ds):
    g = ds[0]
    args = ds[1]
    args["seed"] = int(ds[2])
    first_half = {}

    for k, v in args.items():
        first_half[k] = v
    first_half["episodes"] = 100 #args["ps"]
    fhf, agent = evalMC([g, first_half])

    return (sum(fhf)) / 100


def parallel_val_MC(candidates, keys):
    prate = int(keys.split("_")[0])
    hn = int(keys.split("_")[1])

    args = {"episodes": 100, "render": False, "task": "MountainCar-v0", "trl": True,
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

    prunning_timestep = []
    prunning_rate = []
    hnodes = []
    for ps in range(0, 25, 5):
        prunning_timestep.append(ps)

    for prate in range(0, 100, 20):
        prunning_rate.append(prate)

    for hnode in range(5, 10):
        hnodes.append(hnode)

    data = {}

    for hn in hnodes:
        for prate in prunning_rate:
            data[str(prate) + "_" + str(hn)] = load_bests_genome(
                os.path.join("results_nn_all",  str(hn), str(prate))
            )


    data_CP = {}
    for keys in data.keys():
        fits = parallel_val_CP(data[keys], keys)
        # fits = [np.mean(r[0]) for r in res]
        data_CP[keys] = fits

    data_MC = {}
    for keys in data.keys():
        fits = parallel_val_MC(data[keys], keys)
        # fits = [np.mean(r[0]) for r in res]
        data_MC[keys] = fits

    with open("nn_mc_log.txt", "w") as f:
        for k in data_MC.keys():
            l = str(k) + ";"
            for v in data_MC[k]:
                l += str(v) + ";"
            f.write(l[:-1] + "\n")

    with open("nn_cp_log.txt", "w") as f:
        for k in data_CP.keys():
            l = str(k) + ";"
            for v in data_CP[k]:
                l += str(v) + ";"
            f.write(l[:-1] + "\n")
    '''

    g = None
    for ind in ["shc"]:
        with open("./pkl/" + str(ind) + ".pkl", "rb") as f:
            g = pickle.load(f)
        print(type(g))

        os.makedirs("./ndr_results", exist_ok=True)
        fold = "./ndr_results"
        args = {"episodes": 20, "upds": 20, "step": 20, "render": False, "task": "CartPole-v1",
                "trl": True, "old": None, "pr": 60, "hn": 5, "keys":"20_60_5"}
        #a, agent = evalCP(g,args)
        #agent.prune_weights()
        #args["old"] = agent
        #args["episodes"] = 80
        #args["upds"] = -1

        res = parallel_val_CP([g], "20_60_5")
        print("training fitness: " + str(res))
        
        print(sum(res[20:]))
        print(np.mean(res))
        
        res, agent = eval(g, episodes=20)
        print(np.mean(res))
        pregraph = nx.from_numpy_matrix(agent.weights, create_using=nx.DiGraph)
        pos = agent.nx_pos_because_it_was_too_hard_to_add_a_multipartite_from_a_list_as_it_works_for_the_bipartite(
            pregraph)
        plt.clf()
        nx.draw(pregraph, pos=pos, with_labels=True, font_weight='bold')
        plt.savefig(fold + "/initial_"+str(ind)+".png")
        agent.prune_weights()
        pregraph = nx.from_numpy_matrix(agent.weights, create_using=nx.DiGraph)
        pos = agent.nx_pos_because_it_was_too_hard_to_add_a_multipartite_from_a_list_as_it_works_for_the_bipartite(
            pregraph)
        plt.clf()
        nx.draw(pregraph, pos=pos, with_labels=True, font_weight='bold')
        plt.savefig(fold + "/pruned_"+str(ind)+".png")
        dag, hc = agent.dag("./ndr_results")
        top_sort = nx.topological_sort(dag)
        agent.top_sort = list(top_sort)
        agent.cycle_history = hc
        print(agent.cycle_history)
        print(agent.top_sort)
        print(nx.to_dict_of_lists(dag))
        plt.clf()
        pos = agent.nxpbiwthtaamfalaiwftb(dag)
        nx.draw(dag, pos=pos, with_labels=True, font_weight='bold')
        plt.savefig(fold + "/final_dag_"+str(ind)+".png")
        with open(fold + "/hc_"+str(ind)+".txt", "wb") as f:
            pickle.dump(hc, f)
        '''

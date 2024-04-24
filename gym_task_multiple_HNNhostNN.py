from cma import CMAEvolutionStrategy as cmaes
import gymnasium as gym
import numpy as np
import functools
from random import Random
from multiprocessing import Pool
import pickle
import sys
import os
import math
from plot_utils import plot_generations, plot_weights, plot_weights_gif
from HNNhostNN import HNNHostMultipleNN

# training an NN, called HostMultipleNN, that works with other NNs, each deciding a weight the host

# executed one time for each candidate
def eval(ds, render=False):

    x = ds[0]
    hnodes = ds[1]
    pruning_rate = ds[2]
    hostNodes = ds[3]
    guestNodes = ds[4]
    inputType = ds[5]

    cumulative_rewards = []
    weights = []

    task = gym.make("CartPole-v1")
    agent = HNNHostMultipleNN(hostNodes, guestNodes, inputType)

    agent.set_guest_weights(x)

    # each candidate tries the task 100 times, then the resulting reward is the average
    for i in range(100):
        cumulative_rewards.append(0)
        done = False
        obs, info = task.reset(seed=i, options={})
        counter = 0

        agent.compute_h_rules()

        # add weights to dict for plotting
        weights.append(agent.get_list_weights())

        while not done:
            output = agent.activate(obs)

            # arr[output]+=1
            if render:
                task.render()
            obs, rew, terminated, truncated, info = task.step(np.argmax(output))
            cumulative_rewards[-1] += rew

            agent.update_weights()

            done = terminated or truncated
        counter += 1

    return -sum(cumulative_rewards) / 100, weights


def generator(random, args):
    return np.asarray([random.uniform(args["pop_init_range"][0],
                                      args["pop_init_range"][1])
                       for _ in range(args["num_vars"])])


def generator_wrapper(func):
    @functools.wraps(func)
    def _generator(random, args):
        return np.asarray(func(random, args))

    return _generator

# multiprocessing
def parallel_val(candidates, hnodes, pr_ratio, hostNodes, guestNodes, inputType):
    with Pool(20) as p:
        # executes one eval foreach candidate
        x = p.map(eval, [[c, hnodes, pr_ratio, hostNodes, guestNodes, inputType] for c in candidates])
    #split the fitness values and the weights of the host
    return list(zip(*x))[0], list(zip(*x))[1]


def experiment_launcher(config):
    seed = config["seed"]
    hnodes = config["hnodes"]
    pr_ratio = config["pr_ratio"]

    # get the characteristics of the NN
    hostNodes = config["hostNodes"]
    guestNodes = config["guestNodes"]
    inputType = config["inputType"]

    # archives
    means = []
    bests = []
    final_weights = dict()

    args = {}
    fka = HNNHostMultipleNN(hostNodes, guestNodes, inputType)

    args["num_vars"] = fka.guestnWeights * 4 # Number of dimensions of the search space
    args["max_generations"] = config["max_generations"]
    args["sigma"] = 1.0  # default standard deviation
    args["num_offspring"] = 50 #4 + int(math.floor(3 * math.log(fka.nweights)))  # lambda
    args["pop_size"] = 10 #int(math.floor(args["num_offspring"] / 2))  # mu
    args["pop_init_range"] = [-1, 1]  # Range for the initial population
    args["hnodes"] = hnodes
    args["seed"] = seed


    random = Random()
    es = cmaes(generator(random, args),
               args["sigma"],
               {'popsize': args["num_offspring"],
                'seed': seed,
                'CMA_mu': args["pop_size"]})
    
    best = [0,0,0]
    worst = [0,0,-500]

    gen = 0
    logs = []
    while gen <= args["max_generations"]:
        candidates = es.ask()  # get list of new solutions
        fitnesses, all_weights = parallel_val(candidates, hnodes, pr_ratio, hostNodes, guestNodes, inputType)
        final_weights[gen] = all_weights
        print("generation "+str(gen)+"  "+str(fitnesses)+"  "+str(np.mean(fitnesses)))

        # to print weights
        if best[2] >= min(fitnesses):
          best[0] = gen
          best[1] = fitnesses.index(min(fitnesses))
          best[2] = min(fitnesses)

        if worst[2] <= max(fitnesses):
          worst[0] = gen
          worst[1] = fitnesses.index(max(fitnesses))
          worst[2] = max(fitnesses)

        means.append(np.mean(fitnesses))
        bests.append(min(fitnesses))

        es.tell(candidates, fitnesses)
        gen += 1
    #final_pop = np.asarray(es.ask())
    parallel_res, all_weights = parallel_val(candidates, hnodes, pr_ratio, hostNodes, guestNodes, inputType)
    final_weights[gen] = all_weights
    final_pop_fitnesses = np.asarray(parallel_res)

    best_guy = es.best.x
    best_fitness = es.best.f

    pickle_data = {"type": "HNNmultiple", "hostNodes": hostNodes, "guestNodes": guestNodes, "inputType": inputType, "candidate": best_guy}

    with open("pkl/host_single_HNNhostNN_"+str(best_fitness)+".pkl", "wb") as f:
        pickle.dump(pickle_data, f)

    return means, bests, final_weights, best, worst


if __name__ == "__main__":
    
    HostNodes = [4, 4, 2]
    GuestNodes =  [2, 3, 4]
    inputType = "ID" # @param ["IDA", "ID", "A", "IDAW", "IDAL"]

    seed = 0
    means, bests, all_weights, best, worst = experiment_launcher({"seed": seed, "hnodes": 5, "pr_ratio": 30,
                         "hostNodes": HostNodes, "guestNodes": GuestNodes, "inputType": inputType,
                         "max_generations": 2})


    plot_generations('Cartpole - Single NN compute W', means, bests)
    # plot worst
    plot_weights(all_weights, HostNodes, worst[0], worst[1], "Worst")
    plot_weights(all_weights, HostNodes, best[0], best[1], "Best")
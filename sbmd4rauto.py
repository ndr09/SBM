import argparse
import math
import os
import sys
from cma import CMAEvolutionStrategy as cmaes
import gym
from network5 import SBMD4Rauto
import numpy as np
import functools
from random import Random
from multiprocessing import Pool
import pickle
import matplotlib.pyplot as plt
import json

def eval(data, render=False):
    x = data[0]
    args = data[1]
    cumulative_rewards = []
    task = gym.make("LunarLander-v2")
    agent = SBMD4Rauto([8, args["hnodes"], 4], 0, 0.005, args["seed"], False,args["ahl"], args["rst"])
    agent.set_hrules(x)
    for i in range(100):
        cumulative_rewards.append(0)
        done = False
        task.seed(i)
        obs = task.reset()
        # counter = 0
        while not done:
            output = agent.distance_activation(obs)
            agent.store_activation()
            # print((np.argmax(output), output, agent.stable_nodes))
            if render:
                task.render()
            obs, rew, done, _ = task.step(np.argmax(output))
            cumulative_rewards[-1] += rew
    return -np.mean(cumulative_rewards)


def generator(random, args):
    return np.asarray([random.uniform(args["pop_init_range"][0],
                                      args["pop_init_range"][1])
                       for _ in range(args["num_vars"])])


def generator_wrapper(func):
    @functools.wraps(func)
    def _generator(random, args):
        return np.asarray(func(random, args))

    return _generator


def parallel_val(candidates, args):
    # with parallel_backend('multiprocessing'):
    with Pool(20) as p:
        return p.map(eval, [[c, json.loads(json.dumps(args))] for c in candidates])


def experiment_launcher(config):
    seed = config["seed"]
    hnodes = config["hnodes"]
    rst = config["rst"]
    ahl = config["ahl"]
    print(config)
    os.makedirs("./results_SBMD4rauto_ll/", exist_ok=True)
    os.makedirs("./results_SBMD4rauto_ll/" + str(ahl), exist_ok=True)
    os.makedirs("./results_SBMD4rauto_ll/" + str(ahl) + "/" + str(rst), exist_ok=True)
    os.makedirs("./results_SBMD4rauto_ll/" + str(ahl) + "/" + str(rst) + "/" + str(hnodes), exist_ok=True)
    os.makedirs("./results_SBMD4rauto_ll/" + str(ahl) + "/" + str(rst) + "/" + str(hnodes) + "/" + str(seed),
                exist_ok=True)

    fka = SBMD4Rauto([4, hnodes, 2], rst, 0.01, seed, True)
    rng = np.random.default_rng()
    # fka.set_hrules(rng.random(fka.tns * 4))
    args = {}
    args["num_vars"] = fka.tns * 4 - fka.nodes[0] - fka.nodes[-1]  # Number of dimensions of the search space
    args["sigma"] = 1.0  # default standard deviation

    args["num_offspring"] = 20  # 4 + int(math.floor(3 * math.log(fka.nweights * 4)))  # lambda
    args["pop_size"] = int(math.floor(args["num_offspring"] / 2))  # mu
    args["max_generations"] = (1000 - args["pop_size"]) // args["num_offspring"] + 1
    args["pop_init_range"] = [-1, 1]  # Range for the initial population
    args["ahl"] = ahl
    args["rst"] = rst
    args["hnodes"] = hnodes
    args["seed"] = seed
    random = Random(seed)
    es = cmaes(generator(random, args),
               args["sigma"],
               {'popsize': args["num_offspring"],
                'seed': seed,
                'CMA_mu': args["pop_size"]})
    gen = 0
    logs = []
    while gen <= args["max_generations"]:
        candidates = es.ask()  # get list of new solutions
        fitnesses = parallel_val(candidates, args)
        log = "generation " + str(gen) + "  " + str(min(fitnesses)) + "  " + str(np.mean(fitnesses))
        with open("./results_SBMD4rauto_ll/" + str(ahl) + "/" + str(rst) + "/" + str(hnodes) + "/" + str(
                seed) + "/best_" + str(
            gen) + ".pkl", "wb") as f:
            pickle.dump(candidates[np.argmin(fitnesses)], f)
        logs.append(log)

        print(log)
        es.tell(candidates, fitnesses)
        gen += 1
    final_pop = np.asarray(es.ask())
    final_pop_fitnesses = np.asarray(parallel_val(final_pop, args))

    best_guy = es.best.x
    best_fitness = es.best.f

    with open("./results_SBMD4rauto_ll/" + str(ahl) + "/" + str(rst) + "/" + str(hnodes) + "/" + str(seed) + "/" + str(
            best_fitness) + ".pkl", "wb") as f:
        pickle.dump(best_guy, f)
    with open("./results_SBMD4rauto_ll/" + str(ahl) + "/" + str(rst) + "/" + str(hnodes) + "/" + str(seed) + "/log.txt",
              "w") as f:
        for l in logs:
            f.write(l + "\n")


def chs(dir):
    return os.path.exists(dir + "/log.txt")


if __name__ == "__main__":
    c = 0
    if c ==0:
        seed = 0
        for ahl in [20,30]:
            for rst in [10]:
                for hnodes in [9]:
                    dir = "./results_SBMD4rauto_ll/" + str(ahl) + "/" + str(rst) + "/" + str(hnodes) + "/" + str(seed) + "/"
                    if not chs(dir):
                        experiment_launcher({"seed": seed, "rst": rst, "ahl": ahl, "hnodes": hnodes})
                        print("ended experiment " + str({"seed": seed, "rst": rst, "ahl": ahl, "hnodes": hnodes}))
    else:
        d = {"seed": 0, "rst": 0, "ahl": [101], "hnodes": 5}
        a = SBMD4Rauto([4, 5, 2], 0, 0.01, 0, False)
        x = [np.random.random() for i in range(a.nweights * 4)]
        a.set_hrules(x)
    # a.distance_activation([1,1,1,1])
    # a.update_weights()
    # a.distance_activation([1,1,1,1])

        eval([x, d], False)


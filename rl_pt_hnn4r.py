import argparse
import math
import os
import sys
from cma import CMAEvolutionStrategy as cmaes
import gym
from network_pt import NHNN
import numpy as np
import functools
from random import Random
from multiprocessing import Pool
import pickle
import matplotlib.pyplot as plt
from optimizer import LMMAES
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch
from vision_task import eval_minst
from network_pt import NHNN
import json


def eval(data, render=False):
    x = data[0]
    # print(x.tolist())
    args = data[1]
    cumulative_rewards = []
    task = gym.make("LunarLander-v2")
    agent = NHNN([8, args["hnodes"], 4], 0.001, device="cpu")
    agent.set_hrules(x)
    for i in range(100):
        cumulative_rewards.append(0)
        done = False
        task.seed(i)
        obs = task.reset()
        # counter = 0
        while not done:
            output = agent.forward(torch.tensor(obs, dtype=torch.float))

            if render:
                task.render()
            obs, rew, done, _ = task.step(np.argmax(output).item())
            cumulative_rewards[-1] += rew
            agent.update_weights()

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
    with Pool(20) as p:
        return p.map(eval, [[c, json.loads(json.dumps(args))] for c in candidates])
    # res = []
    # # print(args)
    #
    # for c in candidates:
    #     res.append(eval((c, json.loads(json.dumps(args)))))
    #     print("ended ", len(res))
    # return res


def experiment_launcher(config):
    seed = config["seed"]
    hnodes = config["hnodes"]
    print(config)

    fka = NHNN([8, hnodes, 4], 0.001)
    rng = np.random.default_rng()
    # fka.set_hrules(rng.random(fka.tns * 4))
    args = {}
    args["num_vars"] = fka.nparams.item()  # Number of dimensions of the search space
    args["sigma"] = 1.0  # default standard deviation
    args["num_offspring"] = 20  # 4 + int(math.floor(3 * math.log(fka.nweights * 4)))  # lambda
    args["pop_size"] = int(math.floor(args["num_offspring"] / 2))  # mu
    args["max_generations"] = (20000 - args["pop_size"]) // args["num_offspring"] + 1
    args["pop_init_range"] = [-1, 1]  # Range for the initial population
    args["hnodes"] = hnodes
    args["seed"] = seed
    args["dir"] = config["dir"]
    random = Random(seed)
    es = LMMAES(args["num_vars"], lambda_=20, mu=10, sigma=1)

    # es = cmaes(generator(random, args),
    #            args["sigma"],
    #            {'popsize': args["num_offspring"],
    #             'seed': seed,
    #             'CMA_mu': args["pop_size"]})
    gen = 0
    logs = []
    while gen <= args["max_generations"]:
        candidates = es.ask()  # get list of new solutions

        fitnesses = parallel_val(candidates, args)
        log = "generation " + str(gen) + "  " + str(min(fitnesses)) + "  " + str(np.mean(fitnesses))
        with open(args["dir"] + "/best_" + str(gen) + ".pkl", "wb") as f:
            pickle.dump(candidates[np.argmin(fitnesses)], f)

        with open(args["dir"] + "/tlog.txt", "a") as f:
            f.write(log + "\n")

        logs.append(log)

        es.tell(fitnesses)
        gen += 1

    best_guy = es.best.x
    best_fitness = es.best.f

    with open(args["dir"] + "/" + str(best_fitness) + ".pkl", "wb") as f:
        pickle.dump(best_guy, f)
    with open(args["dir"] + "/log.txt", "w") as f:
        for l in logs:
            f.write(l + "\n")


def chs(dir):
    return os.path.exists(dir + "/log.txt")


if __name__ == "__main__":
    seed = int(sys.argv[1])
    for ps in [[1], [10], [20]]:
        for prate in [0, 80, 90]:
            for hnodes in [5, 20, 90]:

                os.makedirs("./results_pt_NHNN/", exist_ok=True)
                os.makedirs("./results_pt_NHNN/" + str(ps[0]), exist_ok=True)
                os.makedirs("./results_pt_NHNN/" + str(ps[0]) + "/" + str(prate), exist_ok=True)
                os.makedirs("./results_pt_NHNN/" + str(ps[0]) + "/" + str(prate) + "/" + str(hnodes), exist_ok=True)
                os.makedirs("./results_pt_NHNN/" + str(ps[0]) + "/" + str(prate) + "/" + str(hnodes) + "/" + str(seed),
                            exist_ok=True)
                dir = "./results_pt_NHNN/" + str(ps[0]) + "/" + str(prate) + "/" + str(hnodes) + "/" + str(seed) + "/"
                if not chs(dir):
                    experiment_launcher({"seed": seed, "prate": prate, "ps": ps, "hnodes": hnodes, "dir": dir})
                    print("ended experiment " + str({"seed": seed, "prate": prate, "ps": ps, "hnodes": hnodes}))

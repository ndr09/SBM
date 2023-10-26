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
    # with Pool(20) as p:
    #     return p.map(eval, [[c, args] for c in candidates])
    res = []
    for c in candidates:
        res.append(eval_minst((c, args)))
    return res


def experiment_launcher(config):
    seed = config["seed"]
    hnodes = config["hnodes"]
    rst = config["rst"]
    ahl = config["ahl"]
    print(config)
    os.makedirs("./results_vis_HNN4R/", exist_ok=True)
    os.makedirs("./results_vis_HNN4R/" + "/" + str(hnodes), exist_ok=True)
    os.makedirs("./results_vis_HNN4R/" + "/" + str(hnodes) + "/" + str(seed), exist_ok=True)

    fka = NHNN([28*28, 100, 100, 100, 10], 0.01)
    rng = np.random.default_rng()
    # fka.set_hrules(rng.random(fka.tns * 4))
    args = {}
    args["num_vars"] = fka.nparams #3 * (28*28) + 4 * 300 + 3 * 10  # Number of dimensions of the search space
    args["sigma"] = 1.0  # default standard deviation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)
        )
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    train_kwargs = {'batch_size': 128}
    args["tl"] = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    args["num_offspring"] = 20  # 4 + int(math.floor(3 * math.log(fka.nweights * 4)))  # lambda
    args["pop_size"] = int(math.floor(args["num_offspring"] / 2))  # mu
    args["max_generations"] = (10000 - args["pop_size"]) // args["num_offspring"] + 1
    args["pop_init_range"] = [-1, 1]  # Range for the initial population
    args["ahl"] = ahl
    args["rst"] = rst
    args["hnodes"] = hnodes
    args["seed"] = seed
    random = Random(seed)
    es = LMMAES(args["num_vars"], lambda_=10, mu=5, sigma=1)

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
        with open("./results_vis_HNN4R/" + "/" + str(hnodes) + "/" + str(seed) + "/best_" + str(
                gen) + ".pkl", "wb") as f:
            pickle.dump(candidates[np.argmin(fitnesses)], f)
        logs.append(log)

        print(log)
        es.tell(fitnesses)
        gen += 1
    final_pop = np.asarray(es.ask())
    final_pop_fitnesses = np.asarray(parallel_val(final_pop, args))

    best_guy = es.best.x
    best_fitness = es.best.f

    with open("./results_vis_HNN4R/" + "/" + str(hnodes) + "/" + str(seed) + "/" + str(
            best_fitness) + ".pkl", "wb") as f:
        pickle.dump(best_guy, f)
    with open("./results_vis_HNN4R/" + "/" + str(hnodes) + "/" + str(seed) + "/log.txt",
              "w") as f:
        for l in logs:
            f.write(l + "\n")


def chs(dir):
    return os.path.exists(dir + "/log.txt")


if __name__ == "__main__":
    c = 0
    seed = 0

    dir = "./results_vis_HNN4R/" + "/" + str(300) + "/" + str(seed) + "/"
    if not chs(dir):
        experiment_launcher({"seed": seed, "rst": 0, "ahl": 0, "hnodes": 300})
        print("ended experiment " + str({"seed": seed, "rst": 0, "ahl": 0, "hnodes": 300}))

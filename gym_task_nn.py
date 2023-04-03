from cma import CMAEvolutionStrategy as cmaes
import gym
from network import RNN, NN, HNN
import numpy as np
import functools
from random import Random
from multiprocessing import Pool
import pickle
import sys
import os
import math


def eval(ds, render=False):
    x = ds[0]
    hnodes = ds[1]
    cumulative_rewards = []
    task = gym.make("LunarLander-v2")
    agent = NN([8, hnodes, 4])
    agent.set_weights(x)
    agent.nn_prune_weights(ds[2])
    for i in range(100):
        cumulative_rewards.append(0)
        done = False
        task.seed(i)
        obs = task.reset()
        counter = 0
        while not done:
            output = agent.activate(obs)

            # arr[output]+=1
            if render:
                task.render()
            obs, rew, done, _ = task.step(np.argmax(output))
            cumulative_rewards[-1] += rew
        counter += 1

    return -sum(cumulative_rewards) / 100


def generator(random, args):
    return np.asarray([random.uniform(args["pop_init_range"][0],
                                      args["pop_init_range"][1])
                       for _ in range(args["num_vars"])])


def generator_wrapper(func):
    @functools.wraps(func)
    def _generator(random, args):
        return np.asarray(func(random, args))

    return _generator


def parallel_val(candidates, hnodes, pr_ratio):
    with Pool(20) as p:
        return p.map(eval, [[c, hnodes, pr_ratio] for c in candidates])


def experiment_launcher(config):
    seed = config["seed"]
    hnodes = config["hnodes"]
    pr_ratio = config["pr_ratio"]
    os.makedirs("./results_NN/", exist_ok=True)
    os.makedirs("./results_NN/" + str(hnodes), exist_ok=True)
    os.makedirs("./results_NN/" + str(hnodes) + "/" + str(seed), exist_ok=True)
    os.makedirs("./results_NN/" + str(hnodes) + "/" + str(pr_ratio) + "/" + str(seed), exist_ok=True)
    fka = NN([8, hnodes, 4])
    args = {}
    args["num_vars"] = fka.nweights  # Number of dimensions of the search space
    args["max_generations"] = (2000-args["pop_size"]) // args["num_offspring"] + 1
    args["sigma"] = 1.0  # default standard deviation

    args["num_offspring"] = 4 + int(math.floor(3 * math.log(fka.nweights)))  # lambda
    args["pop_size"] = int(math.floor(args["num_offspring"] / 2))  # mu
    args["pop_init_range"] = [-1, 1]  # Range for the initial population
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
        fitnesses = parallel_val(candidates, hnodes, pr_ratio)
        log = "generation " + str(gen) + "  " + str(min(fitnesses)) + "  " + str(np.mean(fitnesses))
        with open("./results_NN/" + str(hnodes) + "/" + str(pr_ratio) + "/" + str(seed) + "/best_" + str(
                gen) + ".pkl", "wb") as f:
            pickle.dump(candidates[np.argmin(fitnesses)], f)
        logs.append(log)
        print(log)
        es.tell(candidates, fitnesses)
        gen += 1
    final_pop = np.asarray(es.ask())
    final_pop_fitnesses = np.asarray(parallel_val(final_pop, hnodes, pr_ratio))

    best_guy = es.best.x
    best_fitness = es.best.f

    with open("./results_NN/" + str(hnodes) + "/" + str(pr_ratio) + "/" + str(seed) + "/" + str(
            best_fitness) + ".pkl", "wb") as f:
        pickle.dump(best_guy, f)
    with open("./results_NN/" + str(hnodes) + "/" + str(pr_ratio) + "/" + str(seed) + "/log.txt", "w") as f:
        for l in logs:
            f.write(l + "\n")


if __name__ == "__main__":
    args = {}
    seed = int(sys.argv[1])
    for hn in range(5, 10):
        for pr_ratio in range(20, 100, 20):
            experiment_launcher({"seed": seed, "hnodes": hn, "pr_ratio": pr_ratio})

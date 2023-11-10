import os

from cma import CMAEvolutionStrategy as cmaes
import gym
from network5 import NNN
import numpy as np
import functools
from random import Random
from multiprocessing import Pool
import pickle

def eval(x, render=False):
    cumulative_rewards = []
    task = gym.make("CartPole-v1")
    agent = NNN([4, 5, 2])
    agent.set_weights(x)
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

def parallel_val(candidates):
    with Pool(20) as p:
        return p.map(eval, candidates)

if __name__ == "__main__":
    args = {}
    fka = NNN([4, 5, 2])
    rng = np.random.default_rng()
    #eval(rng.random(fka.nweights*4), render=True)
    os.makedirs("results_nnn", exist_ok=True)

    args["num_vars"] = fka.nweights  # Number of dimensions of the search space
    args["max_generations"] = 50
    args["sigma"] = 1.0  # default standard deviation
    args["pop_size"] = 4  # mu
    args["num_offspring"] = 20  # lambda
    args["pop_init_range"] = [-1, 1]  # Range for the initial population

    random = Random(0)
    es = cmaes(generator(random, args),
               args["sigma"],
               {'popsize': args["num_offspring"],
                'seed': 0,
                'CMA_mu': args["pop_size"]})
    gen = 0
    while gen <= args["max_generations"]:
        candidates = es.ask()  # get list of new solutions
        fitnesses = parallel_val(candidates)
        print("generation "+str(gen)+"  "+str(min(fitnesses))+"  "+str(np.mean(fitnesses)))
        es.tell(candidates, fitnesses)
        gen += 1
    final_pop = np.asarray(es.ask())
    final_pop_fitnesses = np.asarray(parallel_val(final_pop))

    best_guy = es.best.x
    best_fitness = es.best.f
    with open("./results_nnn/nnn_test_"+str(best_fitness)+".pkl", "wb") as f:
        pickle.dump(best_guy, f)

from cma import CMAEvolutionStrategy as cmaes
import gym
from network import RNN
import numpy as np
import functools
from random import Random
from multiprocessing import Pool
import pickle
import math


def normLL(obs):
    low = np.array(
        [
            # these are bounds for position
            # realistically the environment should have ended
            # long before we reach more than 50% outside
            -1.,
            -1.,
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
            1.,
            1.,
            # velocity bounds is 5x rated speed
            5.0,
            5.0,
            3.14,
            5.0,
            1.0,
            1.0,
        ]
    ).astype(np.float32)
    a = np.array(obs)
    a[2] = a[2] / 5.
    a[3] = a[3] / 5.
    a[4] = a[4] / 3.14
    a[5] = a[5] / 5.
    a[-2] = obs[-2]
    a[-1] = obs[-1]
    return a

def eval(x, render=False):
    cumulative_rewards = []
    task = gym.make("LunarLander-v2")
    agent = RNN([8, 10, 4], 20, 0.01, 0, False)
    agent.set_hrules(x)
    for i in range(120):
        cumulative_rewards.append(0)
        done = False
        task.seed(i)
        obs = task.reset() #normLL(task.reset())
        counter = 0
        while not done:
            output = agent.activate(obs)

            # arr[output]+=1
            if render:
                task.render()
            obs, rew, done, _ = task.step(np.argmax(output))
            #obs = normLL(obs)
            cumulative_rewards[-1] += rew
            if counter<20:
                agent.update_weights()
        counter += 1
        if counter in range(9,20,1):
            agent.prune_weights()
    return -sum(cumulative_rewards[20:]) / 100


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
    fka = RNN([8, 10, 4], 20, 0.01, 0, False)
    rng = np.random.default_rng(0)
    fka.set_hrules(rng.random(fka.nweights*4))
    #eval(rng.random(fka.nweights*4), render=True)


    args["num_vars"] = fka.nweights * 4  # Number of dimensions of the search space
    args["max_generations"] = 70
    args["sigma"] = 0.5  # default standard deviation
    args["num_offspring"] = 4 + int(math.floor(3*math.log(fka.nweights * 4)))  # lambda
    args["pop_size"] = int(math.floor( args["num_offspring"] / 2))  # mu
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
        with open("./pkl/n_nomr_best_"+str(gen)+".pkl", "wb") as f:
            pickle.dump(candidates[np.argmin(fitnesses)], f)
        es.tell(candidates, fitnesses)
        gen += 1
    final_pop = np.asarray(es.ask())
    final_pop_fitnesses = np.asarray(parallel_val(final_pop))

    best_guy = es.best.x
    best_fitness = es.best.f
    with open("./pkl/n_norm_"+str(best_fitness)+"_+.pkl", "wb") as f:
        pickle.dump(best_guy, f)

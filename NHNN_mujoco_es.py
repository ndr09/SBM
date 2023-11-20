
import os
from optimizer import *
import  gym
import numpy as np
import functools
from random import Random
from multiprocessing import Pool
import pickle
import torch
from network_pt import NHNN
import json

gym.logger.set_level(40)
import contextlib
from PIL import Image
with contextlib.redirect_stdout(None):
    import pybullet_envs

def eval(data, render=False):
    import ctypes
    import sys


    x = data[0]
    # print(x.tolist())
    args = data[1]
    cumulative_rewards = []
    task = None

    task = gym.make("AntBulletEnv-v0")
    agent = NHNN([28, 128, 64, 8], 0.01, init="rand")
    decay = - 0.01 * np.mean(agent.get_weights().numpy() ** 2)
    agent.set_hrules(x)
    obs = task.reset()

    action = np.ones(8)
    #
    # for i in range(40):
    #     _,_,done,_ = task.step(action)
    #     a = task.render(mode="rgb_array")
    #     print(i, done)
    #     im = Image.fromarray(a)
    #     im.save("img/"+str(i)+".jpeg")
    cumulative_rewards.append(0)
    done = False
    # exit(1)

    neg_count = 0
    rew_ep = 0
    t = 0
    # counter = 0
    while not done:

        output = agent.forward(torch.tensor(obs, dtype=torch.float))

        if render:
            task.render(mode="human")

        obs, _, done, info = task.step(output.numpy())

        rew = task.unwrapped.rewards[1]
        rew_ep += rew
        agent.update_weights()

        if t > 200:
            neg_count = neg_count + 1 if rew < 0.0 else 0
            if (neg_count > 30):
                done = True
        t+=1
    # print("=====", t)

    return rew_ep+decay


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
    # res = [eval([c, json.loads(json.dumps(args))]) for c in candidates]
    # return res

def experiment_launcher(config):
    seed = config["seed"]
    hnodes = config["hnodes"]
    print(config)

    fka = NHNN([28, 128, 64, 8], 0.001)
    rng = np.random.default_rng()
    args = {}
    args["num_vars"] = fka.nparams.item()  # Number of dimensions of the search space
    print("this problem has "+str(args["num_vars"] )+" parameters")
    args["sigma"] = 1.0  # default standard deviation
    args["num_offspring"] = 20  # 4 + int(math.floor(3 * math.log(fka.nweights * 4)))  # lambda
    args["pop_size"] = int(math.floor(args["num_offspring"] / 2))  # mu
    args["max_generations"] = 500#(20000 - args["pop_size"]) // args["num_offspring"] + 1
    args["pop_init_range"] = [-1, 1]  # Range for the initial population
    args["hnodes"] = hnodes
    args["seed"] = seed
    args["dir"] = config["dir"]
    random = Random(seed)
    # es = cmaes(generator(random, args),
    #            args["sigma"],
    #            {'popsize': args["num_offspring"],
    #             'seed': seed,
    #             'CMA_mu': args["pop_size"]})
    es = EvolutionStrategy(seed,args["num_vars"],population_size=500,learning_rate=0.2,sigma=0.1,decay=0.995 )



    #LMMAES(args["num_vars"], lambda_=20, mu=10, sigma=1)

    gen = 0
    logs = []
    while gen <= args["max_generations"]:
        candidates = es.ask()  # get list of new solutions
        print("ask")
        fitnesses = parallel_val(candidates, args)
        log = "generation " + str(gen) + "  " + str(max(fitnesses)) + "  " + str(np.mean(fitnesses))
        print(log)
        with open(args["dir"] + "/best_" + str(gen) + ".pkl", "wb") as f:
            pickle.dump(candidates[np.argmin(fitnesses)], f)

        with open(args["dir"] + "/tlog.txt", "a") as f:
            f.write(log + "\n")

        logs.append(log)

        es.tell(fitnesses)
        print("tell")
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
    debug = True
    for ps in [[101]]:
        for prate in [0]:
            for hnodes in ["ml"]:
                bd = "./results_pbant_NHNN/"
                os.makedirs(bd, exist_ok=True)
                os.makedirs(bd + str(ps[0]), exist_ok=True)
                os.makedirs(bd + str(ps[0]) + "/" + str(prate), exist_ok=True)
                os.makedirs(bd + str(ps[0]) + "/" + str(prate) + "/" + str(hnodes), exist_ok=True)
                os.makedirs(bd + str(ps[0]) + "/" + str(prate) + "/" + str(hnodes) + "/" + str(seed),
                            exist_ok=True)
                dir = bd + str(ps[0]) + "/" + str(prate) + "/" + str(hnodes) + "/" + str(seed) + "/"
                if not chs(dir) or debug:
                    experiment_launcher({"seed": seed, "prate": prate, "ps": ps, "hnodes": hnodes, "dir": dir})
                    print("ended experiment " + str({"seed": seed, "prate": prate, "ps": ps, "hnodes": hnodes}))

import sys
import os

import gymnasium as gym
import numpy as np
import functools
import pickle
from multiprocessing import Pool
from SBM.common.network5 import ND_SBM
import json


def eval(data):
    x = data[0]
    args = data[1]

    task = gym.make(args["task"] + "-v4")

    agent = ND_SBM(args["nodes"], prune_ratio=args["pr"], seed=args['seed'])
    agent.set_hrules(x)
    obs, info = task.reset()
    done = False
    truncated = False
    rew_ep = 0
    t = 0
    uflag = True
    while not (done or truncated):
        output = agent.activate(obs, uflag)
        obs, rew, done, truncated, info = task.step(output)
        if t == args['ps']:
            agent.prune_weights()
            uflag = False
        rew_ep += rew
        t += 1
    task.close()

    return rew_ep


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
    with Pool(-1) as p:
        return p.map(eval, [[c, json.loads(json.dumps(args))] for c in candidates])
    # res = [eval([c, json.loads(json.dumps(args))]) for c in candidates]
    # return res


def experiment_launcher(config):
    seed = config["seed"]

    print(config)
    fka = ND_SBM(config["nodes"], 0, 0)
    args = config
    args["generations"] = 1000
    args["num_vars"] = fka.nparams  # Number of dimensions of the search space
    print("this problem has " + str(args["num_vars"]) + " parameters")
    args["seed"] = seed

    es = ESE(seed, args["num_vars"], 100, 0.35)

    gen = 0
    logs = []
    while gen <= args["generations"]:
        candidates = es.ask()  # get list of new solutions
        fitnesses = parallel_val(candidates, args)
        log = "generation " + str(gen) + "  " + str(max(fitnesses)) + "  " + str(np.mean(fitnesses))
        print(log)
        with open(os.path.join(args["dir"], "best_" + str(gen) + ".pkl"), "wb") as f:
            pickle.dump(candidates[np.argmax(fitnesses)], f)

        with open(os.path.join(args["dir"], "tlog.txt"), "a") as f:
            f.write(log + "\n")

        logs.append(log)

        es.tell(candidates, fitnesses)
        gen += 1

    best_guy = es.elite[1]
    best_fitness = es.elite[0]

    with open(os.path.join(args["dir"], str(best_fitness) + ".pkl"), "wb") as f:
        pickle.dump(best_guy, f)

    with open(os.path.join(args["dir"], "log.txt"), "w") as f:
        for l in logs:
            f.write(l + "\n")


def chs(dir):
    return os.path.exists(os.path.join(dir, "log.txt"))


if __name__ == "__main__":
    seed = int(sys.argv[1])
    task = sys.argv[2]

    args = {"seed": seed,
            "task": task}
    for hnodes in [100, 200, 300]:
        for pr in [0, 40, 60, 80, 90, 99]:
            for ps in [2, 200, 400, 600, 800]:
                if not chs(os.path.join("RQ2", "DSBM", args["task"], str(hnodes), str(pr), str(ps), str(seed))):
                    args["dir"] = os.path.join("RQ2","DSBM", args["task"], str(hnodes), str(pr), str(ps), str(seed))

                    os.makedirs(os.path.join("RQ2", "DSBM"), exist_ok=True)
                    os.makedirs(os.path.join("RQ2", "DSBM", task), exist_ok=True)
                    os.makedirs(os.path.join("RQ2", "DSBM", task, str(hnodes)), exist_ok=True)
                    os.makedirs(os.path.join("RQ2", "DSBM", task, str(hnodes), str(pr)), exist_ok=True)
                    os.makedirs(os.path.join("RQ2", "DSBM", task, str(hnodes), str(pr), str(ps)), exist_ok=True)
                    os.makedirs(os.path.join("RQ2", "DSBM", task, str(hnodes), str(pr), str(ps), str(seed)),
                                exist_ok=True)

                    taskinfo = {"Ant": [27, 8],
                                "HalfCheetah": [17, 6],
                                "Hopper": [11, 3],
                                "Pusher": [23, 7]
                                }
                    args["nodes"] = [taskinfo[task][0], hnodes, taskinfo[task][0]]

                    experiment_launcher(args)
                    print("ended experiment " + str(args))

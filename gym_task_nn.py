from cma import CMAEvolutionStrategy as cmaes
import gym
from network import RNN, NN, HNN
import numpy as np
import functools
from random import Random
from multiprocessing import Pool
import pickle
import matplotlib.pyplot as plt

def eval(x, render=False):
    cumulative_rewards = []
    task = gym.make("LunarLander-v2")
    agent = NN([8, 5, 4], 20)
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

def get_pruning_mask(nn, weights, prune_ratio):
    mask = np.ones_like(weights)
    c = 0
    nn.set_weights(weights)
    for i in range(1,len(nn.nodes)):
        for j in range(nn.nodes[i]):
            num_to_prune = round(prune_ratio/100 * nn.nodes[i-1])
            neuron_weights = nn.weights[i-1][j]
            sorted_weights = np.sort(np.abs(neuron_weights))
            threshold = sorted_weights[num_to_prune-1]
            neuron_mask = np.ones_like(neuron_weights)
            neuron_mask[np.abs(neuron_weights) <= threshold] = 0
            for l in range(nn.nodes[i-1]):
                    mask[c] = neuron_mask[l]
                    c +=1
    return mask


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

def plot_fitnesses(fitnesses):
    c = 0
    fig, axes = plt.subplots()
    for i, fitness in enumerate(fitnesses):
        x = list(range(1, len(fitness) + 1))
        if len(x) > len(fitness):
            x = x[:len(fitness)]
        label = f"Sparsity {round((0.8 ** i) * 100, 1)}%"
        axes.plot(x, fitness, label=label)
        c += 1
    axes.axhline(y=-200, linestyle='--', color='black', label='Successful score')
    axes.legend()
    axes.set_xlabel('Numero di generazioni')
    axes.set_ylabel('Punteggio medio')
    plt.savefig(f'./gen{c*10}.png')


if __name__ == "__main__":
    args = {}
    fka = NN([8, 5, 4], 20)
    rng = np.random.default_rng()

    args["num_vars"] = fka.nweights
    args["max_total_generations"] = 80
    args["max_partial_generations"] = 10
    args["sigma"] = 1.0
    args["pop_size"] = 4
    args["num_offspring"] = 20
    args["pop_init_range"] = [0, 1]

    random = Random(0)
    es = cmaes(generator(random, args),
               args["sigma"],
               {'popsize': args["num_offspring"],
                'seed': 0,
                'CMA_mu': args["pop_size"]})

    gen = 0
    n = 1
    pruning_mask = np.ones(args["num_vars"])
    initial_candidates = es.ask()
    candidates = initial_candidates
    fitnesses = np.zeros(args["num_offspring"])
    current_prune_ratio = 0

    fitnesses_history = []
    
    while gen <= args["max_total_generations"]:
        fitnesses_history.append([])
        for i in range(args["max_partial_generations"]*n):
            if i == 0:
                gen = 0
                candidates = es.ask()
                # apply the pruning mask to the whole offspring
                candidates = np.multiply(initial_candidates, np.tile(pruning_mask, (args["num_offspring"],1)))
            else:
                candidates = es.ask()
                # apply the pruning mask to the whole offspring
                candidates = np.multiply(candidates, np.tile(pruning_mask, (args["num_offspring"],1)))
            fitnesses = parallel_val(candidates)
            fitnesses_history[n-1].append(np.mean(fitnesses))
            print("Network density: "+str((0.8**(n-1))*100)+"%, Gen "+str(gen+1)+"  "+str(min(fitnesses))+"  "+str(np.mean(fitnesses)))
            es.tell(candidates, fitnesses)
            gen += 1
        print()
        plot_fitnesses(fitnesses_history)
        # calculate the pruning mask on the best guy in the current generation
        best_guy = es.best.x
        current_prune_ratio += (100-current_prune_ratio)*fka.prune_ratio/100
        pruning_mask = get_pruning_mask(fka, best_guy, current_prune_ratio)
        n += 1
    final_pop = np.asarray(es.ask())
    final_pop_fitnesses = np.asarray(parallel_val(final_pop))
    
    best_guy = es.best.x
    best_fitness = es.best.f

    with open("./nn_test_"+str(best_fitness)+".pkl", "wb") as f:
        pickle.dump(best_guy, f)
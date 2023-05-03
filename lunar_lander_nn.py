from cma import CMAEvolutionStrategy as cmaes
import gym
from network import RNN, NN, HNN
import numpy as np
import functools
from random import Random
from multiprocessing import Pool
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm


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


def plot_fitnesses(average_scores_history, best_scores_history):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(24,12))
    colors = cm.rainbow(np.linspace(0, 1, len(average_scores_history)))

    average_scores_history_concat = sum(average_scores_history, [])
    best_scores_history_concat = sum(best_scores_history, [])

    for i, fitness in enumerate(average_scores_history):
        x = list(range(sum(map(len, average_scores_history[:i])), sum(map(len, average_scores_history[:i+1]))))
        axes[0].plot(x, fitness, label=f'Sparsity {round((0.8 ** i) * 100, 1)}%', color=colors[i])
    axes[0].axhline(y=-200, linestyle='--', color='black', label='Successful score')
    axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    axes[0].set_xlabel('Numero di generazioni')
    axes[0].set_ylabel('Punteggio medio')
    axes[0].set_title('Average scores')
    axes[0].set_ylim(bottom=min(min(best_scores_history_concat), min(average_scores_history_concat), -200)-30, top=max(max(best_scores_history_concat), max(average_scores_history_concat), -200)+30)

    for i, fitness in enumerate(best_scores_history):
        x = list(range(sum(map(len, best_scores_history[:i])), sum(map(len, best_scores_history[:i+1]))))
        axes[1].plot(x, fitness, label=f'Sparsity {round((0.8 ** i) * 100, 1)}%', color=colors[i])
    axes[1].axhline(y=-200, linestyle='--', color='black', label='Successful score')
    axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    axes[1].set_xlabel('Numero di generazioni')
    axes[1].set_ylabel('Punteggio migliore')
    axes[1].set_title('Best guy scores')
    axes[1].set_ylim(bottom=min(min(best_scores_history_concat), min(average_scores_history_concat), -200)-30, top=max(max(best_scores_history_concat), max(average_scores_history_concat), -200)+30)

    plt.subplots_adjust(right=0.82)
    
    plt.savefig(f'./LunarLander_PruningIteration#{len(average_scores_history)-1}.png')


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

    average_scores_history = []
    best_scores_history = []
    
    while gen <= args["max_total_generations"]-args["max_partial_generations"]:
        average_scores_history.append([])
        best_scores_history.append([])
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
            average_scores_history[n-1].append(np.mean(fitnesses))
            best_scores_history[n-1].append(min(fitnesses))
            print("Network density: "+str((0.8**(n-1))*100)+"%, Gen "+str(gen+1)+"  "+str(min(fitnesses))+"  "+str(np.mean(fitnesses)))
            es.tell(candidates, fitnesses)
            gen += 1
        print()
        plot_fitnesses(average_scores_history, best_scores_history)
        # calculate the pruning mask on the best guy in the current generation
        best_guy = es.best.x
        current_prune_ratio += (100-current_prune_ratio)*fka.prune_ratio/100
        pruning_mask = get_pruning_mask(fka, best_guy, current_prune_ratio)
        n += 1
    final_pop = np.asarray(es.ask())
    final_pop_fitnesses = np.asarray(parallel_val(final_pop))
    
    best_guy = es.best.x
    best_fitness = es.best.f

    with open("./nn_lunarlander_"+str(best_fitness)+".pkl", "wb") as f:
        pickle.dump(best_guy, f)
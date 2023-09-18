from cma import CMAEvolutionStrategy as cmaes
import gym
from spiking_network_v5 import SNN
import numpy as np
import functools
from random import Random
from multiprocessing import Pool
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
from snntorch import spikegen
import torch
import math


def closest_binary(n, m):
    binary_number = []
    for i in range(n):
        if m >= 2**(n - 1 - i):
            binary_number.append(1)
            m -= 2**(n - 1 - i)
        else:
            binary_number.append(0)
    return binary_number


def rate_code(inputs, n_timesteps):
    scaled_inputs = []
    scaled_inputs.append((inputs[0]+1.5)/3)
    scaled_inputs.append((inputs[1]+1.5)/3)
    scaled_inputs.append((inputs[2]+5)/10)
    scaled_inputs.append((inputs[3]+5)/10)
    scaled_inputs.append((inputs[4]+3.15)/6.30)
    scaled_inputs.append((inputs[5]+5)/10)
    scaled_inputs.append(inputs[6])
    scaled_inputs.append(inputs[7])
    encoded_inputs = spikegen.rate(torch.tensor(scaled_inputs), num_steps = n_timesteps)
    return encoded_inputs.tolist()


def latency_code(inputs, n_timesteps):
    scaled_inputs = []
    scaled_inputs.append((inputs[0]+1.5)/3)
    scaled_inputs.append((inputs[1]+1.5)/3)
    scaled_inputs.append((inputs[2]+5)/10)
    scaled_inputs.append((inputs[3]+5)/10)
    scaled_inputs.append((inputs[4]+3.15)/6.30)
    scaled_inputs.append((inputs[5]+5)/10)
    scaled_inputs.append(inputs[6])
    scaled_inputs.append(inputs[7])
    encoded_inputs = spikegen.latency(torch.tensor(scaled_inputs), num_steps = n_timesteps, normalize=True, linear=True)
    return encoded_inputs.tolist()


def no_code(inputs, n_timesteps):
    scaled_inputs = []
    scaled_inputs.append((inputs[0]+1.5)/3)
    scaled_inputs.append((inputs[1]+1.5)/3)
    scaled_inputs.append((inputs[2]+5)/10)
    scaled_inputs.append((inputs[3]+5)/10)
    scaled_inputs.append((inputs[4]+3.15)/6.30)
    scaled_inputs.append((inputs[5]+5)/10)
    scaled_inputs.append(inputs[6])
    scaled_inputs.append(inputs[7])
    return [scaled_inputs.copy() for _ in range(n_timesteps)]


def phase_code(inputs, n_timesteps):
    
    scaled_inputs = []
    scaled_inputs.append((inputs[0]+1.5)/3)
    scaled_inputs.append((inputs[1]+1.5)/3)
    scaled_inputs.append((inputs[2]+5)/10)
    scaled_inputs.append((inputs[3]+5)/10)
    scaled_inputs.append((inputs[4]+3.15)/6.30)
    scaled_inputs.append((inputs[5]+5)/10)
    scaled_inputs.append(inputs[6])
    scaled_inputs.append(inputs[7])
    
    phase_coded_inputs = []

    max_representabile_number = 2**n_timesteps
    for i in range(len(scaled_inputs)):
        scaled_inputs[i] *= max_representabile_number

    for i in range(len(scaled_inputs)):
        phase_coded_inputs.append(closest_binary(n_timesteps, scaled_inputs[i]))

    phase_coded_inputs_transposed = list(zip(*phase_coded_inputs))

    return phase_coded_inputs_transposed


def eval(x, render=False):
    cumulative_rewards = []
    task = gym.make("LunarLander-v2")
    agent = SNN([8, 5, 4], 20)
    agent.set_params(x)
    for i in range(100):
        cumulative_rewards.append(0)
        done = False
        task.seed(i)
        obs = task.reset()
        counter = 0
        while not done:
            output = agent.activate(no_code(obs, 30))

            # arr[output]+=1
            if render:
                task.render()
            obs, rew, done, _ = task.step(np.argmax(output))
            cumulative_rewards[-1] += rew
        counter += 1

    return -sum(cumulative_rewards) / 100

def get_pruning_mask(snn, weights, prune_ratio):
    mask = np.ones_like(weights)
    c = 0
    snn.set_params(weights)
    for i in range(1,len(snn.neurons)):
        for j in range(len(snn.neurons[i])):
            num_to_prune = round(prune_ratio/100 * (len(snn.neurons[i-1])+1))
            neuron_weights = snn.weights[i-1][j]
            sorted_weights = np.sort(np.abs(neuron_weights))
            threshold = sorted_weights[num_to_prune-1]
            neuron_mask = np.ones_like(neuron_weights)
            neuron_mask[np.abs(neuron_weights) <= threshold] = 0
            for l in range(len(snn.neurons[i-1])):
                    mask[c] = neuron_mask[l]
                    c += 1
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


def save_scores(average_scores_history, best_scores_history, file_prefix='lunar_lander_'):
    with open(f'{file_prefix}average_scores.txt', 'w') as f:
        f.write(' '.join(map(str, average_scores_history)))
    with open(f'{file_prefix}best_scores.txt', 'w') as f:
        f.write(' '.join(map(str, best_scores_history)))


def plot_fitnesses(average_scores_history, best_scores_history):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(24,12))
    colors = cm.rainbow(np.linspace(0, 1, len(average_scores_history)))

    average_scores_history_concat = sum(average_scores_history, [])
    best_scores_history_concat = sum(best_scores_history, [])

    for i, fitness in enumerate(average_scores_history):
        x = list(range(sum(map(len, average_scores_history[:i])), sum(map(len, average_scores_history[:i+1]))))
        axes[0].plot(x, fitness, label=f'Density {round((0.8 ** i) * 100, 1)}%', color=colors[i])
    axes[0].axhline(y=-200, linestyle='--', color='black', label='Successful score')
    axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    axes[0].set_xlabel('Numero di generazioni')
    axes[0].set_ylabel('Punteggio medio')
    axes[0].set_title('Average scores')
    axes[0].set_ylim(bottom=min(min(best_scores_history_concat), min(average_scores_history_concat), -200)-30, top=max(max(best_scores_history_concat), max(average_scores_history_concat), -200)+30)

    for i, fitness in enumerate(best_scores_history):
        x = list(range(sum(map(len, best_scores_history[:i])), sum(map(len, best_scores_history[:i+1]))))
        axes[1].plot(x, fitness, label=f'Density {round((0.8 ** i) * 100, 1)}%', color=colors[i])
    axes[1].axhline(y=-200, linestyle='--', color='black', label='Successful score')
    axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    axes[1].set_xlabel('Numero di generazioni')
    axes[1].set_ylabel('Punteggio migliore')
    axes[1].set_title('Best guy scores')
    axes[1].set_ylim(bottom=min(min(best_scores_history_concat), min(average_scores_history_concat), -200)-30, top=max(max(best_scores_history_concat), max(average_scores_history_concat), -200)+30)

    plt.subplots_adjust(right=0.82)
    
    plt.savefig(f'./LunarLander_SNN_PhaseCode_PruningIteration#{len(average_scores_history)-1}.png')


if __name__ == "__main__":
    args = {}
    fka = SNN([8, 5, 4], 20)

    args["num_vars"] = fka.nweights + fka.nthresholds + fka.nbetas
    args["max_total_generations"] = 500
    args["added_generations"] = 50

    args["max_initial_generations"] = 500
    args["sigma"] = 1.0
    args["pop_size"] = 4
    args["num_offspring"] = 20
    args["pop_init_range"] = [0, 1]

    random = Random(0)

    es = cmaes(generator(random, args),
               args["sigma"],
               {'popsize': args["num_offspring"],
                'seed': 0,
                'CMA_mu': args["pop_size"],
                'bounds': [0, np.inf]})

    gen = 0
    n = 0
    pruning_mask = np.ones(args["num_vars"])
    initial_candidates = es.ask()
    candidates = initial_candidates
    fitnesses = np.zeros(args["num_offspring"])
    current_prune_ratio = 0

    average_scores_history = []
    best_scores_history = []
    
    while gen <= args["max_total_generations"]-args["max_initial_generations"]:
        average_scores_history.append([])
        best_scores_history.append([])
        for i in range(args["max_initial_generations"]+ args["added_generations"]*n):
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
            average_scores_history[n].append(np.mean(fitnesses))
            best_scores_history[n].append(min(fitnesses))
            print("Network density: "+str((0.8**(n))*100)+"%, Gen "+str(gen+1)+"  "+str(min(fitnesses))+"  "+str(np.mean(fitnesses)))
            es.tell(candidates, fitnesses)

            best_guy = es.best.x
            best = SNN([8, 5, 4], 20)
            best.set_params(best_guy)
            with open("thresholds-biases.txt", "a") as file:
                file.write("GEN " + str(gen+1) + "\n")
                file.write("thresholds:\n")
                d = 0
                for neuron in best.neurons:
                    for i in range(len(neuron)):
                        file.write(str(best.neurons[d][i].threshold) + "\n")
                    d += 1
                file.write("\n")
                file.write("betas:\n")
                d = 0
                for neuron in best.neurons:
                    for i in range(len(neuron)):
                        file.write(str(best.neurons[d][i].beta) + "\n")
                    d += 1
                file.write("\n\n")

            gen += 1
        print()
        plot_fitnesses(average_scores_history, best_scores_history)
        # calculate the pruning mask on the best guy in the current generation
        best_guy = es.best.x
        print(best_guy)
        best = SNN([8, 5, 4], 20)
        best.set_params(best_guy)
            
        current_prune_ratio += (100-current_prune_ratio)*fka.prune_ratio/100
        pruning_mask = get_pruning_mask(fka, best_guy, current_prune_ratio)
        
        n += 1
    final_pop = np.asarray(es.ask())
    final_pop_fitnesses = np.asarray(parallel_val(final_pop))
    
    best_guy = es.best.x
    best_fitness = es.best.f

    save_scores(average_scores_history, best_scores_history)

    with open("./nn_lunarlander_SNN_phasecode"+str(best_fitness)+".pkl", "wb") as f:
        pickle.dump(best_guy, f)
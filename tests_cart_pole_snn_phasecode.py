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


def inputs_resizing(inputs, params):
    output = []
    for i in range(len(inputs)):
        output.append(inputs[i]*params[-1-i])
    return output

def double_encoding(inputs):
    output = []
    for i in inputs:
        if i >= 0:
            output.append(0)
            output.append(i)
        else:
            output.append(-i)
            output.append(0)
    return output


def rate_code(inputs, n_timesteps):
    double_inputs = double_encoding(inputs)
    scaled_inputs = []
    scaled_inputs.append(math.tanh(double_inputs[0]))
    scaled_inputs.append(math.tanh(double_inputs[1]))
    scaled_inputs.append(math.tanh(double_inputs[2]))
    scaled_inputs.append(math.tanh(double_inputs[3]))
    scaled_inputs.append(math.tanh(double_inputs[4]))
    scaled_inputs.append(math.tanh(double_inputs[5]))
    scaled_inputs.append(math.tanh(double_inputs[6]))
    scaled_inputs.append(math.tanh(double_inputs[7]))
    encoded_inputs = spikegen.rate(torch.tensor(scaled_inputs), num_steps = n_timesteps)
    return encoded_inputs.tolist()


def latency_code(inputs, n_timesteps):
    double_inputs = double_encoding(inputs)
    scaled_inputs = []
    scaled_inputs.append(math.tanh(double_inputs[0]))
    scaled_inputs.append(math.tanh(double_inputs[1]))
    scaled_inputs.append(math.tanh(double_inputs[2]))
    scaled_inputs.append(math.tanh(double_inputs[3]))
    scaled_inputs.append(math.tanh(double_inputs[4]))
    scaled_inputs.append(math.tanh(double_inputs[5]))
    scaled_inputs.append(math.tanh(double_inputs[6]))
    scaled_inputs.append(math.tanh(double_inputs[7]))
    encoded_inputs = spikegen.latency(torch.tensor(scaled_inputs), num_steps = n_timesteps, normalize=True, linear=True)
    return encoded_inputs.tolist()


def no_code(inputs, n_timesteps):
    double_inputs = double_encoding(inputs)
    return [double_inputs.copy() for _ in range(n_timesteps)]

'''def no_code(inputs, n_timesteps):
    double_inputs = double_encoding(inputs)
    scaled_inputs = []
    scaled_inputs.append(math.tanh(double_inputs[0]))
    scaled_inputs.append(math.tanh(double_inputs[1]))
    scaled_inputs.append(math.tanh(double_inputs[2]))
    scaled_inputs.append(math.tanh(double_inputs[3]))
    scaled_inputs.append(math.tanh(double_inputs[4]))
    scaled_inputs.append(math.tanh(double_inputs[5]))
    scaled_inputs.append(math.tanh(double_inputs[6]))
    scaled_inputs.append(math.tanh(double_inputs[7]))
    return [scaled_inputs.copy() for _ in range(n_timesteps)]'''


def phase_code(inputs, n_timesteps):
    
    double_inputs = double_encoding(inputs)
    scaled_inputs = []
    scaled_inputs.append(math.tanh(double_inputs[0]))
    scaled_inputs.append(math.tanh(double_inputs[1]))
    scaled_inputs.append(math.tanh(double_inputs[2]))
    scaled_inputs.append(math.tanh(double_inputs[3]))
    scaled_inputs.append(math.tanh(double_inputs[4]))
    scaled_inputs.append(math.tanh(double_inputs[5]))
    scaled_inputs.append(math.tanh(double_inputs[6]))
    scaled_inputs.append(math.tanh(double_inputs[7]))
    
    phase_coded_inputs = []

    max_representabile_number = 2**n_timesteps
    scaled_inputs[0] *= max_representabile_number
    scaled_inputs[1] *= max_representabile_number
    scaled_inputs[2] *= max_representabile_number
    scaled_inputs[3] *= max_representabile_number
    scaled_inputs[4] *= max_representabile_number
    scaled_inputs[5] *= max_representabile_number
    scaled_inputs[6] *= max_representabile_number
    scaled_inputs[7] *= max_representabile_number

    phase_coded_inputs.append(closest_binary(n_timesteps, scaled_inputs[0]))
    phase_coded_inputs.append(closest_binary(n_timesteps, scaled_inputs[1]))
    phase_coded_inputs.append(closest_binary(n_timesteps, scaled_inputs[2]))
    phase_coded_inputs.append(closest_binary(n_timesteps, scaled_inputs[3]))
    phase_coded_inputs.append(closest_binary(n_timesteps, scaled_inputs[4]))
    phase_coded_inputs.append(closest_binary(n_timesteps, scaled_inputs[5]))
    phase_coded_inputs.append(closest_binary(n_timesteps, scaled_inputs[6]))
    phase_coded_inputs.append(closest_binary(n_timesteps, scaled_inputs[7]))

    phase_coded_inputs_transposed = list(zip(*phase_coded_inputs))

    return phase_coded_inputs_transposed


def eval(x, render=False):
    cumulative_rewards = []
    task = gym.make("CartPole-v1")
    agent = SNN([8, 4, 4, 2], 20)
    agent.set_params(x)
    for i in range(100):
        cumulative_rewards.append(0)
        done = False
        task.seed(i)
        obs = task.reset()
        counter = 0
        while not done:
            output = agent.activate(phase_code(inputs_resizing(obs, x), 30))

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


def save_scores(average_scores_history, best_scores_history, test_number, file_prefix):
    with open(f'{file_prefix}scores_run_{test_number}.txt', 'w') as f:
        f.write(' '.join(map(str, average_scores_history)))
        f.write('\n\n')
        f.write(' '.join(map(str, best_scores_history)))


def rimuovi_nan(lista):
    return [x for x in lista if not math.isnan(x)]

def leggi_dati_da_file(nome_file):
    with open(nome_file, 'r') as file:
        dati = file.read()
    # Divide i dati in average_scores_history e best_scores_history
    dati_divisi = dati.split('\n\n')

    scores_history = []
    
    i = 0

    # Trova la lunghezza massima delle sottoliste per riempimento
    max_len = max([len(item.split(',')) for contenuto in dati_divisi for item in contenuto.split('] [')])

    for contenuto in dati_divisi:
        scores_history.append([])
        contenuto = contenuto.strip('[] ')
        liste_interne = contenuto.split('] [')
        
        for lista_stringa in liste_interne:
            lista_stringa = lista_stringa.strip()
            valori_stringa = lista_stringa.split(',')
            lista_float = [float(valore) for valore in valori_stringa]
            
            # Riempimento della lista se è più corta della lunghezza massima
            while len(lista_float) < max_len:
                lista_float.append(np.nan)
                
            scores_history[i].append(lista_float)
        i += 1

    return scores_history[0], scores_history[1]


def plot_average_fitnesses(files, output_file_name):
    all_avg_scores = []
    all_best_scores = []

    for file in files:
        average_scores, best_scores = leggi_dati_da_file(file)
        all_avg_scores.append(average_scores)
        all_best_scores.append(best_scores)

    # Calcola la media, il valore minimo e il valore massimo per i dati
    avg_scores_mean = np.mean(all_avg_scores, axis=0)
    avg_scores_min = np.min(all_avg_scores, axis=0)
    avg_scores_max = np.max(all_avg_scores, axis=0)
    lower_error_avg = avg_scores_mean - avg_scores_min
    upper_error_avg = avg_scores_max - avg_scores_mean

    best_scores_mean = np.mean(all_best_scores, axis=0)
    best_scores_min = np.min(all_best_scores, axis=0)
    best_scores_max = np.max(all_best_scores, axis=0)
    lower_error_best = best_scores_mean - best_scores_min
    upper_error_best = best_scores_max - best_scores_mean

    avg_scores_mean_cut = []
    avg_scores_min_cut = []
    avg_scores_max_cut = []
    lower_error_avg_cut = []
    upper_error_avg_cut = []

    best_scores_mean_cut = []
    best_scores_min_cut = []
    best_scores_max_cut = []
    lower_error_best_cut = []
    upper_error_best_cut = []

    for i in range(len(avg_scores_mean)):

        avg_scores_mean_cut.append(rimuovi_nan(avg_scores_mean[i]))
        avg_scores_min_cut.append(rimuovi_nan(avg_scores_min[i]))
        avg_scores_max_cut.append(rimuovi_nan(avg_scores_max[i]))
        lower_error_avg_cut.append(rimuovi_nan(lower_error_avg[i]))
        upper_error_avg_cut.append(rimuovi_nan(upper_error_avg[i]))

        best_scores_mean_cut.append(rimuovi_nan(best_scores_mean[i]))
        best_scores_min_cut.append(rimuovi_nan(best_scores_min[i]))
        best_scores_max_cut.append(rimuovi_nan(best_scores_max[i]))
        lower_error_best_cut.append(rimuovi_nan(lower_error_best[i]))
        upper_error_best_cut.append(rimuovi_nan(upper_error_best[i]))


    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(24,12))
    colors = cm.rainbow(np.linspace(0, 1, len(avg_scores_mean)))

    for i in range(len(avg_scores_mean_cut)):
        x = list(range(sum(map(len, avg_scores_mean_cut[:i])), sum(map(len, avg_scores_mean_cut[:i+1]))))
        axes[0].errorbar(x, avg_scores_mean_cut[i], yerr=[lower_error_avg_cut[i], upper_error_avg_cut[i]], label=f'Density {round((0.8 ** i) * 100, 1)}%', color=colors[i], alpha=0.7)
        print("MEDIA AVERAGE SCORES [" + str(i) + "]:")
        print(avg_scores_mean_cut[i])
    
    axes[0].axhline(y=-475, linestyle='--', color='black', label='Successful score')
    axes[0].set_xlabel('Numero di generazioni')
    axes[0].set_ylabel('Punteggio medio')
    axes[0].set_title('Average scores')

    for i in range(len(best_scores_mean_cut)):
        x = list(range(sum(map(len, best_scores_mean_cut[:i])), sum(map(len, best_scores_mean_cut[:i+1]))))
        axes[1].errorbar(x, best_scores_mean_cut[i], yerr=[lower_error_best_cut[i], upper_error_best_cut[i]], label=f'Density {round((0.8 ** i) * 100, 1)}%', color=colors[i], alpha=0.7)
        print("MEDIA BEST SCORES [" + str(i) + "]:")
        print(best_scores_mean_cut[i])

    axes[1].axhline(y=-475, linestyle='--', color='black', label='Successful score')
    axes[1].set_xlabel('Numero di generazioni')
    axes[1].set_ylabel('Punteggio migliore')
    axes[1].set_title('Best guy scores')

    # Includi le legende e regola gli spazi
    for ax in axes:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    plt.subplots_adjust(right=0.82)

    plt.savefig(f'./' + output_file_name)


def cart_pole_SNN_phasecode():
        
    for tests in range(5): # esegue 5 test ed effettua il grafico con medie e error bars

        print("TEST " + str(tests) + " CART POLE SNN PHASECODE")

        args = {}
        fka = SNN([8, 4, 4, 2], 20)

        args["num_vars"] = fka.nweights + fka.nthresholds + fka.nbetas + len(fka.neurons[0])
        args["max_total_generations"] = 500
        args["added_generations"] = 50
        args["max_initial_generations"] = 100
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

        '''es = cmaes(generator(random, args),
                args["sigma"],
                {'popsize': args["num_offspring"],
                    'seed': 0,
                    'CMA_mu': args["pop_size"]})'''

        gen = 0
        n = 0
        pruning_mask = np.ones(args["num_vars"])
        initial_candidates = es.ask()
        candidates = initial_candidates
        fitnesses = np.zeros(args["num_offspring"])
        current_prune_ratio = 0

        average_scores_history = []
        best_scores_history = []
        
        while gen <= args["max_total_generations"]-args["max_initial_generations"]-1:
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
                best = SNN([8, 4, 4, 2], 20)
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
            #plot_average_fitnesses(average_scores_history, best_scores_history)
            # calculate the pruning mask on the best guy in the current generation
            best_guy = es.best.x
            print(best_guy)
            best = SNN([8, 4, 4, 2], 20)
            best.set_params(best_guy)
            

            '''
            print("thresholds:")
            d = 0
            for neuron in best.neurons:
                for i in range(len(neuron)):
                    print(best.neurons[d][i].threshold)
                d += 1
            print()
            print("betas:")
            d = 0
            for neuron in best.neurons:
                for i in range(len(neuron)):
                    print(best.neurons[d][i].beta)
                d += 1
            '''
                
            current_prune_ratio += (100-current_prune_ratio)*fka.prune_ratio/100
            pruning_mask = get_pruning_mask(fka, best_guy, current_prune_ratio)
            
            n += 1
        final_pop = np.asarray(es.ask())
        final_pop_fitnesses = np.asarray(parallel_val(final_pop))
        
        best_guy = es.best.x
        best_fitness = es.best.f

        save_scores(average_scores_history, best_scores_history, tests, "cart_pole_SNN_phasecode_")

        with open("./nn_cartpole_SNN_phasecode"+str(best_fitness)+".pkl", "wb") as f:
            pickle.dump(best_guy, f)

    plot_average_fitnesses(['cart_pole_SNN_phasecode_scores_run_0.txt', 'cart_pole_SNN_phasecode_scores_run_1.txt', 'cart_pole_SNN_phasecode_scores_run_2.txt', 'cart_pole_SNN_phasecode_scores_run_3.txt', 'cart_pole_SNN_phasecode_scores_run_4.txt'], "CartPole_SNN_phasecode.png")
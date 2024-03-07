from deap import base, creator, tools, algorithms
import numpy as np
from scipy.special import gamma
from joblib import Parallel, delayed
from bisect import bisect_right
import torch
import matplotlib.pyplot as plt
import gc

from cProfile import Profile
from pstats import SortKey, Stats
import os
import psutil


def eaGenerateUpdateBatched(toolbox, train_loader, ngen, halloffame=None, stats=None, run_parallel=False, verbose=__debug__):
    """
    This function is a modified version of the eaGenerateUpdate function of the DEAP library.
    It is used to train the network in a batched way. At every generation, the network is trained on a different batch
    :param toolbox: toolbox
    :param loader: the dataloader with positive and negative examples
    :param ngen: number of generations
    :param halloffame: hall of fame object
    :param stats: statistics recorder
    :param verbose: tell if the function should print the statistics
    :return: the final population and the logbook
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    step = 0

    with Profile() as prof:
        for _ in range(ngen):
            for batch in train_loader:
                toolbox.set_batch(batch)

                # Generate a new population
                population = toolbox.generate()
                weights = []

                if run_parallel:
                    # Parallel evaluation of individuals
                    fitnesses, weights = Parallel(n_jobs=-1)(delayed(toolbox.evaluate)(ind) for ind in population)

                    for ind, (fit, w) in zip(population, fitnesses):
                        weights.append(w)
                        ind.fitness.values = fit,

                else:
                    # Evaluate the individuals
                    # print_memory_usage(f"Generation {step}, before evaluation")
                    fitnesses = toolbox.map(toolbox.evaluate, population)
                    for ind, (fit, w) in zip(population, fitnesses):
                        # print_memory_usage(f"Generation {step}, after evaluation")
                        weights.append(w)
                        ind.fitness.values = fit,

                # Update the hall of fame with the generated individuals
                if halloffame is not None:
                    halloffame.update(population)

                # Update the strategy with the evaluated individuals
                toolbox.update(population)

                # get index of best individual
                best_index = np.argmax(fitnesses)
                toolbox.set_best_weights(weights[best_index])

                # Append the current generation statistics to the logbook
                record = stats.compile(population) if stats is not None else {}
                logbook.record(gen=step, nevals=len(population), **record)
                if verbose:
                    print(logbook.stream)

                # Update the current generation

                step += 1


    return population, logbook




def create_torch_stats():
    """
    Create the statistics for the genetic algorithm
    """
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", stat_on_tensor, np.mean)
    stats.register("std", stat_on_tensor, np.std)
    stats.register("min", stat_on_tensor, np.min)
    stats.register("max", stat_on_tensor, np.max)
    return stats


def stat_on_tensor(op, *args):
    """ Apply a numpy operation on a tensor """
    return op([arg.cpu().detach().numpy() for arg, in args[0]])


def plot_res(logbook):
    """ Plot the results of the genetic algorithm"""
    gen = logbook.select("gen")

    fig, ax = plt.subplots()
    #ax.plot(gen, logbook.select("avg"), "r-", label="Average Size")
    ax.plot(gen, logbook.select("max"), "b-", label="Maximum Fitness")
    ax.plot(gen, logbook.select("min"), "g-", label="Minimum Fitness")
    ax.legend()



class TorchHallOfFame(tools.HallOfFame):
    """
    Hall of fame for the genetic algorithm. It is used to store the best individuals.
    It is a modified version of the HallOfFame class of the DEAP library.
    I needed to modify it because the legacy uses deepcopy, which is not supported by pytorch.
    """
    def __init__(self, ind_init, maxsize=0, similar=np.array_equal):
        super().__init__(maxsize, similar)
        self.ind_init = ind_init

    def insert(self, item):
        item2 = item.clone().detach()
        i = bisect_right(self.keys, item.fitness)
        ind = self.ind_init(item2)
        ind.fitness = item.fitness
        self.items.insert(len(self) - i, ind)
        self.keys.insert(i, ind)


def print_memory_usage(description=""):
    process = psutil.Process(os.getpid())
    print(f"{description} Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

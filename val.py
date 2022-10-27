import os

from cma import CMAEvolutionStrategy as cmaes
import gym
from network import RNN
import numpy as np
import functools
from random import Random
from multiprocessing import Pool
import pickle
import matplotlib.pyplot as plt
import networkx as nx


def eval(x, render=False, episodes=100):
    cumulative_rewards = []
    task = gym.make("LunarLander-v2")
    agent = RNN([8, 5, 4], 60, 0.01, 0, False)
    agent.set_hrules(x)
    for i in range(episodes):

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
            if counter < 20:
                agent.update_weights()

        counter += 1
        if counter == 20:
            agent.prune_weights()
    return cumulative_rewards, agent


if __name__ == "__main__":
    g = None
    for ind in [38,42,45,52]:
        with open("./pkl/best_"+str(ind)+".pkl", "rb") as f:
            g = pickle.load(f)
        print(type(g))

        os.makedirs("./ndr_results", exist_ok=True)
        fold = "./ndr_results"

        res, agent = eval(g, episodes=100)
        print(sum(res[:20]))
        print(sum(res[20:]))
        print(np.mean(res))
        plt.plot(res)
        plt.axvline(20)
        plt.savefig("n_behave.png")
        plt.clf()

        res, agent = eval(g, episodes=20)
        print(np.mean(res))
        pregraph = nx.from_numpy_matrix(agent.weights, create_using=nx.DiGraph)
        pos = agent.nx_pos_because_it_was_too_hard_to_add_a_multipartite_from_a_list_as_it_works_for_the_bipartite(
            pregraph)
        plt.clf()
        nx.draw(pregraph, pos=pos, with_labels=True, font_weight='bold')
        plt.savefig(fold + "/initial_"+str(ind)+".png")
        agent.prune_weights()
        pregraph = nx.from_numpy_matrix(agent.weights, create_using=nx.DiGraph)
        pos = agent.nx_pos_because_it_was_too_hard_to_add_a_multipartite_from_a_list_as_it_works_for_the_bipartite(
            pregraph)
        plt.clf()
        nx.draw(pregraph, pos=pos, with_labels=True, font_weight='bold')
        plt.savefig(fold + "/pruned_"+str(ind)+".png")
        dag, hc = agent.dag("./ndr_results")
        top_sort = nx.topological_sort(dag)
        agent.top_sort = list(top_sort)
        agent.cycle_history = hc
        print(agent.cycle_history)
        print(agent.top_sort)
        print(nx.to_dict_of_lists(dag))
        plt.clf()
        pos = agent.nxpbiwthtaamfalaiwftb(dag)
        nx.draw(dag, pos=pos, with_labels=True, font_weight='bold')
        plt.savefig(fold + "/final_dag_"+str(ind)+".png")
        with open(fold + "/hc_"+str(ind)+".txt", "wb") as f:
            pickle.dump(hc, f)

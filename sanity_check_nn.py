import os

from network import RNN, NN
import gym
import numpy as np
import pickle
import nevergrad as ng
def eval(data, render=False, sdir=None, seed=None):
    x = data[0]
    args = data[1]
    cumulative_rewards = []
    task = gym.make("LunarLander-v2")
    agent = RNN([8, args["hnodes"], 4], args["prate"], 0.01, args["seed"], False)

    agent.set_hrules(x)
    pf = False
    for i in range(100):
        cumulative_rewards.append(0)
        done = False
        task.seed(i)
        obs = task.reset()

        while not done:
            output = agent.activate(obs)

            # arr[output]+=1
            if render:
                task.render()
            act = np.argmax(output)

            obs, rew, done, _ = task.step(act)


            cumulative_rewards[-1] += rew
            if i < max(args["ps"]):
                agent.update_weights()
        if i in args["ps"]:
            os.makedirs("networks_img/"+sdir, exist_ok=True)
            agent.prune_weights(os.path.join("networks_img/",sdir,seed))

            pf =True
    return agent, cumulative_rewards

def eval1(ds, render=False, sdir=None, seed=None):
    x = ds[0]
    args = ds[1]
    cumulative_rewards = []
    task = gym.make("LunarLander-v2")
    agent = NN([8, args["hnodes"], 4])
    agent.set_weights(x)
    if sdir is not None:
        os.makedirs("networks_img_nn/" + sdir, exist_ok=True)
        agent.nn_prune_weights(args["prate"], os.path.join("networks_img_nn/",sdir,seed))
    arr = [0,0,0]
    for i in range(100):
        cumulative_rewards.append(0)
        done = False
        task.seed(i)
        obs = task.reset()
        counter = 0
        while not done:
            output = agent.activate(obs)
            if render:
                task.render()
            obs, rew, done, _ = task.step(np.argmax(output))
            cumulative_rewards[-1] += rew
        counter += 1
    return agent, cumulative_rewards

if __name__=="__main__":
    pre = []
    pop = []
    fdir = "C:\\Users\\opuse\\Desktop\\ndr_hs\\fin_res\\results_NN_ll"
    for ps in [20]:
        for pr in [20,40,60,80]:
            for hn in range(5,10):

                dir =os.path.join(fdir, str(hn), str(pr))
                print(dir)
                for seed in range(30):
                    listDir = os.listdir(dir+"/"+str(seed))
                    fl = None
                    for fn in listDir:
                        if (not fn.startswith("best") ) and fn.endswith("pkl"):
                            fl = fn
                    if fl is not None:
                        x = pickle.load(open(dir+"/"+str(seed)+"/"+fl, "rb"))
                        _, rew = eval1([x, {"seed":seed, "ps":[ps], "prate":pr, "hnodes":hn}], render=False, sdir=os.path.join(str(hn), str(pr)), seed =str(seed))
                        pre.append(np.mean(rew[:ps]))
                        pop.append(np.mean(rew[ps:]))
                print((ps,pr,hn))
                print("pre "+str(np.mean(pre))+" "+str(np.std(pre)))
                print("post " + str(np.mean(pop)) + " " + str(np.std(pop)))
                print("==================================================")
    '''
    instrum = ng.p.Instrumentation(ng.p.Array(shape=(2,)), y=ng.p.Scalar())  # We are working on R^2 x R.
    optimizer = ng.optimizers.OnePlusOne(parametrization=instrum, budget=100, num_workers=1)
    print(optimizer)
    x = NN([2,5,5,3])
    x.set_weights(np.random.uniform(-1,1,x.nweights))
    print(x.activate([-1,1]))
    '''


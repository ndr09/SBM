from torchvision import datasets, transforms
import torch.nn.functional as F
import torch
from network_pt import *
import numpy as np
import time

def eval_minst(data):
    x = data[0]
    args = data[1]
    agent = NHNN([28*28, 100, 100, 100, 10], 0.000001)
    agent.set_hrules(x)
    agent.to(agent.device)
    outputs = None
    targets = None
    times = []
    count = 0
    w = []
    for batch_idx, (data, target) in enumerate(args["tl"]):

        data = data.to(agent.device)
        target = target.to(agent.device)

        for i  in range(len(data)):
            out = F.log_softmax(agent.forward(data[i].flatten()), dim=0)
            w.append([ torch.max(l).item() for l in agent.get_weights()])
            outputs = out if outputs is None else torch.vstack((outputs, out))
            targets = target[i] if targets is None else torch.hstack((targets, target[i]))
            agent.update_weights()

        count += 1
        if count == 10:
            break

    # print(w)
    # print(outputs.size())
    # print(targets.size())

    loss = F.nll_loss(outputs, targets)

    return loss.item()


if __name__=="__main__":
    tl =  3 * (28*28) + 4 * 300 + 3 * 10
    rng= np.random.default_rng()
    x = [rng.random()*2-1 for i in range(tl)]
    args = {}
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    train_kwargs = {'batch_size': 32}
    args["tl"] = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    times= []
    for i in range(100):
        t = time.time()
        eval_minst((x,args))
        times.append((time.time()-t))
    print(np.mean(times))


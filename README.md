# Self BuildingNeural Network
---
The self-building NN has a mechanism that allows it to grow and prune its connection at runtime. 
They achieve this using a plasticity model (Hebbian Learning) and a pruning algorithm. 

---
# Usage
The following code initializes an SBNN with 2 inputs, 5 hidden nodes, and 3 outputs, Using $\eta=0.1$ and pruning ratio $pr=40%$.
Note that the last two parameters refer to the random activation order, the first is the seed, and the second refers to the use of a random activation order, before the pruning.

```
from network import SBM
sbnn = SBM([2,5,3], 40, 0.1, 0, True) 
```

---
# Citing
If you like this project, we would appreciate it if you starred the repository in order to help us increase its visibility. Furthermore, if you find the framework useful in your research, we would be grateful if you could cite our [publication](https://arxiv.org/abs/2304.01086) using the following bibtex entry:

```bib
@misc{ferigo2023sbm,
      title={Self-building Neural Networks}, 
      author={Andrea Ferigo and Giovanni Iacca},
      year={2023},
      eprint={2304.01086},
      archivePrefix={arXiv},
      primaryClass={cs.NE}
}
```


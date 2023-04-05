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
# References
If you use this work, please cite us :) 
- [Self Building Neural Networks](https://arxiv.org/abs/2304.01086)


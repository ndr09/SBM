# HostNN Learning

# Usage

The repository contains two example files, namely `gym_task_single_hostNN.py` and `gym_task_multiple_hostNN.py`, which demonstrate the usage of the classes provided in the package. Running either of these files will execute the example code and save a pickle file consisting of the weights of the best individual discovered by CMA-ES.

After the neural networks have been trained, use the `val_host_nn.py` file to execute them. Before execution, modify the path of the pickle file containing all the information about the neural network, including its type, structure, etc.

This also applies similarly to the **HNNhost** counterpart.

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


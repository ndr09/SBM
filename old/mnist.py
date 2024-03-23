import torch
from torchvision import datasets, transforms
from utility.HebbianNetworkClassifier import HebbianNetworkClassifier
import matplotlib.pyplot as plt
import numpy as np
import pickle
import wandb
from utility.transforms import PermutePixels, RandomBoxOverlay

# set seed for reproducibility
torch.manual_seed(42)
torch.mps.manual_seed(42)

# torch.autograd.set_detect_anomaly(True)


# Define the transformation to apply to the data
transform_train = transforms.Compose([
    transforms.ToTensor(),
    # transforms.GaussianBlur(5, sigma=(1, 1.)),
    transforms.Normalize((0.1307,), (0.3081,))
])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

batch_size = 64
# Download and load the MNIST training dataset
trn_dataset = datasets.MNIST(root='./data', train=True, transform=transform_train, download=True)
trn_loader = torch.utils.data.DataLoader(trn_dataset, batch_size=batch_size, shuffle=True)

# Split the training dataset into training and validation datasets
train_size = int(0.85 * len(trn_dataset))
val_size = len(trn_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(trn_dataset, [train_size, val_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)


# Download and load the MNIST test dataset
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# permuted datasets -> not used for now
# permutation = PermutePixels.random_permutation(28*28)
# transform_permuted = transforms.Compose([
#     transforms.ToTensor(),
#     PermutePixels(permutation),
#     transforms.Normalize((0.1307,), (0.3081,))
# ])
# permuted_train_dataset = datasets.MNIST(root='./data', train=True, transform=transform_permuted, download=True)
# permuted_train_loader = torch.utils.data.DataLoader(permuted_train_dataset, batch_size=batch_size, shuffle=True)
# permuted_test_dataset = datasets.MNIST(root='./data', train=False, transform=transform_permuted, download=True)
# permuted_test_loader = torch.utils.data.DataLoader(permuted_test_dataset, batch_size=64, shuffle=False)

# Print the number of samples in the training and test datasets
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

model_size = [784, 256, 10]
model = HebbianNetworkClassifier(
    model_size, 
    device=device, 
    init="linear",
    # dropout=0.1,
    # bias=True,
    activation=torch.functional.F.tanh,
    neuron_centric=True,
    use_d=True,
    use_tatgets=False
)

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)

# print num parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters: {num_params}")

log = False
if log: wandb.init(
    # st the wandb project where this run will be logged
    project="neuro-hebbian-learning-mnist-short",
        
    # track hyperparameters and run metadata
    config={ 
        "learning_rate": optimizer.defaults["lr"],
        "weight_decay": optimizer.defaults["weight_decay"],
        "architecture": f"{model_size}",
        "dataset": "MNIST",
        "batch_size": batch_size,
    }
)

train_loss, val_loss, test_loss, train_accuracy, val_accuracy, test_accuracy, confusion_matrix = model.train_loop(
    optimizer, loss_fn, train_loader, val_loader, test_loader, epochs=10, log=log, reset_every=-1, scheduler=scheduler, backprop_every=50, reset_every_batch=None
)


print(f"Test accuracy: {test_accuracy}")


# train only with hebbian
train_loss, val_loss, test_loss, train_accuracy, val_accuracy, test_accuracy, confusion_matrix = model.hebbian_train_loop(
    loss_fn, trn_loader, None, test_loader, max_iter=704, log=log, epochs=10, reset=False
) # used trn_loader instead of train_loader and None instead of val_loader

print(f"Test accuracy hebbian: {test_accuracy}")

if log: wandb.finish()
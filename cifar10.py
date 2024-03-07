import torch
from torchvision import datasets, transforms
from utility.HebbianNetworkClassifier import HebbianNetworkClassifier
import matplotlib.pyplot as plt
import numpy as np
import pickle
import wandb

# set seed for reproducibility
torch.manual_seed(42)

# torch.autograd.set_detect_anomaly(True)

# Define the transformation to apply to the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # flatten on the channel dimension
    transforms.Lambda(lambda x: x.view(x.size(0), -1))
])

batch_size = 64
# Download and load the CIFAR-10 training dataset
trn_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
trn_loader = torch.utils.data.DataLoader(trn_dataset, batch_size=batch_size, shuffle=True)

# Split the training dataset into training and validation datasets
train_size = int(0.85 * len(trn_dataset))
val_size = len(trn_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(trn_dataset, [train_size, val_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1024, shuffle=True)


# Download and load the MNIST test dataset
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False)

# Print the number of samples in the training and test datasets
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model_size = [32*32*3, 1024, 512, 10]
model = HebbianNetworkClassifier(
    model_size, 
    device=device, 
    init="linear",
    dropout=0.1,
    bias=False,
    activation=torch.functional.F.tanh,
)

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)

log = True
if log: wandb.init(
    # st the wandb project where this run will be logged
    project="neuro-hebbian-learning-cifar",
        
    # track hyperparameters and run metadata
    config={
        "learning_rate": optimizer.defaults["lr"],
        "architecture": f"{model_size}",
        "dataset": "MNIST",
        "batch_size": train_loader.batch_size,
    }
)

train_loss, val_loss, test_loss, train_accuracy, val_accuracy, test_accuracy, confusion_matrix = model.train_loop(
    optimizer, loss_fn, train_loader, val_loader, test_loader, epochs=25, log=log, scheduler=scheduler # reset_every=1, backprop_every=5
)

print(f"Test accuracy: {test_accuracy}")


# train only with hebbian
train_loss, val_loss, test_loss, train_accuracy, val_accuracy, test_accuracy, confusion_matrix = model.hebbian_train_loop(
    loss_fn, train_loader, val_loader, test_loader, max_iter=300, log=log
) # used trn_loader instead of train_loader and None instead of val_loader

print(f"Test accuracy hebbian: {test_accuracy}")

if log: wandb.finish()
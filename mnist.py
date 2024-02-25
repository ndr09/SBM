import torch
from torchvision import datasets, transforms
from utility.HebbianNetworkClassifier import HebbianNetworkClassifier
import matplotlib.pyplot as plt
import numpy as np
import pickle
import wandb

# set seed for reproducibility
torch.manual_seed(42)

# torch.autograd.set_detect_anomaly(True)

# Define the transformation to apply to the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# Download and load the MNIST training dataset
trn_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
trn_loader = torch.utils.data.DataLoader(trn_dataset, batch_size=64, shuffle=True)

# Split the training dataset into training and validation datasets
train_size = int(0.8 * len(trn_dataset))
val_size = len(trn_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(trn_dataset, [train_size, val_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=512, shuffle=True)


# Download and load the MNIST test dataset
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False)

# Print the number of samples in the training and test datasets
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model_size = [784, 512, 256, 10]
model = HebbianNetworkClassifier(model_size, device=device, init="uni", dropout=0.1)

# if the model already exists, load it from the pickle file
# otherwise train it
loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0000001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)

epochs = 3

log = False
if log:
    wandb.init(
    # st the wandb project where this run will be logged
        project="neuro-hebbian-learning-mnist",
        
        # track hyperparameters and run metadata
        config={
            "learning_rate": optimizer.defaults["lr"],
            "architecture": f"{model_size}",
            "dataset": "MNIST",
            "epochs": epochs,
            "batch_size": train_loader.batch_size,
        }
    )


train_loss, val_loss, test_loss, train_accuracy, val_accuracy, test_accuracy = model.train_loop(
     optimizer, loss_fn, train_loader, val_loader, test_loader, epochs=epochs, scheduler=scheduler, reset_every=1
)



# train only with hebbian
train_loss, val_loss, test_loss, train_accuracy, val_accuracy, test_accuracy = model.train_loop(
    optimizer, loss_fn, trn_loader, None, test_loader, epochs=10, backprop=False
) # used trn_loader instead of train_loader and None instead of val_loader

wandb.finish()



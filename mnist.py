import torch
from torchvision import datasets, transforms
from utility.HebbianNetwork import HebbianNetwork
import matplotlib.pyplot as plt
import numpy as np
import pickle

# set seed for reproducibility
torch.manual_seed(42)

# Define the transformation to apply to the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download and load the MNIST training dataset
trn_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
trn_loader = torch.utils.data.DataLoader(trn_dataset, batch_size=32, shuffle=True)

# Split the training dataset into training and validation datasets
train_size = int(0.8 * len(trn_dataset))
val_size = len(trn_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(trn_dataset, [train_size, val_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=True)


# Download and load the MNIST test dataset
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

# Print the number of samples in the training and test datasets
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model = HebbianNetwork([784, 512, 256, 10], device=device, init="uni")

# if the model already exists, load it from the pickle file
# otherwise train it
loss_fn = torch.nn.CrossEntropyLoss()


optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.000001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)

train_loss, val_loss, test_loss, train_accuracy, val_accuracy, test_accuracy = model.train_loop(
    optimizer, loss_fn, train_loader, val_loader, test_loader, epochs=100, scheduler=scheduler
)

print(f"Validation loss: {val_loss[-10:]}")
print(f"Validation accuracy: {val_accuracy[-10:]}")
print(f"Test loss: {np.mean(test_loss)}")
print(f"Test accuracy: {np.mean(test_accuracy)}")

plt.plot(train_loss, label="Train")
plt.plot(val_loss, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
# save the plot
plt.savefig('loss.png')

# clear the plot
plt.clf()

plt.plot(train_accuracy, label="Train")
plt.plot(val_accuracy, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('accuracy.png')

train_loss, test_loss, test_accuracy = model.train_weights(loss_fn, trn_loader, test_loader, epochs=1)

print(f"Test loss: {np.mean(test_loss)}")
print(f"Test accuracy: {np.mean(test_accuracy)}")



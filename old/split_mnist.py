import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from utility.HebbianNetworkClassifier import HebbianNetworkClassifier
import wandb

# set seed for reproducibility
torch.manual_seed(42)
torch.mps.manual_seed(42)

torch.autograd.set_detect_anomaly(True)

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.1307,), (0.3081,)) # Standard MNIST normalization
])

# Download datasets if not already available
train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)

tasks = [
    [0, 1],  # Task 1: Digits 0 vs 1
    [2, 3],  # Task 2: Digits 2 vs 3
    [4, 5], 
    [6, 7],
    [8, 9] 
]


def create_single_head_split(dataset, task_classes):
    """Creates a subset for a single-headed task"""
    indices = torch.where(torch.logical_or(dataset.targets == task_classes[0], 
                                           dataset.targets == task_classes[1]))[0]
    # set all targets to 0 or 1
    targets = (dataset.targets[indices] == task_classes[1]).long()
    dataset.targets[indices] = targets
    return Subset(dataset, indices)

def split_train_val(dataset, val_size=0.15):
    """Splits a dataset into training and validation sets"""
        
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    return torch.utils.data.random_split(dataset, [train_size, val_size])


train_datasets = []
val_datasets = []
test_datasets = []

for task in tasks:
    train_dataset, val_dataset = split_train_val(create_single_head_split(train_data, task))
    train_datasets.append(train_dataset)
    val_datasets.append(val_dataset)
    test_datasets.append(create_single_head_split(test_data, task))

batch_size = 64

train_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True) 
                    for dataset in train_datasets]
val_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=False)
                     for dataset in val_datasets]
test_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=False) 
                   for dataset in test_datasets]

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model_size = [784, 256, 2]
lr = 0.001
wd = 0.001
model = HebbianNetworkClassifier(
    model_size, 
    device=device, 
    init="linear",
    # dropout=0.1,
    bias=False,
    activation=torch.functional.F.relu,
    neuron_centric=True,
    use_d=True,
    num_classes=2,
    train_weights=False,
    use_tatgets=False
)

loss_fn = torch.nn.CrossEntropyLoss()


# print num parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters: {num_params}")

log = False
if log: wandb.init(
    # st the wandb project where this run will be logged
    project="neuro-hebbian-learning-split-mnist",
        
    # track hyperparameters and run metadata
    config={ 
        "learning_rate": lr,
        "weight_decay": wd,
        "architecture": f"{model_size}",
        "dataset": "MNIST",
        "batch_size": batch_size,
    }
)
    
for i, (train_loader, val_loader, test_loader) in enumerate(zip(train_loaders, val_loaders, test_loaders)):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

   #if i == 0:
    train_loss, val_loss, test_loss, train_accuracy, val_accuracy, test_accuracy, confusion_matrix = model.train_loop(
        optimizer, loss_fn, train_loader, val_loader, test_loader, epochs=2, log=log, reset_every=-1, #  early_stop=80
    )
    print(f"Test accuracy for task {tasks[i]}: {test_accuracy}")
    # if i == 2: break

model.reset_weights()
print("----Hebbian training-----")

# reverse order
tasks = tasks[::-1]
train_loaders = train_loaders[::-1]
val_loaders = val_loaders[::-1]
test_loaders = test_loaders[::-1]

for i, (val_loader, test_loader) in enumerate(zip(val_loaders, test_loaders)):
    train_loss, val_loss, test_loss, train_accuracy, val_accuracy, test_accuracy, confusion_matrix = model.hebbian_train_loop(
            loss_fn, val_loader, None, test_loader, epochs=30, max_iter=3000, log=log, reset=False
    )

    print(f"Test accuracy for task {tasks[i]}: {test_accuracy}")
    # if i == 2: break

print("----Final test-----")
for i, test_loader in enumerate(test_loaders):
    test_loss, test_accuracy, confusion_matrix = model.test(test_loader, loss_fn, log=log, online=True)
    print(f"Test accuracy for task {tasks[i]}: {test_accuracy}")
    # if i == 2: break

if log: wandb.finish()
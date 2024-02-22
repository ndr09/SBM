import torch
import torch.nn as nn
from utility.HebbianLinearLayer import HebbianLinearLayer
from tqdm import tqdm
import sklearn.metrics
import wandb

class HebbianNetwork(nn.Module):

    def __init__(
            self, 
            layers: list, 
            init,
            device='cpu',
    ) -> None:
        super(HebbianNetwork, self).__init__()
        self.layers = nn.modules.ModuleList()
        self.device = device

        self.layer_list = layers

        for i in range(len(layers) - 1):
            last_layer = i == len(layers) - 2
            if last_layer:
                self.num_output = layers[i + 1]
            self.layers.append(HebbianLinearLayer(layers[i], layers[i + 1], device=device, last_layer=last_layer))
            if i > 0:
                self.layers[i - 1].attach_hebbian_layer(self.layers[i])

        self.reset_weights(init)
        self.float()

    def learn(self, input):
        self.reset_weights('mantain')
        for layer in self.layers:
            input = layer.learn(input)
        return input

    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        return input
    
    def reset_weights(self, init='uni'):
        for layer in self.layers:
            layer.reset_weights(init)

    def train_loop(self, optimizer, loss_fn, train_dataloader, val_dataloader, test_dataloader, epochs=10, scheduler=None):
        train_loss = []
        val_loss = []
        train_accuracy = []
        val_accuracy = []
        test_loss = 0.0
        test_accuracy = 0.0


        wandb.init(
            # set the wandb project where this run will be logged
            project="neuro-hebbian-learning",
            
            # track hyperparameters and run metadata
            config={
                "learning_rate": optimizer.defaults["lr"],
                "architecture": f"{self.layer_list}",
                "dataset": "MNIST",
                "epochs": epochs,
                "batch_size": train_dataloader.batch_size,
            }
        )

        best_params = None

        with tqdm(total=epochs, desc='Training', unit='epoch') as pbar:
            for _ in range(epochs):
                # Training loop
                self.train()
                self.reset_weights('zero')

                epoch_train_loss = 0.0
                epoch_train_accuracy = 0.0
                with tqdm(total=len(train_dataloader), desc='Train', unit='batch', leave=False) as train_pbar:
                    for inputs, targets in train_dataloader:
                        out_base = self.learn(inputs.to(self.device))
                        out_improved = self.forward(inputs.to(self.device))

                        loss_base = loss_fn(out_base, targets.to(self.device))
                        loss_improved = loss_fn(out_improved, targets.to(self.device))

                        loss = loss_improved #- loss_base

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        epoch_train_loss += loss_improved.item()
                        train_pbar.update(1)
                        train_pbar.set_postfix({'Base Loss': loss_base.item(), 'Improved Loss': loss_improved.item()})

                        # Calculate accuracy
                        predicted_labels = torch.argmax(out_improved, dim=1)
                        accuracy = sklearn.metrics.accuracy_score(targets.cpu(), predicted_labels.cpu())
                        epoch_train_accuracy += accuracy
                train_loss.append(epoch_train_loss / len(train_dataloader))
                train_accuracy.append(epoch_train_accuracy / len(train_dataloader))

                wandb.log({"train_loss": train_loss[-1], "train_accuracy": train_accuracy[-1]})

                if scheduler is not None:
                    scheduler.step()

                # Validation loop
                self.eval()
                epoch_val_loss = 0.0
                epoch_val_accuracy = 0.0
                with tqdm(total=len(val_dataloader), desc='Validation', unit='batch', leave=False) as val_pbar:
                    with torch.no_grad():
                        for inputs, targets in val_dataloader:
                            output = self.forward(inputs.to(self.device))

                            loss = loss_fn(output, targets.to(self.device))
                            epoch_val_loss += loss.item()
                            val_pbar.update(1)
                            val_pbar.set_postfix({'Loss': loss.item()})

                            # Calculate accuracy
                            predicted_labels = torch.argmax(output, dim=1)
                            accuracy = sklearn.metrics.accuracy_score(targets.cpu(), predicted_labels.cpu())
                            epoch_val_accuracy += accuracy
                val_loss.append(epoch_val_loss / len(val_dataloader))
                acc = epoch_val_accuracy / len(val_dataloader)
                if len(val_accuracy) == 0 or acc > max(val_accuracy):
                    best_params = {key: value.clone() for key, value in self.state_dict().items()}
                val_accuracy.append(acc)
                pbar.update(1)
                pbar.set_postfix({'Train Loss': train_loss[-1], 'Val Loss': val_loss[-1], 'Train Accuracy': train_accuracy[-1], 'Val Accuracy': val_accuracy[-1]})

                wandb.log({"val_loss": val_loss[-1], "val_accuracy": val_accuracy[-1]})

            # Load best params
            self.load_state_dict(best_params)

            self.eval()
            test_loss = 0.0
            test_accuracy = 0.0
            with tqdm(total=len(test_dataloader), desc='Test', unit='batch') as test_pbar:
                with torch.no_grad():
                    for inputs, targets in test_dataloader:
                        output = self.forward(inputs.to(self.device))

                        loss = loss_fn(output, targets.to(self.device))
                        test_loss = loss.item() / len(inputs) # keep only last
                        test_pbar.update(1)
                        test_pbar.set_postfix({'Loss': loss.item()})

                        # Calculate accuracy
                        predicted_labels = torch.argmax(output, dim=1)
                        accuracy = sklearn.metrics.accuracy_score(targets.cpu(), predicted_labels.cpu())
                        test_accuracy += accuracy
            test_loss = (test_loss / len(test_dataloader))
            test_accuracy = (test_accuracy / len(test_dataloader))

        wandb.finish()

        return train_loss, val_loss, test_loss, train_accuracy, val_accuracy, test_accuracy
    
    def train_weights(self, loss_fn, train_dataloader, test_dataloader, epochs=2):
        train_loss = []
        test_loss = []
        test_accuracy = []
        self.reset_weights('zero')

        with tqdm(total=epochs, desc='Training weights', unit='epoch') as pbar:
            with torch.no_grad():
                self.eval()
                for _ in range(epochs):
                # Training loop
                    epoch_train_loss = 0.0
                    with tqdm(total=len(train_dataloader), desc='Train', unit='batch', leave=False) as train_pbar:
                        for inputs, targets in train_dataloader:
                            self.learn(inputs.to(self.device))
                            output = self.forward(inputs.to(self.device))

                            loss = loss_fn(output, targets.to(self.device))
                            epoch_train_loss += loss.item()
                            train_pbar.update(1)
                            train_pbar.set_postfix({'Loss': loss.item()})
                    train_loss.append(epoch_train_loss / len(train_dataloader))
                    pbar.update(1)
                    pbar.set_postfix({'Train Loss': train_loss[-1]})


            # Test 
            epoch_test_loss = 0.0
            epoch_test_accuracy = 0.0
            with tqdm(total=len(test_dataloader), desc='Test', unit='batch', leave=False) as test_pbar:
                with torch.no_grad():
                    for inputs, targets in test_dataloader:
                        # self.learn(inputs.to(self.device))
                        output = self.forward(inputs.to(self.device))

                        loss = loss_fn(output, targets.to(self.device))
                        epoch_test_loss += loss.item()
                        test_pbar.update(1)
                        test_pbar.set_postfix({'Loss': loss.item()})

                        # Calculate accuracy
                        predicted_labels = torch.argmax(output, dim=1)
                        accuracy = sklearn.metrics.accuracy_score(targets.cpu(), predicted_labels.cpu())
                        epoch_test_accuracy += accuracy
                    test_loss.append(epoch_test_loss / len(test_dataloader))
                    test_accuracy.append(epoch_test_accuracy / len(test_dataloader))


        return train_loss, test_loss, test_accuracy
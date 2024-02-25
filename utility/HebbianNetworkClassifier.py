import utility.HebbianNetwork as hn
from tqdm import tqdm
import sklearn.metrics
import wandb
import torch


class HebbianNetworkClassifier(hn.HebbianNetwork):

    def __init__(
            self, 
            layers: list, 
            init,
            device='cpu',
            dropout=0.0,
    ):
        super(HebbianNetworkClassifier, self).__init__(layers, init, device, dropout)


    def train_loop(self, optimizer, loss_fn, train_dataloader, val_dataloader, test_dataloader, 
                   epochs=10, scheduler=None, backprop=True, reset_every=None, log=False):
        train_loss, val_loss = [], []
        train_accuracy, val_accuracy = [], []
        test_loss, test_accuracy = 0.0, 0.0

        best_params = None
        self.reset_weights('uni')

        with tqdm(total=epochs, desc='Training', unit='epoch') as pbar:
            for e in range(epochs):
                if backprop:
                    self.train()
                    if reset_every is not None and e % reset_every == 0:
                        self.reset_weights('uni')
                    # torch.manual_seed(42)
                else:
                    self.eval()

                epoch_train_loss = 0.0
                epoch_train_accuracy = 0.0
                with tqdm(total=len(train_dataloader), desc='Train', unit='batch', leave=False) as train_pbar:
                    for i, (inputs, targets) in enumerate(train_dataloader):

                        if backprop:
                            self.reset_weights('mantain')
                            output = self.learn(inputs.to(self.device))
                            out_imp = self.forward(inputs.to(self.device))

                            loss = loss_fn(output, targets.to(self.device))
                            loss_imp = loss_fn(out_imp, targets.to(self.device))

                            
                            loss_imp.backward()
                            optimizer.step()
                            optimizer.zero_grad()
                        else:
                            with torch.no_grad():
                                output = self.learn(inputs.to(self.device))
                                # output = self.forward(inputs.to(self.device))
                                loss = loss_fn(output, targets.to(self.device))

                        epoch_train_loss += loss.item()
                        train_pbar.update(1)
                        train_pbar.set_postfix({'Loss': loss.item()})

                        # Calculate accuracy
                        predicted_labels = torch.argmax(output, dim=1)
                        accuracy = sklearn.metrics.accuracy_score(targets.cpu(), predicted_labels.cpu())
                        epoch_train_accuracy += accuracy

                train_loss.append(epoch_train_loss / len(train_dataloader))
                train_accuracy.append(epoch_train_accuracy / len(train_dataloader))

                if backprop:
                    # Validation loop
                    if scheduler is not None:
                        scheduler.step()
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
                    pbar.set_postfix({'Train Loss': train_loss[-1], 'Val Loss': val_loss[-1], 'Train Accuracy': train_accuracy[-1], 'Val Accuracy': val_accuracy[-1]})
                    if log: 
                        wandb.log({"val_loss": val_loss[-1], "val_accuracy": val_accuracy[-1], "train_loss": train_loss[-1], "train_accuracy": train_accuracy[-1]})

                else:
                    pbar.set_postfix({'Hebb Train Loss': train_loss[-1], 'Hebb Train Accuracy': train_accuracy[-1]})
                    if log: 
                        wandb.log({ "hebb_train_loss": train_loss[-1], "hebb_train_accuracy": train_accuracy[-1]})

                pbar.update(1)

            if backprop:
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

            if log: 
                wandb.log({"test_loss": test_loss, "test_accuracy": test_accuracy})

        return train_loss, val_loss, test_loss, train_accuracy, val_accuracy, test_accuracy
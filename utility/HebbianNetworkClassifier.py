import utility.HebbianNetwork as hn
from tqdm import tqdm
import sklearn.metrics
import wandb
import torch


class HebbianNetworkClassifier(hn.HebbianNetwork):

    def __init__(
            self, 
            layers: list, 
            init='linear',
            device='cpu',
            dropout=0.0,
            bias=False,
            activation=torch.tanh,
            num_classes=10,
    ):
        super(HebbianNetworkClassifier, self).__init__(layers, init, device, dropout, bias, activation)
        self.init = init
        self.num_classes = num_classes


    def train_loop(self, optimizer, loss_fn, train_dataloader, val_dataloader, test_dataloader, 
                   epochs=10, scheduler=None, log=False, reset_every=1, early_stop=None, backprop_every=1):
        train_loss, val_loss = [], []
        train_accuracy, val_accuracy = [], []
        test_loss, test_accuracy = 0.0, 0.0

        best_params = None
        self.reset_weights(self.init)

        with tqdm(total=epochs, desc='Training', unit='epoch') as pbar:
            for e in range(epochs):
                self.train()
                if reset_every > 0 and e % reset_every == 0:
                    self.reset_weights(self.init)
                else: 
                    self.reset_weights('mantain')

                epoch_train_loss = 0.0
                epoch_train_accuracy = 0.0
                total = len(train_dataloader) if early_stop is None else early_stop
                with tqdm(total=total, desc='Train', unit='batch', leave=False) as train_pbar:
                    for i, (inputs, targets) in enumerate(train_dataloader):

                        _ = self.learn(inputs.to(self.device))

                        # output = self.learn(inputs.to(self.device))
                        out_imp = self.forward(inputs.to(self.device))

                        # _ = loss_fn(output, targets.to(self.device))
                        loss_imp = loss_fn(out_imp, targets.to(self.device))

                        if i % backprop_every == 0:
                            loss_imp.backward()
                            optimizer.step()
                            optimizer.zero_grad()
                            self.reset_weights('mantain')


                        epoch_train_loss += loss_imp.item()
                        train_pbar.update(1)
                        train_pbar.set_postfix({'Loss': loss_imp.item()})

                        # Calculate accuracy
                        predicted_labels = torch.argmax(out_imp, dim=1)
                        accuracy = sklearn.metrics.accuracy_score(targets.cpu(), predicted_labels.cpu())
                        epoch_train_accuracy += accuracy

                        if early_stop is not None and i >= early_stop:
                            break

                train_loss.append(epoch_train_loss / total)
                train_accuracy.append(epoch_train_accuracy / total)

                # Validation loop
                if scheduler is not None:
                    scheduler.step()

                self.eval()
                epoch_val_loss = 0.0
                epoch_val_accuracy = 0.0
                with tqdm(total=len(val_dataloader), desc='Validation', unit='batch', leave=False) as val_pbar:
                    with torch.no_grad():
                        for inputs, targets in val_dataloader:
                            # _ = self.learn(inputs.to(self.device))
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
                
                pbar.set_postfix({
                    'Train Loss': train_loss[-1],
                    'Val Loss': val_loss[-1],
                    'Train Accuracy': train_accuracy[-1],
                    'Val Accuracy': val_accuracy[-1]
                })
                
                if log: wandb.log({
                    "val_loss": val_loss[-1],
                    "val_accuracy": val_accuracy[-1],
                    "train_loss": train_loss[-1],
                    "train_accuracy": train_accuracy[-1]
                })


                pbar.update(1)

        self.load_state_dict(best_params)

        test_loss, test_accuracy, confusion_matrix = self.test(test_dataloader, loss_fn, log)

        return train_loss, val_loss, test_loss, train_accuracy, val_accuracy, test_accuracy, confusion_matrix
    

    def hebbian_train_loop(self, loss_fn, train_dataloader, val_dataloader, test_dataloader, log=False, max_iter=100):
        train_loss, val_loss = [], []
        train_accuracy, val_accuracy = [], []
        test_loss, test_accuracy = 0.0, 0.0

        best_weights = None
        self.reset_weights(self.init)
        iter = 0

        self.eval()

        with torch.no_grad():
            with tqdm(total=min(len(train_dataloader), max_iter), desc='Train', unit='batch', leave=False) as train_pbar:
                for inputs, targets in train_dataloader:
                    output = self.learn(inputs.to(self.device))
                    loss = loss_fn(output, targets.to(self.device))

                    train_loss.append(loss.item())
                    # Calculate accuracy
                    predicted_labels = torch.argmax(output, dim=1)
                    accuracy = sklearn.metrics.accuracy_score(targets.cpu(), predicted_labels.cpu())
                    train_accuracy.append(accuracy)

                    wandb_log = {
                        "train_loss_hebb": train_loss[-1], 
                        "train_accuracy_hebb": train_accuracy[-1]
                    }
                    postfix = {
                        'Hebb Train Loss': train_loss[-1],
                        'Hebb Train Accuracy': train_accuracy[-1],
                    }

                    if val_dataloader is not None:
                        # Validation loop, at every sample
                        epoch_val_loss = 0.0
                        epoch_val_accuracy = 0.0
                        with tqdm(total=len(val_dataloader), desc='Validation', unit='batch', leave=False) as val_pbar:
                            with torch.no_grad():
                                for inputs, targets in val_dataloader:
                                    # _ = self.learn(inputs.to(self.device))
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
                            best_weights = self.get_weights()
                            
                        val_accuracy.append(acc)
                        postfix['Hebb Val Loss'] = val_loss[-1]
                        postfix['Hebb Val Accuracy'] = val_accuracy[-1]
                        wandb_log["val_loss_hebb"] = val_loss[-1]
                        wandb_log["val_accuracy_hebb"] = val_accuracy[-1]

                    train_pbar.set_postfix(postfix)
                    train_pbar.update(1)

                    if log: wandb.log(wandb_log)

                    iter += 1
                    if iter > max_iter:
                        break

        self.set_weights(best_weights)

        test_loss, test_accuracy, confusion_matrix = self.test(test_dataloader, loss_fn, log)

        return train_loss, val_loss, test_loss, train_accuracy, val_accuracy, test_accuracy, confusion_matrix


    
    def test(self, test_dataloader, loss_fn, log=False):
        self.eval()
        test_loss = 0.0
        test_accuracy = 0.0
        confusion_matrix = torch.zeros((self.num_classes, self.num_classes))
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

                    # Update confusion matrix
                    for t, p in zip(targets.view(-1), predicted_labels.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1

        test_loss = (test_loss / len(test_dataloader))
        test_accuracy = (test_accuracy / len(test_dataloader))

        if log: 
            wandb.log({"test_loss": test_loss, "test_accuracy": test_accuracy})

        return test_loss, test_accuracy, confusion_matrix


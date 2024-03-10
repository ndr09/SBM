import utility.HebbianNetwork as hn
from tqdm import tqdm
import sklearn.metrics
import wandb
import torch


class HebbianNetworkClassifier(hn.HebbianNetwork):
    """
    Wrapper for HebbianNetwork to use it as a classifier. 
    It adds a train_loop method to train the hebbian parameters and a hebbian_train_loop method to train only with hebbian learning.
    """

    def __init__(
            self, 
            layers: list, 
            init='linear',
            device='cpu',
            dropout=0.0,
            bias=False,
            activation=torch.tanh,
            num_classes=10,
            neuron_centric=True,
            use_d=False,
            rank=1,
            train_weights=False,
    ):
        """
        Initializes the HebbianNetworkClassifier.
        
        :param layers: List of integers representing the number of neurons in each layer.
        :param init: Initialization method for the weights.
        :param device: Device to use for the network.
        :param dropout: Dropout rate to use.
        :param bias: Whether to use a bias.
        :param activation: Activation function to use.
        :param num_classes: Number of classes for the classification task.
        :param rank: Rank of the C parameter of Hebbian learning rule. Default is 1.
        """
        super(HebbianNetworkClassifier, self).__init__(
            layers=layers,
            device=device,
            dropout=dropout,
            bias=bias,
            activation=activation,
            neuron_centric=neuron_centric,
            init=init,
            use_d=use_d,
            train_weights=train_weights,
        )
        self.init = init
        self.num_classes = num_classes


    def train_loop(
            self, 
            optimizer,
            loss_fn, 
            train_dataloader, 
            val_dataloader, 
            test_dataloader, 
            epochs=10, 
            scheduler=None, 
            log=False, 
            reset_every=1, 
            early_stop=None, 
            backprop_every=1
        ):
        """
        Trains the network hebbian parameters. The best parameterson the validation dataset are kept.
        
        :param optimizer: Optimizer to use for training.
        :param loss_fn: Loss function to use for training.
        :param train_dataloader: Dataloader for the training set.
        :param val_dataloader: Dataloader for the validation set.
        :param test_dataloader: Dataloader for the test set.
        :param epochs: Number of epochs to train.
        :param scheduler: Learning rate scheduler to use. Default is None.
        :param log: Whether to log the training process to wandb. Default is False.
        :param reset_every: Number of epochs after which to reset the weights. Default is 1.
        :param early_stop: Number of batches to train on before stopping in every epoch. Default is None, so no early stop.
        :param backprop_every: Number of batches after which to backpropagate. Default is 1.
        :return: Train loss, validation loss, test loss, train accuracy, validation accuracy, test accuracy, confusion matrix.
        """

        train_loss, val_loss = [], []
        train_accuracy, val_accuracy = [], []
        test_loss, test_accuracy = 0.0, 0.0

        # best set of parameters
        best_params = None

        with tqdm(total=epochs, desc='Training', unit='epoch') as pbar:
            for e in range(epochs):
                self.train()
                # if i have to reset the weights, do it, otherwise only clean the gradient graph
                if reset_every > 0 and e % reset_every == 0:
                    self.reset_weights(self.init)
                else: 
                    self.reset_weights('mantain')

                # train loop
                epoch_train_loss = 0.0
                epoch_train_accuracy = 0.0
                total = len(train_dataloader) if early_stop is None else early_stop
                with tqdm(total=total, desc='Train', unit='batch', leave=False) as train_pbar:
                    for i, (inputs, targets) in enumerate(train_dataloader):
                        # i have to call learn and forward after in order to have the gradients on the updated weights
                        # get one hot for the targets
                        t = torch.nn.functional.one_hot(targets, self.num_classes).float() * 2. - 1.
                        output = self.learn(inputs.to(self.device), t.to(self.device))
                        # if output.grad_fn is None:
                        #     train_pbar.update(1)
                        #     continue
                        output = self.forward(inputs.to(self.device))

                        # _ = loss_fn(output, targets.to(self.device))
                        loss = loss_fn(output, targets.to(self.device))

                        if i % backprop_every == 0:
                            loss.backward(retain_graph=True)
                            optimizer.step()
                            optimizer.zero_grad()
                            self.reset_weights('mantain')

                        epoch_train_loss += loss.item()
                        train_pbar.update(1)
                        train_pbar.set_postfix({'Loss': loss.item()})

                        # Calculate accuracy
                        predicted_labels = torch.argmax(output, dim=1)
                        accuracy = sklearn.metrics.accuracy_score(targets.cpu(), predicted_labels.cpu())
                        epoch_train_accuracy += accuracy

                        # if early stop is set, stop after the specified number of batches
                        if early_stop is not None and i >= early_stop:
                            break

                train_loss.append(epoch_train_loss / total)
                train_accuracy.append(epoch_train_accuracy / total)

                if scheduler is not None:
                    scheduler.step()

                # Validation loop
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
                pbar.update(1)
                
                if log: wandb.log({
                    "val_loss": val_loss[-1],
                    "val_accuracy": val_accuracy[-1],
                    "train_loss": train_loss[-1],
                    "train_accuracy": train_accuracy[-1]
                })

        # load the best parameters found for hebbian learning
        self.load_state_dict(best_params)

        # test the model
        test_loss, test_accuracy, confusion_matrix = self.test(test_dataloader, loss_fn, log)

        return train_loss, val_loss, test_loss, train_accuracy, val_accuracy, test_accuracy, confusion_matrix
    

    def hebbian_train_loop(self, loss_fn, train_dataloader, val_dataloader, test_dataloader, log=False, epochs=1, max_iter=100, reset=True):
        """
        Trains the network only with hebbian learning. The best parameters on the validation dataset are kept.
        
        :param loss_fn: Loss function to use for training.
        :param train_dataloader: Dataloader for the training set.
        :param val_dataloader: Dataloader for the validation set.
        :param test_dataloader: Dataloader for the test set.
        :param log: Whether to log the training process to wandb. Default is False.
        :param epochs: Number of epochs to train. Default is 1.
        :param max_iter: Maximum number of iterations to train. Default is 100.
        :return: Train loss, validation loss, test loss, train accuracy, validation accuracy, test accuracy, confusion matrix.
        """
        train_loss, val_loss = [], []
        train_accuracy, val_accuracy = [], []
        test_loss, test_accuracy = 0.0, 0.0

        best_weights = None
        if reset:
            self.reset_weights(self.init)

        iter = 0
        self.eval()
        with torch.no_grad():
            total = min(len(train_dataloader) * epochs, max_iter)
            with tqdm(total=total, desc='Train', unit='batch', leave=False) as train_pbar:
                for e in range(epochs):
                    for inputs, targets in train_dataloader:
                        t = torch.nn.functional.one_hot(targets, self.num_classes).float() * 2. - 1.
                        output = self.learn(inputs.to(self.device), t.to(self.device))
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

                        # if there is no validation set, skip the validation loop
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

        if val_dataloader is not None:
            self.set_weights(best_weights)

        # test the model
        test_loss, test_accuracy, confusion_matrix = self.test(test_dataloader, loss_fn, log)

        return train_loss, val_loss, test_loss, train_accuracy, val_accuracy, test_accuracy, confusion_matrix


    
    def test(self, test_dataloader, loss_fn, log=False, online=False):
        """
        Tests the network on the test set.
        
        :param test_dataloader: Dataloader for the test set.
        :param loss_fn: Loss function to use for testing.
        :param log: Whether to log the test results to wandb. Default is False.
        :return: Test loss, test accuracy, confusion matrix.
        """
        self.eval()
        test_loss = 0.0
        test_accuracy = 0.0
        confusion_matrix = torch.zeros((self.num_classes, self.num_classes))
        with tqdm(total=len(test_dataloader), desc='Test', unit='batch') as test_pbar:
            with torch.no_grad():
                for inputs, targets in test_dataloader:
                    if online:
                        output = torch.zeros((len(inputs), self.num_classes)).to(self.device)
                        for i in range(len(inputs)):
                            output[i, :] = self.learn(inputs[i].unsqueeze(0).to(self.device))
                    else:
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
    

    def get_params(self):
        """
        Returns the parameters of the network.
        
        :return: Parameters of the network.
        """
        return self.state_dict()


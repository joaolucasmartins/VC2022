#!/bin/python

import cv2 as cv
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import json

# Dataset functions

def importImage(name):
    image = cv.imread('../dataset/images/' + name + ".png")
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)


def displayImage(image, title=""):
    if (len(image.shape) == 3):
        cmap = None
    else:
        cmap = "gray"
        
    plt.rcParams['figure.figsize'] = [10, 10]
    plt.figure()
    plt.title(title)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def plotTrainingHistory(train_history, val_history):
    plt.subplot(2, 1, 1)
    plt.title('Cross Entropy Loss')
    plt.plot(train_history['loss'], label='train')
    plt.plot(val_history['loss'], label='val')
    plt.legend(loc='best')

    plt.subplot(2, 1, 2)
    plt.title('Classification Accuracy')
    plt.plot(train_history['accuracy'], label='train')
    plt.plot(val_history['accuracy'], label='val')

    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()

# Deep learning functions

class ModelTrainer:
    def __init__(self, *args):
        if len(args) == 5:
            model, model_name, loss, optimizer, device = args
        elif len(args) == 2:
            ks = ["model", "name", "num_epochs", "loss", "optimizer"]
            model, model_name, num_epochs, loss, optimizer = [args[0][k] for k in ks]
            device = args[1]

        self.model = model
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.loss = loss
        self.optimizer = optimizer
        self.device = device

    def predict_data(self, data):
        print("Predicting data")
        preds = []
        actuals = []
        with torch.set_grad_enabled(False):
            for _, (X, y) in enumerate(tqdm(data)):
                pred = self.model(X)
                probs = F.softmax(pred, dim=1)
                final_pred = torch.argmax(probs, dim=1)
                preds.extend(final_pred)
                actuals.extend(y)
        return torch.stack(preds, dim=0), torch.stack(actuals, dim=0)

    def _epoch_iter(self, dataloader, is_train):
        if is_train:
            assert self.optimizer is not None, "When training, please provide an optimizer."

        num_batches = len(dataloader)

        if is_train:
            self.model.train()  # put model in train mode
        else:
            self.model.eval()

        total_loss = 0.0
        preds = []
        labels = []

        with torch.set_grad_enabled(is_train):
            for batch, (X, y) in enumerate(tqdm(dataloader)):
                X, y = X.to(self.device), y.to(self.device)

                # Compute prediction error
                pred = self.model(X)
                loss = self.loss(pred, y)

                if is_train:
                    # Backpropagation
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # Save training metrics
                # IMPORTANT: call .item() to obtain the value of the loss WITHOUT the computational graph attached
                total_loss += loss.item()

                probs = F.softmax(pred, dim=1)
                final_pred = torch.argmax(probs, dim=1)
                preds.extend(final_pred.cpu().numpy())
                labels.extend(y.cpu().numpy())

        return total_loss / num_batches, accuracy_score(labels, preds)
    
    def _save_model(self, t, file_name):
        save_dict = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': t}
        torch.save(save_dict, self.model_name + f'_{ file_name }.pth')

    def train(self, train_dataloader, validation_dataloader, force_load_model=True):
        from os.path import exists
        file_exists = exists(self.model_name + '_latest_model.pth')
        if file_exists:
            if not force_load_model:
                print("File already exists do you wish to overwrite (Y/N)?")
                ans = input()
            if force_load_model or not (ans == 'Y' or ans == 'y'):
                # Load model and display previous results
                dic = torch.load(self.model_name + '_best_model.pth')
                self.model.load_state_dict(dic['model'])
                print(f"Loaded { self.model_name } obtained in epoch { dic['epoch'] }")
                self.model.eval()
                with open(self.model_name + '_train_history.json', 'r') as f:
                    train_history = json.load(f)
                with open(self.model_name + '_val_history.json', 'r') as f:
                    val_history = json.load(f)
                return train_history, val_history


        train_history = {'loss': [], 'accuracy': []}
        val_history = {'loss': [], 'accuracy': []}
        best_val_loss = np.inf
        print("Start training...")

        for t in range(self.num_epochs):
            print(f"\nEpoch {t+1}")

            # Train
            train_loss, train_acc = self._epoch_iter(train_dataloader, True)

            print(f"Train loss: {train_loss:.3f} \t Train acc: {train_acc:.3f}")

            # Test
            val_loss, val_acc = self._epoch_iter(validation_dataloader, False)
            print(f"Val loss: {val_loss:.3f} \t Val acc: {val_acc:.3f}")

            # Save model when validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_model(t, 'best_model')

            # Save latest model
            self._save_model(t, 'best_model')

            # save training history for plotting purposes
            train_history["loss"].append(train_loss)
            train_history["accuracy"].append(train_acc)

            val_history["loss"].append(val_loss)
            val_history["accuracy"].append(val_acc)

        print("Finished")
        with open(self.model_name + '_train_history.json', 'w') as f:
            f.write(json.dumps(train_history))

        with open(self.model_name + '_val_history.json', 'w') as f:
            f.write(json.dumps(val_history))

        return train_history, val_history

    def freeze_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_layers(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def unfreeze_fc(self):
        self.model.fc.weight.requires_grad = True

    def visualize_model(self, val_dataloader, classes, num_images=6): # Adapted from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
        was_training = self.model.training
        self.model.eval()
        images_so_far = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_dataloader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images//2, 2, images_so_far)
                    ax.axis('off')
                    class_names = list(classes.keys())
                    ax.set_title(f'predicted: {class_names[preds[j]]}')
                    cv.imshow(inputs.cpu().data[j])

                    if images_so_far == num_images:
                        self.model.train(mode=was_training)
                        return
            self.model.train(mode=was_training)
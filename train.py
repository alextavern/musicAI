from time import time
import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

def train(model, loss_fun, optimiser, device, epochs, train_data_loader, val_data_loader=None):
    start_training_time = time()

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):

        # TRAINING
        running_losses = []
        for inputs, _, targets in tqdm(train_data_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            # calculate loss
            model.train()
            predictions = model(inputs)
            loss = loss_fun(predictions, targets)
            # backpropagate loss and update weights (gradient descent)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            # gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            running_losses.append(loss.item())

        # TRAINING ACCURACY
        train_accuracy = calc_accuracy(model, train_data_loader, device)
        train_accuracies.append(train_accuracy)

        train_losses.append(np.mean(running_losses))

        # VALIDATION
        if val_data_loader is not None:
            val_loss = validation(model, loss_fun, val_data_loader)
            val_losses.append(val_loss)

            # VALIDATION ACCURACY
            val_accuracy = calc_accuracy(model, val_data_loader, device)
            val_accuracies.append(val_accuracy)

        # PRINT INFO AT END OF EACH EPOCH
        print("Epoch {}".format(epoch + 1))
        print("Training loss: {} - Validation loss: {} ".format(np.average(running_losses), np.average(val_losses)))
        print("Training acc:  {} - Validation acc:  {} ".format(train_accuracy, val_accuracy))

        # print(f"Loss: {loss.item()}")
        # print("---------------------------")

    stop_training_time = time()
    print("\nTraining is done. Training Time (in minutes) =", (stop_training_time - start_training_time) / 60)

    return train_losses, val_losses, train_accuracies, val_accuracies


def validation(model, loss_fn, val_loader):
    # Figures device from where the model parameters (hence, the model) are
    device = next(model.parameters()).device.type

    # no gradients in validation!
    with torch.no_grad():
        val_batch_losses = []
        total = 0
        for x_val, _, y_val in val_loader:
            x_val = x_val.to(device)
            y_val = y_val.to(device)

            # sets model to EVAL mode
            model.eval()

            # make predictions
            pred = model(x_val)
            val_loss = loss_fn(pred, y_val)
            val_batch_losses.append(val_loss.item())

        val_losses = np.mean(val_batch_losses)

    return val_losses


def predict(model, input, target, labels):
    model.eval()

    with torch.no_grad():
        predictions = model(input)  # a tensor object (1, 10)
        predicted_index = predictions[0].argmax(0).cpu().numpy()
        predicted = labels[predicted_index]
        expected = labels[target]

    return predicted, expected


def calc_accuracy(model, data_loader, device):
    with torch.no_grad():
        model.eval()
        total_correct = 0
        total_instances = 0
        for inputs, _, targets in (data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = torch.argmax(model(inputs), dim=1)
            correct_predictions = sum(predictions == targets).item()
            total_correct += correct_predictions
            total_instances += len(inputs)
    return (round(total_correct / total_instances, 3))

def confusion_matrix_and_report(model, device, dataset, labels):
  y_pred = []
  y_true = []

  # inputs, _, targets = next(iter(dataloader_val))

  for input, _, target in dataset:
    input = input.to(device)
    # target = target.to(device)
    predicted, expected = predict(model, input, target, labels)
    y_pred.append(predicted)
    y_true.append(expected)

  cf_matrix = confusion_matrix(y_true, y_pred)
  df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in labels],
                      columns = [i for i in labels])
  plt.figure(figsize = (12,7))
  sn.heatmap(df_cm,
            annot=True,
            square=True,
            xticklabels=labels,
            yticklabels=labels,
            cmap=plt.cm.Blues,
            cbar=False)

  print(classification_report(y_true, y_pred, target_names=labels))
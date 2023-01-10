import torch
from torch import nn
import torchaudio
from UrbanSound8K_dataprep import UrbanSoundPrep
from UrbanSound8K_CNN import CNNNetwork as CNN
from torch.utils.data import DataLoader

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.0001


def train_one_epoch(model, data_loader, loss_fun, optimiser, device):

  for inputs, targets in data_loader:
    inputs, targets = inputs.to(device), targets.to(device)

    # calculate loss
    predictions = model(inputs)
    loss = loss_fun(predictions, targets)

    # backpropagate loss and update weights (gradient descent)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

  print(f"Loss: {loss.item()}")

def train(model, data_loader, loss_fun, optimiser, device, epochs):
  for epoch in range(epochs):
    print(f"Epoch {epoch+1}")
    train_one_epoch(model, data_loader, loss_fun, optimiser, device)
    print("---------------------------")
  print("Training is done")


if __name__ == "__main__":
  if torch.cuda.is_available():
    device = "cuda"
  else:
    device = "cpu"

  U = UrbanSoundPrep("data")
  train_dataloader = DataLoader(U, batch_size=BATCH_SIZE)

  #
  cnn = CNN().to(device)

  loss_fun = nn.CrossEntropyLoss()
  optimiser = torch.optim.Adam(cnn.parameters(),
                               lr=LEARNING_RATE)

  train(cnn, train_dataloader, loss_fun, optimiser, device, EPOCHS)
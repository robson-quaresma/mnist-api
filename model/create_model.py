import torch
import torch.nn as nn
import torch.nn.functional as F

def create_the_model():
  class MnistNet(nn.Module):
    def __init__(self):
      super().__init__()

      # input layer
      self.input = nn.Linear(784, 64)

      # hidden layer
      self.h1 = nn.Linear(64, 32)
      self.h2 = nn.Linear(32, 32)

      # output layer
      self.output = nn.Linear(32, 10)

    def forward(self, x):
      x = F.leaky_relu(self.input(x))
      x = F.leaky_relu(self.h1(x))
      x = F.leaky_relu(self.h2(x))

      return torch.log_softmax(self.output(x), axis=1)

  net = MnistNet()

  # convert model weights to float32
  for param in net.parameters():
      param.data = param.data.float()

  return net, nn.NLLLoss(), torch.optim.Adam(net.parameters(), lr=0.01)

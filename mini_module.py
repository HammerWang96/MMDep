import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class BWFeedForward1(torch.nn.Module):
    def __init__(self, input_size, output_size):  #
        super(BWFeedForward1, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return x


class BWFeedForward2(torch.nn.Module):
    def __init__(self, input_size, output_size):  #
        super(BWFeedForward2, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # x = self.fc1(x)
        x = self.relu(self.fc1(x))
        return x


class Gate1(torch.nn.Module):
    def __init__(self, input_size, output_size):  #
        super(Gate1, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return x


class Gate2(torch.nn.Module):
    def __init__(self, input_size, output_size):  #
        super(Gate2, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return x

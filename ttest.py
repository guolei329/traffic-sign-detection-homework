import torch
import torch.nn as nn
from torch import autograd

m = nn.Linear(20, 30)
input = autograd.Variable(torch.randn(128, 20))
output = m(input)
print(output.size())
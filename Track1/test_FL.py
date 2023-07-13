import torch
import torch.nn as nn
import torch.nn.functional as F
from focal_loss.focal_loss import FocalLoss

batch_size = 10
max_length = 20
n_class = 5

weights = torch.FloatTensor([1, 1, 1, 1, 1])
criterion = FocalLoss(gamma=0.7, weights=weights)

m = torch.nn.Softmax(dim=-1)
logits = torch.randn(batch_size, max_length, n_class)
target = torch.randint(0, n_class, size=(batch_size, max_length))

loss = criterion(m(logits), target)
print(loss)
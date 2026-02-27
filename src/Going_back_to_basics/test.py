import torch

x = torch.tensor([25.0])
print(x.shape)
x = x.unsqueeze(0)
print(x)
#x = x.squeeze()
print(x.shape)
import torch

class Softmax(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, dim: int):
        x_max = input.max(dim=dim, keepdim=True)[0]
        x_shifted = input - x_max
        x_exp = torch.exp(x_shifted)
        x_sum = x_exp.sum(dim=dim, keepdim=True)
        x_softmax = x_exp / x_sum
        return x_softmax

import torch

class CrossEntropy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        batch_size = target.shape[0]
        loss = (torch.logsumexp(x, dim=-1) - x[range(batch_size), target]).mean()
        return loss

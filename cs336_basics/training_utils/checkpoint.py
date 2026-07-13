import torch

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out):
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    obj = {"model_state": model_state, "optimizer_state": optimizer_state, "iteration": iteration}
    torch.save(obj, out)

def load_checkpoint(src, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    obj = torch.load(src)
    model_state = {}
    optimizer_state = {}
    iteration = 0
    if obj["model_state"] is not None:
        model_state = obj["model_state"]
    if obj["optimizer_state"] is not None:
        optimizer_state = obj["optimizer_state"]
    if obj["iteration"] is not None:
        iteration = obj["iteration"]
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    return iteration
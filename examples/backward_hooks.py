"""
Backward hook usage: catch gradient NaNs directly.
OrderedDict gives readable layer names in all log output.
"""
import torch
import torch.nn as nn
from collections import OrderedDict
from nan_detector import NaNDetector

torch.manual_seed(0)
model = nn.Sequential(OrderedDict([
    ("fc1",  nn.Linear(16, 32)),
    ("relu", nn.ReLU()),
    ("fc2",  nn.Linear(32, 1)),
]))
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e8)  # large LR → explosion

with NaNDetector(model, check_backward=True, grad_norm_warn=10.0) as det:
    for i in range(10):
        det.set_batch(i)
        x = torch.randn(8, 16, requires_grad=True)
        y = torch.randn(8, 1)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        warned = det.check_grad_norms()
        optimizer.step()
        if det.triggered or warned:
            print(f"Caught at batch {i}")
            break

print(f"Grad events: {len(det.grad_events)}")

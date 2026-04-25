"""
Basic usage: wrap any training loop with NaNDetector.
"""
import torch
import torch.nn as nn
from nan_detector import NaNDetector

torch.manual_seed(0)
model     = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 1))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loader    = [(torch.randn(16, 32), torch.randn(16, 1)) for _ in range(20)]

with NaNDetector(model, grad_norm_warn=50.0) as det:
    for batch_idx, (x, y) in enumerate(loader):
        det.set_batch(batch_idx)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        det.check_grad_norms()
        optimizer.step()
        print(f"batch {batch_idx:02d}  loss={loss.item():.4f}")
        if det.triggered:
            print(f"\nStopped — NaN at batch {det.event.batch_idx}, layer '{det.event.layer_name}'")
            break

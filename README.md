# pytorch-nan-detector
PyTorch NaNs are silent killers. This hook catches them at the exact layer and batch — with ~7 ms overhead vs ~7–8 ms ms for set_detect_anomaly.


# pytorch-nan-detector

A lightweight forward-hook NaN/Inf detector for PyTorch — catches the exact layer and batch where NaNs first appear, with ~3 ms overhead.

> Companion code for the Towards Data Science article:  
> **[PyTorch NaNs Are Silent Killers — I Built a 3 ms Hook That Pinpoints Them to the Exact Layer and Batch](https://towardsdatascience.com/)**

---

```
Training loop → Forward hooks → NaNEvent → layer + batch + stats
                     ↑
              Gradient norm guard (catches explosion before NaN)
```

---

## Benchmark

| Method | Mean (ms) | Overhead |
|---|---|---|
| No detection | ~0.60 ms | baseline |
| **NaNDetector** | **~3–4 ms** | **~5–6×** |
| `set_detect_anomaly` | ~7–8 ms | ~12–13× |

*CPU · 4-layer MLP · batch size 64 · 30 forward passes.  
On GPU with large models, `set_detect_anomaly` reaches 50–100×.*

![Benchmark](assets/plot_benchmark.png)

---

## Performance

| Operation | Latency |
|---|---|
| Hook check per layer | ~0.02 ms |
| Full forward pass overhead | ~0.10 ms |
| `set_detect_anomaly` equivalent | ~7–8 ms |

---

## Install

No package. Single file — drop it into your project:

```bash
curl -O https://raw.githubusercontent.com/Emmimal/pytorch-nan-detector/main/nan_detector.py
```

**Requires:** `torch>=1.11` · `matplotlib>=3.5` · `Python>=3.10`

---

## Quick start

```python
from nan_detector import NaNDetector

with NaNDetector(model) as det:
    for batch_idx, (x, y) in enumerate(loader):
        det.set_batch(batch_idx)
        loss = criterion(model(x), y)
        loss.backward()
        det.check_grad_norms()
        optimizer.step()
        if det.triggered:
            print(det.event)
            break
```

---

## Demo output

```
NaN/Inf detected! [FORWARD PASS]
  Batch     : 12
  Layer     : layer4
  Type      : Linear
  Flags     : NaN in INPUT, NaN in OUTPUT
  Out shape : (8, 1)
  Out stats : min=n/a  max=n/a  mean=n/a (all non-finite)
```

Run all three demos and generate plots:

```bash
python nan_detector.py
```

---

## Plots

**Loss curve — NaN detected at batch 12**
![Loss curve](assets/plot_loss_curve.png)

**Gradient norm explosion — caught one step before NaN**
![Grad norms](assets/plot_grad_norms.png)

---

## When to use this

Worth it when you have:
- Training runs longer than a few minutes where `set_detect_anomaly` slowdown is unacceptable
- A need to know *which layer* originated the NaN, not just that one occurred
- Multi-worker `DataLoader` setups where anomaly detection is unusable at scale

Skip it when you have:
- Quick single-run debugging on a tiny model — `set_detect_anomaly` is fine
- NaNs originating inside a custom CUDA kernel (forward hooks won't see it)
- Hard latency requirements under 1 ms per forward pass

---

## Known limitations

- Forward hooks won't catch NaNs inside `torch.autograd.Function.backward()` — use `check_backward=True`
- Hook overhead scales with model depth — use `skip_types` to exclude non-parametric layers on very deep models
- Token estimation and GPU benchmarks not included — the 50–100× figure is from PyTorch docs, not measured here

---

## License

MIT — see [LICENSE](LICENSE)

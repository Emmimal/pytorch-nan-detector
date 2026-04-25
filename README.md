# pytorch-nan-detector
PyTorch NaNs are silent killers. This hook catches them at the exact layer and batch — with ~3 ms overhead vs ~7 ms for set_detect_anomaly.


# pytorch-nan-detector

Lightweight forward-hook NaN/Inf detector for PyTorch.  
Pinpoints the **exact layer and batch** where NaNs first appear — with ~3 ms overhead.

> Companion code for the Towards Data Science article:  
> **[PyTorch NaNs Are Silent Killers — I Built a 2 ms Hook That Pinpoints Them to the Exact Layer and Batch](https://towardsdatascience.com/)**

---

## Benchmark

| Method | Mean (ms) | Overhead |
|---|---|---|
| No detection | 0.65 | baseline |
| **NaNDetector** | **2.84** | **4.4×** |
| `set_detect_anomaly` | 7.29 | 11.2× |

*CPU · 4-layer MLP · batch size 64 · 30 forward passes.  
On GPU with large models, `set_detect_anomaly` reaches 50–100×.*

![Benchmark](assets/plot_benchmark.png)

---

## Install

No package. Single file — drop it into your project:

```bash
curl -O https://raw.githubusercontent.com/Emmimal/pytorch-nan-detector/main/nan_detector.py
```

**Requires:** `torch>=1.11` · `matplotlib>=3.5` · `Python>=3.10`

---

## Usage

See [`examples/basic_usage.py`](examples/basic_usage.py) and [`examples/backward_hooks.py`](examples/backward_hooks.py).

Full API reference and walkthrough in the [TDS article](https://towardsdatascience.com/).

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

Run all three demos:

```bash
python nan_detector.py
```

Saves `plot_loss_curve.png`, `plot_grad_norms.png`, `plot_benchmark.png`.

---

## Plots

![Loss curve](assets/plot_loss_curve.png)
![Grad norms](assets/plot_grad_norms.png)

---

## Tests

```bash
python -m pytest tests/ -v
```

---

## License

MIT — see [LICENSE](LICENSE)

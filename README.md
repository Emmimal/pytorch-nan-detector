# pytorch-nan-detector
PyTorch NaNs are silent killers. This hook catches them at the exact layer and batch — with ~3 ms overhead vs ~7 ms for set_detect_anomaly.

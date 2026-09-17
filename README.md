# SignComparisonCNN — BISINDO Sign Recognition

Comparison of two deep learning architectures for BISINDO isolated word/gloss recognition from video: **CNN_LSTM** (ResNet18 + LSTM + Attention) vs **MobileNetTransformer** (MobileNetV2 + Transformer). Self-collected "WLBisindo" dataset, 32 word classes.

## Project Structure

```
data/WLBisindo/
  raw/<label>/<video>.mp4     # raw video, before preprocessing
  frames/<label>/<video>/*.jpg  # extracted frames
  split/{train,val,test}/<label>/<video>/*.jpg  # dataset split
  classes.csv                  # list of 32 classes (semantic order, not the training label order)
models/
  cnn_lstm.py                  # CNN_LSTM (ResNet18 + LSTM + Attention)
  mobile_net.py                # MobileNetTransformer (MobileNetV2 + Transformer)
utils/
  dataloader.py                # VideoDataset, augmentation, motion_mode (diff/optical_flow)
  losses.py                    # FocalLoss
  metrics.py, experiment_logger.py
  EDA.ipynb                    # data exploration
scripts/
  extract_frames.ipynb         # extract frames from raw video
  split_dataset.ipynb          # train/val/test split
train/
  train_cnn_lstm.ipynb         # CNN_LSTM training (baseline + ablations)
  train_mobilenet.ipynb        # MobileNetTransformer training (baseline + ablations)
  model_evaluation.ipynb       # dashboard: aggregates every run (config+metrics) for both models
evaluation/
  efficiency_benchmark.ipynb   # params/FLOPs/latency
  statistical_analysis.ipynb   # per-class + McNemar's test + bootstrap CI
outputs/
  logs/<cnn_lstm|mobile_net>/run_<timestamp>[_<tag>]/  # checkpoint & log per run
  metrics/experiments.csv      # summary of every run
```

## Environment Setup

Environment used: conda env `bisindo`.

```powershell
conda activate bisindo
pip install -r requirements.txt
```

Check CUDA availability (optional, training is much faster with a GPU):
```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

## Replicating From Scratch (new / empty dataset)

If `data/WLBisindo/split/` is empty, do these steps first before training. If the dataset is already split (check the `data/WLBisindo/split/{train,val,test}/` folders), skip straight to the **Training** section.

1. **Prepare raw video.** Place videos per class at `data/WLBisindo/raw/<label_name>/<video>.mp4`. If using a different dataset/folder naming, adjust the paths in `scripts/extract_frames.ipynb`.
2. **Extract frames.** Open `scripts/extract_frames.ipynb`, **Run All**. Frames are taken from the middle 60% of each video, filtered by motion+blur, and saved to `data/WLBisindo/frames/<label>/<video>/*.jpg`.
3. **Check data quality (optional).** Open `utils/EDA.ipynb`, **Run All** — inspect the video-count distribution per class, detect class imbalance, etc.
4. **Split the dataset.** Open `scripts/split_dataset.ipynb`, **Run All**. Splits `frames/` into `split/train`, `split/val`, `split/test` (70/15/15, stratified per class).
5. Continue to the **Training** section below.

## How to Run — Model Training

Training now lives in notebooks with a `CONFIG` cell at the top, so a single notebook can either reproduce the baseline OR run a novelty ablation variant, depending on what's in `CONFIG`.

1. Open `train/train_cnn_lstm.ipynb` (for CNN_LSTM) or `train/train_mobilenet.ipynb` (for MobileNetTransformer) in PyCharm/Jupyter.
2. Make sure the kernel/interpreter uses the `bisindo` environment.
3. Edit the **`CONFIG`** cell if needed (the defaults already reproduce the original baseline — leave them as-is for a plain baseline run):
   - `unfreeze_blocks`: `"full"` (fine-tune the whole backbone) or an integer N (fine-tune only the last N blocks).
   - `motion_mode`: `"diff"` (default) or `"optical_flow"`.
   - `pooling` (MobileNetTransformer only): `"last"` (default) / `"mean"` / `"attention"`.
   - `run_tag`: label used in the run folder name, e.g. `"unfreeze-4"` — leave empty for the default `run_<timestamp>` name.
4. **Run All**. Training progress is visible cell-by-cell (tqdm progress bar per epoch, log printed in the cell output).
5. Once finished, check the new folder under `outputs/logs/<cnn_lstm|mobile_net>/run_<timestamp>[_<tag>]/` — it contains `best_model.pth`, `last_model.pth`, `config.json`, `evaluation.json` (val), `test_eval.json` (test), `classification_report.txt`, `confusion_matrix.png`, `log.txt`.
6. A new row is automatically appended to `outputs/metrics/experiments.csv`.

**Monitor progress live via TensorBoard** (run in a separate terminal while training is in progress):
```powershell
tensorboard --logdir=outputs/logs
```
Open `http://localhost:6006`. Logs (`Loss`, `Accuracy`, `Overfitting/gap`, `LR`, `Weights`/`Grads` histograms) are flushed at the end of every epoch, so the charts update live.

---

## Novelty Additions (for a Scopus conference/journal submission)

The following four additions round out the existing CNN_LSTM vs MobileNetTransformer comparison, filling gaps reviewers typically ask about (efficiency, statistical significance, comparison fairness, and a motion representation ablation). All of them are **notebooks** (`.ipynb`), run with Jupyter/PyCharm just like the older training notebooks, so progress stays visible cell-by-cell.

The old `train_cnn.ipynb` and `train_mobile_net.ipynb` notebooks have been **replaced** by `train/train_cnn_lstm.ipynb` and `train/train_mobilenet.ipynb` below — both can reproduce the original baseline (default config) as well as every ablation variant (via the `CONFIG` cell).

### 1 & 2. Training with ablations (Novelty #2 fairness/pooling, #4 optical flow)

Same usage as the **How to Run — Model Training** section above, just fill in the `CONFIG` cell for whichever ablation variant you want to try. Available fields:
- `unfreeze_blocks`: `"full"` (baseline) or an integer N (fine-tuning depth ablation).
- `motion_mode`: `"diff"` (baseline) or `"optical_flow"` (motion representation ablation).
- `pooling` (`train_mobilenet.ipynb` only): `"last"` (baseline) / `"mean"` / `"attention"` (pooling ablation).
- `transformer_layers` (`train_mobilenet.ipynb` only): number of Transformer layers (baseline: 4).

Recommended ablation combinations for the paper:
```
train_mobilenet.ipynb   : unfreeze_blocks="full"        -> fairness-matched vs CNN_LSTM baseline
train_cnn_lstm.ipynb    : unfreeze_blocks=4              -> fairness-matched, other direction
train_mobilenet.ipynb   : pooling="mean"                 -> pooling ablation
train_mobilenet.ipynb   : pooling="attention"            -> pooling ablation
train_cnn_lstm.ipynb    : motion_mode="optical_flow"     -> motion representation ablation
train_mobilenet.ipynb   : motion_mode="optical_flow"     -> motion representation ablation
```

### 3. Efficiency comparison (Novelty #1)

`evaluation/efficiency_benchmark.ipynb` — compares parameter counts, FLOPs (`torch.profiler` + a manual LSTM correction), and CPU/CUDA latency for two checkpoints. Edit the `CNN_CKPT` / `MOBILENET_CKPT` cell to pick which pair of runs to compare (default: each model's most recent run), then **Run All**. The MobileNet architecture (`pooling`, `transformer_layers`) is read automatically from the checkpoint's `config.json`. Output: `outputs/metrics/efficiency_comparison__<run1>__vs__<run2>.json`.

### 4. Per-class analysis + statistical significance (Novelty #3)

`evaluation/statistical_analysis.ipynb` — evaluates two checkpoints on the same (paired) test samples, then computes per-class precision/recall/F1, confusion matrices, the most frequently confused classes, **McNemar's test**, and a **paired bootstrap CI** for the accuracy difference. Edit the `CNN_CKPT` / `MOBILENET_CKPT` cell, then **Run All**. Automatically handles checkpoints trained with different `motion_mode` values across models. Output: `outputs/metrics/statistical_analysis__<run1>__vs__<run2>.json` and `per_class_report__<run1>__vs__<run2>.csv`.

### Dashboard comparing every run

`train/model_evaluation.ipynb` — **Run All** after any training run (baseline or ablation). Automatically scans every `outputs/logs/{cnn_lstm,mobile_net}/run_*/` folder, reads each run's `config.json` + `evaluation.json` + `test_eval.json`, merges them into one table (saved to `outputs/metrics/run_comparison.csv`), then displays four paper-ready comparisons:
- Baseline CNN_LSTM vs MobileNetTransformer (test set)
- Fairness-matched fine-tuning depth (run pairs sharing the same `unfreeze_blocks` across both models)
- Pooling ablation (`last` vs `mean` vs `attention`, MobileNetTransformer)
- Motion representation ablation (`diff` vs `optical_flow`, both models)

Also surfaces a summary of `efficiency_comparison__*.json` and `statistical_analysis__*.json` if they already exist (produced by the evaluation notebooks above). This is the single dashboard for reviewing every novelty result before writing it up in the paper.

Full details, change history, and implementation status are in `CHANGELOG.md`.

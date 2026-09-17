# Changelog

Technical history of the CNN_LSTM vs MobileNetTransformer comparison pipeline, focused on the additions made to support four novelty contributions for a conference/journal (Scopus) submission: efficiency comparison, fairness-matched fine-tuning depth + pooling ablation, per-class/statistical significance analysis, and an optical-flow motion representation ablation.

## Pipeline rebuild: notebook-based training with ablation support

The original training scripts (`train_cnn.ipynb`, `train_mobile_net.ipynb`) only reproduced a single fixed configuration each. They were replaced with `train/train_cnn_lstm.ipynb` and `train/train_mobilenet.ipynb`, which reproduce the original baselines under their default `CONFIG` cell and additionally expose the knobs needed for every ablation (`unfreeze_blocks`, `motion_mode`, and for MobileNetTransformer also `pooling`/`transformer_layers`). Each run writes to its own timestamped folder under `outputs/logs/<model>/run_<timestamp>[_<tag>]/`, so no run overwrites another.

`scripts/test_cnn.ipynb` and `scripts/test_mobilenet.ipynb` (standalone test-set evaluation + real-time webcam inference) were removed as out of scope for the paper comparison — test-set evaluation now happens automatically at the end of every training run (`test_eval.json`, `classification_report.txt`, `confusion_matrix.png`), and the webcam path was never wired to `motion_mode`/`pooling` config correctly in the first place.

`train/model_evaluation.ipynb` was rewritten from a plain `experiments.csv` plot into a dashboard that scans every run folder under `outputs/logs/{cnn_lstm,mobile_net}/run_*/`, reads each run's `config.json` + `evaluation.json` + `test_eval.json`, and produces four paper-ready comparisons: baseline (test set), fairness-matched fine-tuning depth, pooling ablation, and motion representation ablation. It also surfaces results from the two analysis notebooks below when available. Runs without a `config.json` (failed/incomplete) are skipped automatically.

Old run outputs that failed or were superseded by later tuning iterations were deleted (no `config.json`/checkpoint, or accuracy too low to be useful). Two runs were kept as the pre-novelty baseline: `outputs/logs/cnn_lstm/run_20260415_195341` and `outputs/logs/mobile_net/run_20260415_201349` — the only runs with a complete `test_eval.json`.

## Model changes (`models/cnn_lstm.py`, `models/mobile_net.py`)

- **`CNN_Encoder.set_trainable_blocks(n)`** (ResNet18) and **`MobileNetEncoder.set_trainable_blocks(n)`** (MobileNetV2) were added so both backbones can be fine-tuned to a matched depth. Previously CNN_LSTM was always fully unfrozen while MobileNetTransformer only ever unfroze its last 4 of 19 top-level blocks (hardcoded) — an unmatched comparison that the fairness ablation now corrects.
- **`MobileNetTransformer.pooling`** (`"last"` / `"mean"` / `"attention"`) was added as a real constructor argument. The original model only used `x[:, -1, :]` (last-token pooling), discarding the Transformer's output for every earlier frame. `AttentionPool` mirrors the `Attention` class already used by CNN_LSTM, so `pooling="attention"` removes one more confound between the two architectures.
- **`MobileNetTransformer.num_layers`** is now a real constructor argument (previously hardcoded to 4). Some old run `config.json` files record `"transformer_layers": 2`, which was stale/incorrect metadata — the checkpoints in question were actually trained with 4 layers. `model_evaluation.ipynb`'s `normalize_config()` corrects this explicitly when reading old-style configs.

## Data pipeline changes (`utils/dataloader.py`)

- **`VideoDataset(motion_mode=...)`**: `"diff"` (default) keeps the original behavior (per-pixel difference of normalized RGB frames). `"optical_flow"` adds dense Farneback optical flow computed on true pixel intensities (Normalize is inverted first — see `_to_gray_uint8`), encoded as 3 channels (horizontal flow, vertical flow, magnitude), each scaled by `FLOW_MAX_DISPLACEMENT=20px`.
- **I/O performance fix**: `__getitem__` previously called `os.listdir(vid_path)` on every sample access (every sample, every epoch, every worker). Frame lists are now cached once in `_load_dataset()` into `self.frame_lists` and reused — identical results, far less disk I/O.
- **`get_dataloader()`**: added `persistent_workers=True` and `prefetch_factor=4` (I/O-only, does not affect batch composition or gradients). Default `num_workers` raised from 4 to 8 in the training notebooks. `batch_size` and every other hyperparameter that affects training dynamics were deliberately left unchanged to keep every run comparable to the two preserved baselines.

## Loss function (`utils/losses.py`, new)

A single, correct `FocalLoss` shared by both training notebooks, replacing two previously duplicated and independently incomplete implementations:
- One computed `pt = exp(-ce_loss)` from a batch-mean cross-entropy loss instead of a per-sample one, collapsing the focal modulation to a single value per batch.
- The other computed `pt` correctly per-sample but had no support for class weights (`alpha`), needed for this dataset's per-class imbalance (12–35 samples/class).

`utils/losses.py` combines both fixes: `reduction="none"` (per-sample) with optional `alpha` class weights.

## Evaluation notebooks (new)

- **`evaluation/efficiency_benchmark.ipynb`**: parameter counts, FLOPs (`torch.profiler`, with `nn.LSTM` FLOPs added back analytically since the profiler doesn't instrument `aten::lstm`), and CPU/CUDA latency for a pair of checkpoints selected via the `CNN_CKPT`/`MOBILENET_CKPT` cell. Reads `pooling`/`transformer_layers` back from each checkpoint's `config.json` so it works for any ablation variant.
- **`evaluation/statistical_analysis.ipynb`**: evaluates both checkpoints on the same test samples (verified index-for-index via `assert`), then computes per-class precision/recall/F1, confusion matrices, top confused class pairs, McNemar's test (exact binomial for <25 discordant pairs, continuity-corrected chi-square otherwise — implemented directly against `scipy.stats`, no `statsmodels` dependency), and a paired bootstrap CI for the accuracy difference. Handles checkpoints trained with different `motion_mode` by building a separate test dataset per model when needed.

## Ablation protocol

Each run resets to its model's baseline defaults and changes exactly one field — configurations are not stacked, so any accuracy change can be attributed to a single variable.

**`train_cnn_lstm.ipynb` baseline**: `unfreeze_blocks="full"`, `motion_mode="diff"` (preserved run: `run_20260415_195341`)
**`train_mobilenet.ipynb` baseline**: `unfreeze_blocks=4`, `pooling="last"`, `transformer_layers=4`, `motion_mode="diff"` (preserved run: `run_20260415_201349`)

| # | Notebook | Field changed from default | `run_tag` |
|---|----------|------------------------------|-----------|
| 1 | `train_cnn_lstm.ipynb` | `unfreeze_blocks=4` | `unfreeze-4` |
| 2 | `train_mobilenet.ipynb` | `unfreeze_blocks="full"` | `unfreeze-full` |
| 3 | `train_mobilenet.ipynb` | `pooling="mean"` | `pool-mean` |
| 4 | `train_mobilenet.ipynb` | `pooling="attention"` | `pool-attention` |
| 5 | `train_cnn_lstm.ipynb` | `motion_mode="optical_flow"` | `flow` |
| 6 | `train_mobilenet.ipynb` | `motion_mode="optical_flow"` | `flow` |

Run sequentially (not in parallel) to avoid GPU contention. Together with the two baselines, these 8 runs are sufficient for all four comparison sections in `train/model_evaluation.ipynb`. Combined-variant runs (e.g. MobileNet with unfreeze full + pooling attention + optical flow at once) can be added separately as exploratory "best combination" runs, but do not replace the single-variable ablations above — the two answer different questions.

## MobileNetTransformer training speed (mixed precision)

The preserved MobileNet baseline (`run_20260415_201349`) trained the full 50/50 epochs (val accuracy was still improving at epoch 45), so its wall-clock cost is inherent to the number of epochs needed to converge, not wasted epochs from a loose early-stopping setting. To make the 4 upcoming MobileNet ablation runs (rows 2, 3, 4, 6 below) practical to run, `train/train_mobilenet.ipynb` now trains with automatic mixed precision when CUDA is available (`CONFIG["use_amp"] = True`, `torch.amp.autocast` + `GradScaler`) and a slightly more responsive LR schedule (`scheduler_patience`: 2 → 1). Measured on this project's model/input shape (batch=8, seq_len=20, RTX-class GPU): **1.74x faster per training step** (107.7ms → 62.0ms).

This is a deliberate, documented deviation from bit-for-bit reproduction of the original baseline recipe, traded for speed — not one of the studied ablation variables (fine-tuning depth / pooling / motion representation). `use_amp` and the new `scheduler_patience` value are recorded in every new run's `config.json` for traceability. `train_cnn_lstm.ipynb` is unchanged (fp32, `scheduler_patience=2`) since CNN_LSTM training speed was not reported as a bottleneck.

## Status

Data (`data/WLBisindo/split/{train,val,test}`) is prepared. Two baseline runs are complete. The 6-run ablation protocol above has not yet been executed.

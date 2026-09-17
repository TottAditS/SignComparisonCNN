# SignComparisonCNN — BISINDO Sign Recognition

Perbandingan dua arsitektur deep learning untuk pengenalan kata BISINDO (isolated word/gloss) dari video: **CNN_LSTM** (ResNet18 + LSTM + Attention) vs **MobileNetTransformer** (MobileNetV2 + Transformer). Dataset self-collected "WLBisindo", 32 kelas kata.

## Struktur Proyek

```
data/WLBisindo/
  raw/<label>/<video>.mp4     # video mentah, sebelum diproses
  frames/<label>/<video>/*.jpg  # hasil ekstraksi frame
  split/{train,val,test}/<label>/<video>/*.jpg  # hasil split dataset
  classes.csv                  # daftar 32 kelas (urutan semantik, bukan urutan label training)
models/
  cnn_lstm.py                  # CNN_LSTM (ResNet18 + LSTM + Attention)
  mobile_net.py                # MobileNetTransformer (MobileNetV2 + Transformer)
utils/
  dataloader.py                # VideoDataset, augmentasi, motion_mode (diff/optical_flow)
  losses.py                    # FocalLoss
  metrics.py, evaluate.py, experiment_logger.py
  EDA.ipynb                    # eksplorasi data
scripts/
  extract_frames.ipynb         # ekstraksi frame dari raw video
  split_dataset.ipynb          # split train/val/test
train/
  train_cnn_lstm.ipynb         # training CNN_LSTM (baseline + ablasi)
  train_mobilenet.ipynb        # training MobileNetTransformer (baseline + ablasi)
  model_evaluation.ipynb       # dashboard: kumpulkan semua run (config+metrik) kedua model
evaluation/
  efficiency_benchmark.ipynb   # params/FLOPs/latency
  statistical_analysis.ipynb   # per-class + McNemar's test + bootstrap CI
outputs/
  logs/<cnn_lstm|mobile_net>/run_<timestamp>[_<tag>]/  # checkpoint & log per run
  metrics/experiments.csv      # ringkasan semua run
```

## Setup Environment

Environment yang dipakai: conda env `bisindo`.

```powershell
conda activate bisindo
pip install -r requirements.txt
```

Cek CUDA tersedia (opsional, training jauh lebih cepat dengan GPU):
```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

## Cara Replikasi Dari Awal (dataset baru / dataset kosong)

Jika `data/WLBisindo/split/` belum ada isinya, lakukan langkah ini dulu sebelum training. Kalau dataset sudah ter-split (cek folder `data/WLBisindo/split/{train,val,test}/`), langsung lompat ke bagian **Training**.

1. **Siapkan video mentah.** Taruh video per kelas di `data/WLBisindo/raw/<nama_label>/<video>.mp4`. Kalau pakai dataset/nama folder berbeda, sesuaikan path di `scripts/extract_frames.ipynb`.
2. **Ekstraksi frame.** Buka `scripts/extract_frames.ipynb`, **Run All**. Frame diambil dari 60% tengah video, difilter motion+blur, disimpan ke `data/WLBisindo/frames/<label>/<video>/*.jpg`.
3. **Cek kualitas data (opsional).** Buka `utils/EDA.ipynb`, **Run All** — lihat distribusi jumlah video per kelas, deteksi ketidakseimbangan kelas, dll.
4. **Split dataset.** Buka `scripts/split_dataset.ipynb`, **Run All**. Membagi `frames/` menjadi `split/train`, `split/val`, `split/test` (70/15/15, stratified per kelas).
5. Lanjut ke bagian **Training** di bawah.

## Cara Run — Training Model

Training sekarang dalam bentuk notebook dengan cell `CONFIG` di bagian atas, jadi satu notebook bisa mereproduksi baseline ATAU menjalankan varian ablasi novelty, tergantung isi `CONFIG`.

1. Buka `train/train_cnn_lstm.ipynb` (untuk CNN_LSTM) atau `train/train_mobilenet.ipynb` (untuk MobileNetTransformer) di PyCharm/Jupyter.
2. Pastikan kernel/interpreter memakai environment `bisindo`.
3. Edit cell **`CONFIG`** kalau perlu (default sudah mereproduksi baseline asli — biarkan default untuk training baseline biasa):
   - `unfreeze_blocks`: `"full"` (fine-tune seluruh backbone) atau angka N (fine-tune N block terakhir saja).
   - `motion_mode`: `"diff"` (default) atau `"optical_flow"`.
   - `pooling` (khusus MobileNetTransformer): `"last"` (default) / `"mean"` / `"attention"`.
   - `run_tag`: label buat nama folder run, misal `"unfreeze-4"` — kosongkan untuk nama default `run_<timestamp>`.
4. **Run All**. Progress training terlihat cell-per-cell (progress bar tqdm per epoch, log tercetak di output cell).
5. Setelah selesai, cek folder baru di `outputs/logs/<cnn_lstm|mobile_net>/run_<timestamp>[_<tag>]/` — isinya `best_model.pth`, `last_model.pth`, `config.json`, `evaluation.json` (val), `test_eval.json` (test), `classification_report.txt`, `confusion_matrix.png`, `log.txt`.
6. Baris baru otomatis ditambahkan ke `outputs/metrics/experiments.csv`.

**Pantau progress live via TensorBoard** (jalankan di terminal terpisah sambil training berjalan):
```powershell
tensorboard --logdir=outputs/logs
```
Buka `http://localhost:6006`. Log (`Loss`, `Accuracy`, `Overfitting/gap`, `LR`, histogram `Weights`/`Grads`) di-flush tiap akhir epoch, jadi grafik ter-update live.

---

## Novelty Additions (untuk submission conference/journal Scopus)

Empat penambahan berikut melengkapi perbandingan CNN_LSTM vs MobileNetTransformer yang sudah ada, mengisi gap yang biasanya ditanya reviewer (efisiensi, signifikansi statistik, fairness perbandingan, dan ablasi motion representation). Semuanya berbentuk **notebook** (`.ipynb`), dijalankan dengan Jupyter/PyCharm seperti notebook training lama, supaya progressnya kelihatan cell-per-cell.

Notebook lama `train_cnn.ipynb` dan `train_mobile_net.ipynb` sudah **digantikan** oleh `train/train_cnn_lstm.ipynb` dan `train/train_mobilenet.ipynb` di bawah — keduanya bisa reproduce baseline asli (default config) sekaligus semua varian ablasi (lewat cell `CONFIG`).

### 1 & 2. Training dengan ablasi (Novelty #2 fairness/pooling, #4 optical flow)

Pakai cara pakai yang sama dengan bagian **Cara Run — Training Model** di atas, cuma isi cell `CONFIG` sesuai varian ablasi yang mau dicoba. Field yang tersedia:
- `unfreeze_blocks`: `"full"` (baseline) atau angka N (fine-tuning depth ablation).
- `motion_mode`: `"diff"` (baseline) atau `"optical_flow"` (motion representation ablation).
- `pooling` (khusus `train_mobilenet.ipynb`): `"last"` (baseline) / `"mean"` / `"attention"` (pooling ablation).
- `transformer_layers` (khusus `train_mobilenet.ipynb`): jumlah layer Transformer (baseline: 4).

Kombinasi ablasi yang direkomendasikan untuk paper:
```
train_mobilenet.ipynb   : unfreeze_blocks="full"        -> fairness-matched vs CNN_LSTM baseline
train_cnn_lstm.ipynb    : unfreeze_blocks=4              -> fairness-matched arah sebaliknya
train_mobilenet.ipynb   : pooling="mean"                 -> pooling ablation
train_mobilenet.ipynb   : pooling="attention"            -> pooling ablation
train_cnn_lstm.ipynb    : motion_mode="optical_flow"     -> motion representation ablation
train_mobilenet.ipynb   : motion_mode="optical_flow"     -> motion representation ablation
```

### 3. Efficiency comparison (Novelty #1)

`evaluation/efficiency_benchmark.ipynb` — bandingkan params, FLOPs (`torch.profiler` + koreksi manual LSTM), dan latency CPU/CUDA dari dua checkpoint. Edit cell `CNN_CKPT` / `MOBILENET_CKPT` untuk pilih pasangan run yang mau dibandingkan (default: run terbaru masing-masing model), lalu **Run All**. Arsitektur MobileNet (`pooling`, `transformer_layers`) dibaca otomatis dari `config.json` checkpoint. Output: `outputs/metrics/efficiency_comparison__<run1>__vs__<run2>.json`.

### 4. Per-class analysis + statistical significance (Novelty #3)

`evaluation/statistical_analysis.ipynb` — evaluasi dua checkpoint pada sample test yang sama (paired), lalu hitung per-class precision/recall/F1, confusion matrix, kelas yang paling sering tertukar, **McNemar's test**, dan **paired bootstrap CI** untuk selisih akurasi. Edit cell `CNN_CKPT` / `MOBILENET_CKPT`, lalu **Run All**. Otomatis menangani checkpoint dengan `motion_mode` yang berbeda antar model. Output: `outputs/metrics/statistical_analysis__<run1>__vs__<run2>.json` dan `per_class_report__<run1>__vs__<run2>.csv`.

### Dashboard perbandingan semua run

`train/model_evaluation.ipynb` — **Run All** setelah training run apa pun (baseline atau ablasi). Otomatis scan semua folder `outputs/logs/{cnn_lstm,mobile_net}/run_*/`, baca `config.json` + `evaluation.json` + `test_eval.json` tiap run, gabung jadi satu tabel (disimpan ke `outputs/metrics/run_comparison.csv`), lalu tampilkan 4 perbandingan siap pakai untuk paper:
- Baseline CNN_LSTM vs MobileNetTransformer (test set)
- Fairness-matched fine-tuning depth (pasangan run dengan `unfreeze_blocks` sama di kedua model)
- Pooling ablation (`last` vs `mean` vs `attention`, MobileNetTransformer)
- Motion representation ablation (`diff` vs `optical_flow`, kedua model)

Juga menampilkan ringkasan dari `efficiency_comparison__*.json` dan `statistical_analysis__*.json` kalau sudah ada (hasil dari notebook evaluation di atas). Ini dashboard tunggal yang jadi tempat lihat semua hasil novelty sebelum dituliskan ke paper.

Detail lengkap, riwayat perubahan, dan status implementasi ada di `PROGRESS.md`.

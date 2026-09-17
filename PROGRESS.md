# Progress Checkpoint — Novelty Additions for Paper

Status per 13 September 2026 (sesi ke-5). **Repo sudah di-rebuild bersih** sesuai permintaan user: script/notebook lama yang buggy/tidak lengkap sudah diganti, output run yang gagal/rusak sudah dihapus. Semua 4 novelty sekarang berbentuk **notebook** (bukan CLI), user sudah punya backup repo lama jadi aman.

## Tujuan

Menambahkan 4 novelty ke project SignComparisonCNN (CNN-LSTM+Attention vs MobileNetV2+Transformer untuk BISINDO) supaya layak submit ke conference/journal Scopus:

1. **Efficiency comparison** — params, FLOPs, latency CPU/CUDA kedua model.
2. **Fairness fix + pooling ablation** — samakan strategi fine-tuning, tambah opsi pooling di MobileNetTransformer.
3. **Per-class analysis + statistical significance** — per-class F1, confusion matrix, McNemar's test, bootstrap CI.
4. **Optical flow ablation** — Farneback optical flow sebagai alternatif frame-diff.

## Keputusan penting sesi ini: notebook, bukan CLI

User kerja di PyCharm dan mau progress kelihatan cell-per-cell seperti notebook training lama. Setelah sempat coba pendekatan CLI (`train/train_ablation.py` + `evaluation/*.py`), user minta **replace total** — hapus semua script/notebook lama yang buggy, ganti dengan versi notebook yang mendukung ablasi lewat cell `CONFIG`. Ini sudah dieksekusi penuh.

## Yang dihapus (superseded / rusak)

### Output run gagal/rusak — **dihapus permanen** (user sudah backup repo lama)
- `outputs/logs/cnn_lstm/`: `run_20260408_235004` (tanpa config/eval), `run_20260409_085556`, `run_20260409_092330`, `run_20260409_111427`, `run_20260415_151718`, `run_20260415_153551` (semua iterasi tuning awal, sudah digantikan), `run_20260415_161750` (cuma file event TensorBoard, tanpa checkpoint), `run_20260415_162628`.
- `outputs/logs/mobile_net/`: `run_20260409_100644`, `run_20260409_113840`, `run_20260415_170740` (tanpa checkpoint, ada catatan user sendiri "akurasinya masih rendah"), `run_20260415_182414`, `run_20260415_191345`.
- **Disisakan** (baseline pembanding "sebelum novelty"): `outputs/logs/cnn_lstm/run_20260415_195341` dan `outputs/logs/mobile_net/run_20260415_201349` — satu-satunya run yang punya `test_eval.json` lengkap.
- `outputs/metrics/experiments.csv` dibersihkan, cuma sisa 4 baris baseline resmi (2 baris `evaluation.json`-style lama + 2 baris `test_eval.json`-style/TEST).

### File kode — **dihapus, digantikan versi baru**
- `train/train_cnn.ipynb` → digantikan `train/train_cnn_lstm.ipynb`
- `train/train_mobile_net.ipynb` → digantikan `train/train_mobilenet.ipynb`
- `train/train_ablation.py` (CLI, dari sesi sebelumnya) → dihapus, logic-nya pindah ke 2 notebook di atas
- `evaluation/efficiency_benchmark.py` (CLI) → digantikan `evaluation/efficiency_benchmark.ipynb`
- `evaluation/statistical_analysis.py` (CLI) → digantikan `evaluation/statistical_analysis.ipynb`

### File yang TIDAK disentuh (tetap dipakai)
- `scripts/extract_frames.ipynb`, `scripts/split_dataset.ipynb` — di luar scope novelty.
- `train/model_evaluation.ipynb` — exploratory plot baca `experiments.csv`, masih jalan normal dengan CSV yang sudah dibersihkan.
- `utils/EDA.ipynb`.

## Update sesi ke-6: fix test_cnn.ipynb & test_mobilenet.ipynb

User tanya apakah `scripts/test_cnn.ipynb`, `scripts/test_mobilenet.ipynb`, dan `train/model_evaluation.ipynb` perlu diubah juga. Jawabannya: `model_evaluation.ipynb` tidak perlu (cuma plot dari CSV, tidak terpengaruh), tapi `test_cnn.ipynb`/`test_mobilenet.ipynb` **ternyata punya bug nyata** kalau dipakai untuk checkpoint hasil ablasi novelty — ini ditemukan saat re-check dan langsung diperbaiki:

1. **Bug utama (blocking untuk checkpoint ablasi)**: kedua notebook selalu instansiasi model dengan default (`MobileNetTransformer(num_classes=...)` tanpa `pooling`/`num_layers`) dan `get_dataloader(...)` tanpa `motion_mode`. Akibatnya:
   - Checkpoint hasil `--pooling attention` akan **crash** saat `load_state_dict` (state_dict punya key `attention_pool.*` yang tidak ada di model default `pooling="last"`).
   - Checkpoint hasil `motion_mode="optical_flow"` akan **diam-diam salah** dievaluasi (dievaluasi dengan data frame-diff, bukan optical flow — tidak error karena shape 6-channel tetap sama, tapi hasil metrik jadi tidak berarti).
   - **Fix**: ditambahkan fungsi `get_run_config(model_path)` di kedua notebook yang membaca `config.json` di sebelah checkpoint (ditulis oleh `train_cnn_lstm.ipynb`/`train_mobilenet.ipynb` yang baru), lalu `motion_mode`/`pooling`/`transformer_layers` diteruskan ke `get_dataloader()` dan constructor model. Kalau `config.json` tidak ada (misal checkpoint sangat lama), fallback ke default baseline (aman, backward-compatible).

2. **Bug independen (pre-existing, bukan buatan sesi novelty ini) di bagian realtime webcam**: baik `test_cnn.ipynb` maupun `test_mobilenet.ipynb` punya cell live-camera yang membangun channel ke-4/5/6 dengan `torch.cat([prev_img, img])` — yaitu **menempelkan dua frame RGB mentah**, BUKAN RGB+diff seperti yang dipakai `utils/dataloader.py` saat training. Ini artinya versi live-camera dari codebase asli selalu memberi model input yang polanya berbeda dari training-nya sendiri (meski secara shape tetap 6-channel jadi tidak error). Karena user minta repo bersih dari bug/tidak lengkap, ini ikut diperbaiki:
   - Ditambahkan fungsi `to_gray_uint8()` dan `compute_optical_flow_channels()` di kedua notebook yang **meniru persis** logic `VideoDataset._to_gray_uint8`/`_compute_optical_flow_channels` di `utils/dataloader.py`.
   - Loop webcam sekarang membangun 6-channel sesuai `MOTION_MODE` yang di-resolve dari `config.json` checkpoint yang dipakai (`"diff"` → RGB+selisih normalized-RGB frame sebelumnya, sama seperti training; `"optical_flow"` → RGB+Farneback flow dari grayscale invert-normalize, sama seperti training).
   - `MobileNetTransformer` di cell realtime `test_mobilenet.ipynb` sekarang juga dibuat dengan `pooling`/`num_layers` yang benar dari `config.json`.

Kedua notebook sudah divalidasi ulang: valid JSON + valid syntax Python per cell (`ast.parse`), via script sementara yang sudah dihapus setelah pakai.

**Kesimpulan untuk user**: sekarang ketiga notebook (`train_cnn_lstm.ipynb`, `train_mobilenet.ipynb`, `test_cnn.ipynb`, `test_mobilenet.ipynb`) sudah saling konsisten — checkpoint hasil ablasi apa pun (baseline, unfreeze depth, pooling, motion_mode) bisa langsung dipakai di `test_cnn.ipynb`/`test_mobilenet.ipynb` tanpa perlu edit manual, cukup **Run All** seperti biasa (notebook otomatis pakai `get_latest_model()` untuk ambil run terbaru; kalau mau pilih run spesifik, edit variabel `log_dir`/ganti cara `MODEL_PATH` di-resolve secara manual).

## Update sesi ke-7: hapus realtime webcam testing, model_evaluation.ipynb jadi dashboard pusat

User memutuskan **realtime webcam testing dihapus total dari scope** — fokus balik ke tujuan awal (cuma bandingkan CNN_LSTM vs MobileNetTransformer dan kombinasi ablasinya untuk paper, bukan aplikasi real-time). Perubahan:

1. **Dihapus**: `scripts/test_cnn.ipynb` dan `scripts/test_mobilenet.ipynb` (termasuk semua perbaikan bug realtime dari sesi ke-6 di atas — sudah tidak relevan karena filenya dihapus). Evaluasi test-set yang dulu ada di kedua notebook ini sudah **redundant**: `train_cnn_lstm.ipynb`/`train_mobilenet.ipynb` sudah menghasilkan `test_eval.json` + `classification_report.txt` + `confusion_matrix.png` otomatis di akhir setiap run training.

2. **`train/model_evaluation.ipynb` ditulis ulang total** jadi dashboard pusat yang mengumpulkan SEMUA run (baseline + semua ablasi novelty) dari kedua model sekaligus config-nya masing-masing, bukan cuma baca `experiments.csv` (yang cuma text log accuracy tanpa detail config). Cara kerja:
   - `scan_model_runs()` men-scan folder `outputs/logs/{cnn_lstm,mobile_net}/run_*/`, baca `config.json` + `evaluation.json` (val) + `test_eval.json` (test) tiap run yang punya `config.json` (run tanpa config = gagal/incomplete, di-skip otomatis).
   - `normalize_config()` menyatukan skema config lama (baseline, ditulis notebook lama yang sudah dihapus) dan skema baru (ditulis `train_cnn_lstm.ipynb`/`train_mobilenet.ipynb`) jadi satu set kolom yang sama, dengan **koreksi eksplisit** untuk bug lama: config lama MobileNet baseline bilang `transformer_layers: 2` padahal model aslinya hardcode `num_layers=4` — dashboard ini paksa nilai yang benar (4), bukan ikut nilai yang salah di file.
   - Hasil akhir: satu DataFrame besar, disimpan ke `outputs/metrics/run_comparison.csv`, plus 4 section siap pakai:
     a. **Baseline comparison** (test set) — bar chart accuracy/precision/recall/F1 CNN_LSTM vs MobileNetTransformer.
     b. **Fairness-matched fine-tuning depth** — otomatis pasangkan run dari KEDUA model yang punya `unfreeze_blocks` sama (`"full"` atau `4`), bar chart perbandingan.
     c. **Pooling ablation** — group by `pooling` (`last`/`mean`/`attention`) untuk MobileNetTransformer saja.
     d. **Motion representation ablation** — group by `motion_mode` (`diff`/`optical_flow`) untuk kedua model.
   - Section tambahan: scan `outputs/metrics/efficiency_comparison__*.json` dan `statistical_analysis__*.json` (hasil dari `evaluation/efficiency_benchmark.ipynb`/`evaluation/statistical_analysis.ipynb`), tampilkan ringkasan params/FLOPs dan McNemar/bootstrap kalau filenya sudah ada.
   - Semua section punya pesan fallback yang jelas ("belum ada run yang cocok, jalankan X dulu") kalau data yang dibutuhkan belum ada — jadi notebook ini AMAN di-Run-All kapan saja, di tahap manapun (sebelum atau sesudah semua ablasi selesai dijalankan), tidak akan error.

3. Update `README.md`: hapus baris `test_cnn.ipynb`/`test_mobilenet.ipynb` dari struktur proyek dan section "Cara Run", hapus instruksi "Uji real-time via kamera", tambah penjelasan dashboard baru di section Novelty Additions.

Validasi: `train/model_evaluation.ipynb` sudah dicek valid JSON + valid syntax Python per cell (`ast.parse`), via script sementara yang sudah dihapus setelah pakai.

**State akhir repo**: `scripts/` sekarang cuma berisi `extract_frames.ipynb` dan `split_dataset.ipynb` (persiapan data). `train/` berisi `train_cnn_lstm.ipynb`, `train_mobilenet.ipynb` (training+ablasi), dan `model_evaluation.ipynb` (dashboard perbandingan). `evaluation/` berisi `efficiency_benchmark.ipynb` dan `statistical_analysis.ipynb` (analisis pasangan checkpoint spesifik). Tidak ada lagi kode realtime/webcam di repo.

## File baru / final state

### `utils/losses.py` (baru)
Satu `FocalLoss` yang benar, dipakai bersama oleh 2 notebook training. Menggabungkan fix dari 2 versi lama yang masing-masing cacat:
- `train_cnn.ipynb` lama: `reduction='mean'` dipakai SEBELUM `pt=exp(-ce_loss)` dihitung → modulasi focal loss collapse jadi 1 angka per batch, bukan per-sample.
- `train_mobile_net.ipynb` lama: `reduction='none'` benar (per-sample) tapi TIDAK support class weights (`alpha`).
- `utils/losses.py`: `reduction='none'` + `alpha` opsional — gabungan fix yang benar.

### `models/mobile_net.py` (edit)
- `AttentionPool` class baru (identik `Attention` di `cnn_lstm.py`).
- `MobileNetTransformer(num_classes, seq_len=20, num_layers=4, pooling="last")` — `num_layers` sekarang parameter constructor asli (dulu hardcode 4, ada bug config.json lama yang salah bilang `transformer_layers: 2`). Default `pooling="last"` → checkpoint lama tetap bisa di-load tanpa error.
- `MobileNetEncoder.set_trainable_blocks(N)` + `.num_blocks` — kontrol freeze/unfreeze per top-level block (19 block total).

### `models/cnn_lstm.py` (edit)
- `CNN_Encoder.set_trainable_blocks(N)` + `.num_blocks` — versi simetris untuk ResNet18 (9 block: conv1,bn1,relu,maxpool,layer1-4,avgpool).

### `utils/dataloader.py` (edit)
- `VideoDataset(..., motion_mode="diff")` — default `"diff"` = behavior lama persis sama. `"optical_flow"` = Farneback optical flow dari grayscale **raw pixel** (invert Normalize dulu via `_to_gray_uint8`), encode jadi 3 channel (flow_x, flow_y, magnitude) dinormalisasi `/20px`.
- `get_dataloader(..., seq_len=20, motion_mode="diff", num_workers=4)` — parameter baru diteruskan ke `VideoDataset`, semua default identik behavior lama.

### `train/train_cnn_lstm.ipynb` (baru, replace `train_cnn.ipynb`)
Notebook lengkap dengan struktur: Library Setup → **cell `CONFIG`** (unfreeze_blocks, motion_mode, run_tag, hyperparameter) → Setup Path → Dataloader → Check Shape → Setup Log Dir → Build Model & Fine-Tuning Depth → Loss/Optimizer/Scheduler → Save config.json → **Training loop** (try/finally, TensorBoard flush tiap epoch, scalar Loss/Accuracy/Overfitting-gap/LR + histogram Weights/Grads) → Reload best checkpoint → Evaluasi val (`evaluation.json`) → Evaluasi test (`test_eval.json`) → Classification report + confusion matrix → Log ke `experiments.csv` dengan nama deskriptif (`CNN_LSTM_unfreeze-<N|full>_motion-<mode>[_<run_tag>]`).

Default `CONFIG` mereproduksi baseline asli persis (epochs=50, lr=1e-4, wd=1e-5, batch=8, early_stop_patience=3, unfreeze_blocks="full", motion_mode="diff").

### `train/train_mobilenet.ipynb` (baru, replace `train_mobile_net.ipynb`)
Struktur sama dengan di atas, plus cell `CONFIG` punya `pooling` dan `transformer_layers`. Default `CONFIG` mereproduksi baseline asli (epochs=50, lr=3e-5, wd=1e-5, batch=8, early_stop_patience=7, unfreeze_blocks=4, pooling="last", transformer_layers=4, motion_mode="diff"). Nama experiment: `MobileNetTransformer_unfreeze-<N|full>_pool-<mode>_motion-<mode>[_<run_tag>]`.

### `evaluation/efficiency_benchmark.ipynb` (baru, replace `.py`)
Cell `CNN_CKPT`/`MOBILENET_CKPT` (default: run terbaru masing-masing folder, **edit manual untuk pilih pasangan spesifik**). Baca `config.json` checkpoint MobileNet untuk resolve `pooling`/`num_layers` yang benar (`resolve_mobilenet_arch()`) — supaya tidak crash saat load checkpoint hasil ablasi pooling. Output: `outputs/metrics/efficiency_comparison__<run1>__vs__<run2>.json` (nama file mengandung nama run, supaya banyak pasangan bisa dibandingkan tanpa saling menimpa).

### `evaluation/statistical_analysis.ipynb` (baru, replace `.py`)
Sama seperti di atas, plus resolve `motion_mode` per checkpoint (`resolve_motion_mode()`) — kalau CNN_LSTM dan MobileNet dilatih dengan `motion_mode` berbeda, dibangun **dataset test terpisah** per model dengan verifikasi eksplisit (assert `samples` sama & `y_true` sama) sebelum McNemar/bootstrap, supaya pairing tidak diam-diam salah. Output: `outputs/metrics/statistical_analysis__<run1>__vs__<run2>.json` + `per_class_report__<run1>__vs__<run2>.csv`.

## Validasi yang sudah dilakukan

- Semua 4 notebook baru: **valid JSON** (`json.load()` sukses) — dicek via script Python sementara (sudah dihapus setelah pakai).
- Semua code cell di 4 notebook: **valid syntax Python** (`ast.parse()` sukses per cell) — tidak ada `SyntaxError`.
- `utils/losses.py`, `models/cnn_lstm.py`, `models/mobile_net.py`, `utils/dataloader.py`: lolos `python -m py_compile`.
- **Belum dieksekusi dengan torch** — tidak ada torch di python environment yang dipakai tool ini. User yang akan run di conda env `bisindo` sendiri.

## Cara pakai (ringkasan untuk user)

Semua dijalankan sebagai notebook (Jupyter atau lewat PyCharm), kernel harus pakai env `bisindo`.

### Training ablasi
1. Buka `train/train_mobilenet.ipynb`, edit cell `CONFIG`: `unfreeze_blocks="full"`, `run_tag="unfreeze-full"`. **Run All**.
2. Buka `train/train_cnn_lstm.ipynb`, edit cell `CONFIG`: `unfreeze_blocks=4`, `run_tag="unfreeze-4"`. **Run All**.
3. Buka `train/train_mobilenet.ipynb` lagi, `pooling="mean"`, `run_tag="pool-mean"`. **Run All**.
4. Ulangi `pooling="attention"`, `run_tag="pool-attention"`. **Run All**.
5. `train/train_cnn_lstm.ipynb`, `motion_mode="optical_flow"`, `run_tag="flow"`. **Run All**.
6. `train/train_mobilenet.ipynb`, `motion_mode="optical_flow"`, `run_tag="flow"`. **Run All**.

Jalankan satu-satu (tidak paralel) supaya tidak rebutan GPU. Tiap run masuk folder baru, tidak menimpa apa pun.

### Analisis (setelah training di atas selesai)
7. Buka `evaluation/efficiency_benchmark.ipynb`, edit `CNN_CKPT`/`MOBILENET_CKPT` ke pasangan checkpoint yang mau dibandingkan (path lengkap ke `best_model.pth`). **Run All**.
8. Buka `evaluation/statistical_analysis.ipynb`, edit `CNN_CKPT`/`MOBILENET_CKPT` sama. **Run All**.
9. Ulangi 7-8 untuk tiap pasangan checkpoint yang relevan (baseline vs fairness-matched, pooling A vs B, dst).

### Monitoring
Buka terminal kedua: `tensorboard --logdir=outputs/logs` → `http://localhost:6006`. Log ter-flush tiap epoch jadi bisa dipantau live.

## Langkah Selanjutnya
1. User jalankan semua notebook di atas.
2. Kirim hasil (isi `outputs/metrics/*.json` dan `*.csv` yang baru) ke saya.
3. Saya bantu susun narasi "Results"/"Ablation Study" untuk paper.

## Update sesi ke-8: perbaikan speed training + protokol 6-run yang FINAL (jangan diubah-ubah lagi)

### Speed fixes (aman, tidak menyentuh apa pun yang diukur di 4 novelty)

User komplain `train_mobilenet.ipynb` lambat. Ditemukan bug I/O nyata + beberapa optimasi speed-only yang sudah diterapkan:

1. **`utils/dataloader.py` — bug performa**: `VideoDataset.__getitem__()` dulu memanggil `os.listdir(vid_path)` ULANG setiap kali sample diakses (tiap sample, tiap epoch, tiap worker), padahal daftar frame itu sudah dibaca sekali di `_load_dataset()` lalu dibuang. **Fix**: daftar frame di-cache ke `self.frame_lists` sekali di awal, `__getitem__` sekarang reuse cache itu. Hasil identik, cuma jauh lebih sedikit disk I/O.
2. **`utils/dataloader.py` — `get_dataloader()`**: ditambahkan `persistent_workers=(num_workers > 0)` (worker tidak di-restart tiap awal epoch) dan `prefetch_factor=4` (worker siapkan data lebih jauh di depan).
3. **`train_cnn_lstm.ipynb` & `train_mobilenet.ipynb`**: ditambahkan `torch.backends.cudnn.benchmark = True` (cuDNN auto-tune algoritma conv tercepat untuk shape input yang fixed 224x224) tepat setelah `torch.manual_seed()`.
4. **`CONFIG` default `num_workers`**: 4 → **8** di kedua notebook (lebih banyak proses paralel baca/preprocess frame).

**PENTING — yang SENGAJA TIDAK diubah** karena akan merusak fairness perbandingan antar run: `batch_size` (tetap 8), `epochs`, `lr`, `weight_decay`, `focal_gamma`, `grad_clip`, `early_stop_patience`, `scheduler_patience`, `scheduler_factor`, `seq_len`, `seed`. `num_workers` aman diubah karena cuma soal I/O paralel, tidak menyentuh gradient/batch composition sama sekali — beda dengan `batch_size` yang mempengaruhi gradient noise.

CNN_LSTM baseline (`run_20260415_195341`) dan MobileNet baseline (`run_20260415_201349`) sudah selesai duluan dengan `num_workers=4` (default lama) — ini TIDAK masalah untuk fairness karena `num_workers` cuma throughput, bukan hyperparameter training. Run-run baru boleh pakai `num_workers=8` yang lebih cepat tanpa mengorbankan validitas perbandingan.

Semua perubahan sudah divalidasi (`ast.parse` per cell notebook + `py_compile` untuk `dataloader.py`), tidak ada error.

### Protokol 6-run ablasi — FINAL, JANGAN DIUBAH LAGI

User sempat salah paham dan mengusulkan config yang di-stack (numpuk beberapa perubahan sekaligus dalam satu run, misal unfreeze+pooling+flow bareng). Ini SALAH untuk tujuan ablasi karena kalau beberapa hal diubah sekaligus, efeknya tidak bisa diatribusikan ke variabel mana yang menyebabkan perubahan hasil. Protokol yang benar dan FINAL:

**Prinsip: setiap run RESET ke default dulu, lalu ubah HANYA SATU field yang jadi fokus ablasi run tersebut.** Tidak di-stack/numpuk dari run sebelumnya.

**Default `train_cnn_lstm.ipynb`** (baseline yang sudah selesai jalan, `run_20260415_195341`):
```
unfreeze_blocks = "full"
motion_mode     = "diff"
```

**Default `train_mobilenet.ipynb`** (baseline yang sudah selesai jalan, `run_20260415_201349`):
```
unfreeze_blocks    = 4
pooling            = "last"
transformer_layers = 4
motion_mode        = "diff"
```

**6 run yang harus dijalankan** (satu-satu, berurutan, tidak paralel, supaya tidak rebutan GPU):

| # | Notebook | Field yang diubah dari default (HANYA ini) | Field lain | `run_tag` |
|---|----------|----------------------------------------------|------------|-----------|
| 1 | `train_cnn_lstm.ipynb` | `unfreeze_blocks=4` | `motion_mode="diff"` (tetap default) | `unfreeze-4` |
| 2 | `train_mobilenet.ipynb` | `unfreeze_blocks="full"` | `pooling="last"`, `motion_mode="diff"` (tetap default) | `unfreeze-full` |
| 3 | `train_mobilenet.ipynb` | `pooling="mean"` | `unfreeze_blocks=4`, `motion_mode="diff"` (tetap default) | `pool-mean` |
| 4 | `train_mobilenet.ipynb` | `pooling="attention"` | `unfreeze_blocks=4`, `motion_mode="diff"` (tetap default) | `pool-attention` |
| 5 | `train_cnn_lstm.ipynb` | `motion_mode="optical_flow"` | `unfreeze_blocks="full"` (tetap default) | `flow` |
| 6 | `train_mobilenet.ipynb` | `motion_mode="optical_flow"` | `unfreeze_blocks=4`, `pooling="last"` (tetap default) | `flow` |

Setelah 6 run ini + 2 baseline = 8 run total, cukup untuk semua 4 section perbandingan di `model_evaluation.ipynb` (baseline comparison, fairness-matched depth, pooling ablation, motion ablation).

**Run kombinasi (opsional, DI LUAR 6 run wajib di atas)**: kalau user mau lihat efek gabungan beberapa perubahan sekaligus (misal MobileNet dengan unfreeze full + pooling attention + optical flow bersamaan) sebagai eksplorasi "kombinasi terbaik", itu boleh ditambahkan sebagai run ke-7+ terpisah, TAPI itu bukan pengganti 6 run ablasi murni di atas — keduanya punya tujuan analisis yang berbeda (ablasi murni vs kombinasi terbaik) dan tidak saling menggantikan.

### State saat ini (per pengecekan terakhir)

`outputs/logs/cnn_lstm/`: hanya `run_20260415_195341` (baseline). `outputs/logs/mobile_net/`: hanya `run_20260415_201349` (baseline). Belum ada satu pun dari 6 run ablasi yang berhasil disimpan (user pernah run CNN_LSTM ablasi tapi foldernya dihapus manual oleh user sebelum sempat dianalisis). Semua 6 run di tabel atas masih perlu dijalankan dari awal.

# Panduan Menjalankan GMOCAT-Enhanced di Jupyter Notebook Lokal

Panduan ini akan membantu Anda menyiapkan lingkungan dan menjalankan eksperimen GMOCAT (Graph-Enhanced Multi-Objective Computerized Adaptive Testing) yang telah dimodifikasi langsung dari Jupyter Notebook di komputer lokal Anda (Windows/Mac/Linux).

## Prasyarat

Sebelum memulai, pastikan Anda memiliki:
1.  **Anaconda** atau **Miniconda** terinstal. (Direkomendasikan untuk manajemen environment).
2.  **Visual Studio Code** (dengan ekstensi Jupyter) atau **Jupyter Lab**.
3.  Akses ke terminal/command prompt.

---

## Langkah 1: Persiapan Environment

Kita akan membuat environment Python khusus agar tidak mengganggu instalasi lain.

1.  Buka terminal (Anaconda Prompt di Windows).
2.  Buat environment baru (misal bernama `gmocat_env`) dengan Python 3.12:
    ```bash
    conda create -n gmocat_env python=3.12
    conda activate gmocat_env
    ```

3.  Instal PyTorch.
    *   **Jika menggunakan GPU (NVIDIA):** Cek versi CUDA Anda, lalu instal yang sesuai (lihat [pytorch.org](https://pytorch.org/get-started/locally/)).
        Contoh (CUDA 12.1):
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ```
    *   **Jika menggunakan CPU saja:**
        ```bash
        pip install torch torchvision torchaudio
        ```

4.  Instal DGL (Deep Graph Library).
    *   Sangat penting untuk menginstal versi yang kompatibel. Disarankan menggunakan versi yang sesuai dengan versi PyTorch dan CUDA Anda.
    *   Contoh untuk **CPU**:
        ```bash
        pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/repo.html
        ```
    *   Contoh untuk **CUDA 12.1**:
        ```bash
        pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
        ```

5.  Instal dependensi lainnya:
    ```bash
    pip install torchdata scikit-learn pandas matplotlib pyyaml pydantic
    ```

---

## Langkah 2: Persiapan Kode

Pastikan Anda berada di dalam folder proyek `GMOCAT-modif`.

Struktur folder harus terlihat seperti ini:
```
GMOCAT-modif/
├── agents/
├── data/
├── envs/
├── function/
├── graph_data/
├── models/
├── launch_gcat.py
├── train_gcat.sh
└── ...
```

---

## Langkah 3: Membuat Jupyter Notebook

Buat file baru bernama `run_experiment.ipynb` di dalam folder `GMOCAT-modif/`.
Salin dan tempel kode berikut ke dalam sel-sel notebook tersebut.

### Sel 1: Import Library & Setup Path
Kode ini memastikan Python bisa menemukan modul-modul GMOCAT.

```python
import sys
import os
import torch

# Tambahkan direktori saat ini ke path agar modul bisa diimport
sys.path.append(os.getcwd())

# Cek device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Menggunakan device: {device}")
```

### Sel 2: Definisi Argumen Eksperimen
Alih-alih menggunakan command line arguments, kita mendefinisikan konfigurasi menggunakan class sederhana. Anda bisa mengubah parameter di sini.

```python
class Args:
    def __init__(self):
        # Konfigurasi Dasar
        self.seed = 145
        self.environment = "GCATEnv"
        self.data_path = "./data/"
        self.data_name = "dbekt22" # Dataset target
        self.agent = "GCATAgent"
        self.FA = "GCAT"
        self.CDM = "NCD" # Model Neural Cognitive Diagnosis

        # Pengaturan Testing
        self.T = 20 # Panjang tes (jumlah soal)
        self.ST = [5, 10, 20] # Step evaluasi
        self.student_ids = [0] # Dummy, akan di-reset oleh env
        self.target_concepts = [0] # Dummy

        # Hyperparameters Training
        self.gpu_no = "0"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' # Otomatis
        self.learning_rate = 0.001
        self.training_epoch = 2 # Jumlah epoch (naikkan untuk hasil lebih baik, misal 20-50)
        self.train_bs = 64      # Batch size training
        self.test_bs = 1024     # Batch size testing
        self.batch = 128

        # Parameter Model & RL
        self.cdm_lr = 0.02
        self.cdm_epoch = 2
        self.cdm_bs = 128
        self.gamma = 0.5
        self.latent_factor = 256
        self.n_block = 1
        self.graph_block = 1
        self.n_head = 1
        self.dropout_rate = 0.1
        self.policy_epoch = 1
        self.morl_weights = [1, 1, 1] # Bobot Accuracy, Diversity, Novelty
        self.emb_dim = 128
        self.use_graph = True
        self.use_attention = True
        self.store_action = False

    def __str__(self):
        return str(self.__dict__)

args = Args()
print("Konfigurasi dimuat.")
```

### Sel 3: Inisialisasi Environment & Model
Langkah ini memuat data graf, menginisialisasi environment, dan membangun model.

```python
from launch_gcat import construct_local_map
from util import get_objects, set_global_seeds
import envs as all_envs
import agents as all_agents
import function as all_FA

# Set seed
set_global_seeds(args.seed)
if args.device == 'cuda':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_no)

# Load Graph
print(f"Loading Graph Data untuk {args.data_name}...")
local_map = construct_local_map(args, path=f'graph_data/{args.data_name}/')

# Setup Environment
print("Setup Environment...")
envs = get_objects(all_envs)
env = envs[args.environment](args)

# Update args dengan info dari env
args.user_num = env.user_num
args.item_num = env.item_num
args.know_num = env.know_num

print(f"Environment siap. User: {args.user_num}, Item: {args.item_num}, Concept: {args.know_num}")

# Setup Model (Function Approximation)
print("Membangun Model GCAT...")
nets = get_objects(all_FA)
fa = nets[args.FA].create_model(args, local_map)

# Setup Agent
print("Inisialisasi Agen...")
agents = get_objects(all_agents)
agent = agents[args.agent](env, fa, args)
print("Sistem siap dilatih.")
```

### Sel 4: Jalankan Training
Eksekusi sel ini untuk memulai proses pelatihan. Output log akan muncul di bawah sel.

```python
print("Memulai Training...")
agent.train()
print("Training Selesai.")
```

### Sel 5: Visualisasi Hasil
Setelah training selesai, gunakan kode ini untuk melihat grafik coverage secara langsung.

```python
import matplotlib.pyplot as plt
from analyze_results import parse_log
import glob

# Cari file log terbaru
log_dir = f'baseline_log/{args.data_name}'
list_of_files = glob.glob(f'{log_dir}/*.txt')
latest_file = max(list_of_files, key=os.path.getctime)
print(f"Menganalisis log: {latest_file}")

# Parse dan Plot
data = parse_log(latest_file)

if data:
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(data) + 1), data, marker='o', label='Coverage')
    plt.title(f'Concept Coverage Curve ({args.data_name})')
    plt.xlabel('Step')
    plt.ylabel('Coverage Ratio')
    plt.grid(True)
    plt.legend()
    plt.show()
    print(f"Final Coverage: {data[-1]*100:.2f}%")
else:
    print("Tidak ada data coverage yang ditemukan di log.")
```

---

## Masalah Umum & Solusi

1.  **`ModuleNotFoundError: No module named 'torchdata.datapipes'`**
    *   Ini terjadi karena konflik versi `dgl` dan `torchdata`.
    *   **Solusi:** Pastikan Anda menggunakan versi yang kompatibel (lihat Langkah 1) atau jalankan kode di repositori ini yang sudah dimodifikasi (di-patch) untuk menghindari impor yang bermasalah.

2.  **`RuntimeError: CUDA out of memory`**
    *   Jika GPU Anda kehabisan memori.
    *   **Solusi:** Kurangi `args.train_bs` (batch size) di Sel 2 menjadi angka yang lebih kecil (misal 32 atau 16).

3.  **Training Terlalu Lama**
    *   Jika menjalankan di CPU, proses bisa lambat.
    *   **Solusi:** Kurangi `args.training_epoch` atau `args.train_bs` untuk eksperimen cepat.

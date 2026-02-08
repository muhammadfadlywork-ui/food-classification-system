# KAMI - Klasifikasi Makanan Khas Indonesia

Aplikasi web untuk klasifikasi makanan khas Indonesia secara otomatis menggunakan deep learning. Sistem ini memanfaatkan model **EfficientNetV2-S** yang telah dilatih untuk mengenali 30 jenis makanan khas dari berbagai daerah di Indonesia.

## Fitur

- **Klasifikasi Otomatis** - Upload gambar makanan dan dapatkan prediksi jenis makanan secara instan
- **30 Kelas Makanan** - Mendukung pengenalan berbagai makanan khas daerah Indonesia
- **Responsive UI** - Antarmuka web modern dengan Tailwind CSS
- **GPU Support** - Mendukung inferensi menggunakan CUDA GPU untuk performa lebih cepat

## Tech Stack

| Komponen | Teknologi |
|---|---|
| Backend | Python, Flask |
| Deep Learning | PyTorch, timm (EfficientNetV2-S) |
| Image Processing | Pillow, NumPy |
| Frontend | HTML, Tailwind CSS, JavaScript |

## Struktur Project

```
Klasifikasi Makanan/
├── app.py                 # Entry point aplikasi Flask
├── classifier.py          # Model deep learning & inferensi
├── file_handler.py        # Handler upload file
├── requirements.txt       # Dependensi Python
├── static/
│   ├── model/
│   │   ├── V2S_BestModel.pth   # Model terlatih (tidak termasuk di repo)
│   │   └── classes.json         # Mapping 30 kelas makanan
│   ├── uploads/                 # Direktori upload gambar
│   └── images/                  # Asset gambar statis
└── templates/
    ├── belajar.html       # Halaman utama / landing page
    ├── predict.html       # Halaman upload gambar
    └── result.html        # Halaman hasil klasifikasi
```

## Makanan yang Didukung

| No | Makanan | No | Makanan |
|---|---|---|---|
| 1 | Asinan Jakarta | 16 | Nagasari |
| 2 | Ayam Betutu | 17 | Nasi Goreng |
| 3 | Ayam Goreng Lengkuas | 18 | Papeda |
| 4 | Ayam Bumbu Rujak | 19 | Pempek |
| 5 | Bika Ambon | 20 | Rawon Surabaya |
| 6 | Bubur Manado | 21 | Rendang |
| 7 | Gado Gado | 22 | Rujak Cingur |
| 8 | Gudeg | 23 | Sate Ayam Madura |
| 9 | Gulai Ikan Mas | 24 | Sate Lilit |
| 10 | Kerak Telor | 25 | Sate Maranggi |
| 11 | Klappertaart | 26 | Serabi |
| 12 | Kolak | 27 | Soto Banjar |
| 13 | Kue Lumpur | 28 | Soto Lamongan |
| 14 | Laksa Bogor | 29 | Tahu Telur |
| 15 | Lumpia Semarang | 30 | Mie Aceh |

## Instalasi

### Prasyarat

- Python 3.8+
- pip
- (Opsional) NVIDIA GPU + CUDA untuk inferensi lebih cepat

### Langkah Instalasi

1. **Clone repository**
   ```bash
   git clone <repository-url>
   cd "Klasifikasi Makanan"
   ```

2. **Buat virtual environment**
   ```bash
   python -m venv venv
   ```

   Aktivasi:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

3. **Install dependensi**
   ```bash
   pip install -r requirements.txt
   ```

4. **Siapkan model**

   Letakkan file model terlatih `V2S_BestModel.pth` ke dalam folder `static/model/`.

## Menjalankan Aplikasi

```bash
python app.py
```

Aplikasi akan berjalan di `http://127.0.0.1:5000`.

## Cara Kerja

1. User mengupload gambar makanan melalui halaman `/predict`
2. Gambar di-resize dengan padding ke ukuran **380x380** (mempertahankan aspect ratio)
3. Gambar dinormalisasi menggunakan standar ImageNet
4. Model **EfficientNetV2-S** melakukan inferensi dan menghasilkan prediksi
5. Hasil klasifikasi ditampilkan di halaman `/result`

### Arsitektur Model

- **Base Model**: EfficientNetV2-S (dari library `timm`)
- **Custom Classifier Head**: Dropout(0.3) -> Linear(1024) -> SiLU -> Dropout(0.3) -> Linear(30 classes)
- **Input Size**: 380 x 380 piksel (RGB)
- **Normalisasi**: ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## Lisensi

&copy; Muhammad Fadly 2025

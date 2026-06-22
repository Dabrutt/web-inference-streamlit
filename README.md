# Trashification Streamlit

Trashification Streamlit adalah aplikasi berbasis web untuk melakukan deteksi dan klasifikasi sampah menggunakan model YOLO. Aplikasi ini dirancang agar pengguna dapat mengunggah gambar, mengambil foto langsung dari kamera, atau menjalankan deteksi secara realtime melalui webcam.

Setiap objek sampah yang terdeteksi akan ditampilkan dalam bentuk bounding box beserta nama kelas dan nilai confidence. Hasil deteksi kemudian dianalisis lebih lanjut untuk menampilkan jumlah sampah, jumlah label unik, rata-rata confidence, sifat sampah, kategori penanganan, dan bahan dasar sampah.

## Fitur Utama

- Impor foto dari perangkat.
- Ambil foto langsung menggunakan kamera.
- Deteksi objek sampah menggunakan model YOLO `best.pt`.
- Deteksi sampah secara realtime menggunakan WebRTC.
- Pengaturan nilai confidence threshold.
- Menampilkan bounding box, nama kelas, dan confidence.
- Menampilkan jumlah seluruh sampah yang terdeteksi.
- Menampilkan jumlah label unik.
- Menampilkan rata-rata confidence hasil deteksi.
- Pengelompokan sampah menjadi Organik dan Anorganik.
- Pengelompokan berdasarkan kategori Recyclable, Biodegradable, Hazardous, dan Residual.
- Pengelompokan berdasarkan bahan dasar, seperti Plastik, Logam, Kertas/Karton, Kaca, Kayu, dan Baterai.
- Menampilkan tabel detail seluruh hasil deteksi.
- Menampilkan penjelasan mengenai kelompok sampah yang digunakan.

## Kelas Sampah

Model yang digunakan memiliki 22 kelas:

1. `battery`
2. `can`
3. `cardboard_bowl`
4. `cardboard_box`
5. `chemical_plastic_bottle`
6. `chemical_plastic_gallon`
7. `chemical_spray_can`
8. `light_bulb`
9. `paint_bucket`
10. `plastic_bag`
11. `plastic_bottle`
12. `plastic_bottle_cap`
13. `plastic_box`
14. `plastic_cultery`
15. `plastic_cup`
16. `plastic_cup_lid`
17. `reuseable_paper`
18. `scrap_paper`
19. `scrap_plastic`
20. `snack_bag`
21. `stick`
22. `straw`

## 1. Struktur Folder

```text
trashification_streamlit/
├── app.py
├── requirements.txt
├── models/
│   └── best.pt
├── assets/
│   └── hero.jpg
└── .streamlit/
    └── config.toml
```

Keterangan:

- `app.py` merupakan kode utama aplikasi Streamlit.
- `requirements.txt` berisi dependensi yang dibutuhkan aplikasi.
- `models/best.pt` merupakan weight model YOLO hasil training.
- `assets/hero.jpg` merupakan gambar tampilan bagian header aplikasi.
- `.streamlit/config.toml` berisi konfigurasi tema dan pengaturan Streamlit.

## 2. Persiapan Model

Salin weight terbaik hasil training YOLO ke dalam folder:

```text
models/best.pt
```

Model harus menggunakan urutan 22 kelas yang sama dengan daftar kelas pada `app.py`.

Struktur folder model:

```text
models/
└── best.pt
```

## 3. Membuat Virtual Environment

Disarankan menggunakan Python 3.10.

### Windows PowerShell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Jika PowerShell menolak menjalankan script aktivasi, gunakan:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Linux/macOS

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 4. Menjalankan Aplikasi

Pastikan virtual environment sudah aktif, kemudian jalankan:

```bash
python -m streamlit run app.py
```

Aplikasi biasanya dapat diakses melalui:

```text
http://localhost:8501
```

Saat pertama kali dibuka, aplikasi menggunakan mode Impor Foto sehingga kamera tidak langsung aktif. Kamera hanya digunakan ketika pengguna memilih fitur Ambil Foto atau mengaktifkan Kamera Realtime.

## 5. Mengubah Pengelompokan Sampah

Pengelompokan sampah dapat disesuaikan melalui beberapa variabel pada `app.py`.

### Kategori Penanganan

Edit variabel:

```python
CATEGORY_CLASSES
```

Variabel tersebut digunakan untuk mengelompokkan sampah menjadi:

- Recyclable
- Biodegradable
- Hazardous
- Residual

### Sifat Sampah

Edit variabel yang mengatur kelompok:

- Organik
- Anorganik

### Bahan Dasar

Edit variabel pemetaan bahan dasar untuk mengelompokkan kelas ke dalam kategori seperti:

- Plastik
- Logam
- Kertas/Karton
- Kaca
- Kayu
- Baterai

Pastikan setiap nama kelas pada mapping sama dengan nama kelas yang tersimpan di dalam model YOLO.

## 6. Deployment

Aplikasi dapat di-deploy menggunakan Streamlit Community Cloud.

Sebelum deployment, pastikan repository memiliki file dan folder berikut:

```text
app.py
requirements.txt
models/best.pt
assets/hero.jpg
.streamlit/config.toml
```

Langkah umum deployment:

1. Upload project ke repository GitHub.
2. Masuk ke Streamlit Community Cloud.
3. Pilih repository yang berisi project.
4. Tentukan file utama aplikasi sebagai `app.py`.
5. Jalankan proses deployment.

File `requirements.txt` akan digunakan secara otomatis untuk menginstal dependensi aplikasi.

Kamera realtime pada deployment publik memerlukan koneksi HTTPS. Pada beberapa jaringan, koneksi WebRTC dapat memerlukan konfigurasi TURN server agar kamera dapat terhubung dengan baik.

## Teknologi yang Digunakan

- Python
- Streamlit
- Ultralytics YOLO
- OpenCV
- NumPy
- Pandas
- Pillow
- Streamlit WebRTC
- PyAV

## Catatan

- Nilai IoU digunakan secara internal oleh aplikasi dan tidak ditampilkan kepada pengguna.
- Pengguna hanya dapat mengatur confidence threshold.
- Kecepatan deteksi realtime dipengaruhi oleh perangkat, ukuran inferensi, model YOLO, dan interval frame.
- Mapping kategori sampah dapat disesuaikan dengan kebutuhan penelitian atau kebijakan pengelolaan sampah yang digunakan.

# Deskripsi 
Sistem Absensi AI adalah aplikasi berbasis web yang menggunakan teknologi Face Recognition (Pengenalan Wajah) untuk melakukan absensi siswa secara otomatis. Aplikasi ini memanfaatkan algoritma LBPH (Local Binary Pattern Histograms) dari OpenCV untuk mengenali wajah siswa dan mencatat kehadiran secara real-time.

# Teknologi yang Digunakan:
Backend: Flask (Python)
Database: SQLite dengan SQLAlchemy ORM
Face Recognition: OpenCV + LBPH Face Recognizer
Frontend: HTML, CSS, Bootstrap 5
Kamera: WebRTC (JavaScript) & OpenCV (Python)

# Python Dependencies:
Flask==2.3.3

Flask-SQLAlchemy==3.0.5

opencv-python==4.8.1.78

opencv-contrib-python==4.8.1.78

numpy==1.24.3

Pillow==10.0.0

pandas==2.0.3

openpyxl==3.1.2

# Fitur Utama
**1. Manajemen Data Siswa**
Tambah, edit, dan hapus data siswa
Informasi: NIS, Nama, Kelas

**2. Capture Wajah**
Ambil sampel wajah siswa melalui webcam browser
Penyimpanan otomatis ke folder dataset/{id_siswa}/
Minimal 10 foto per siswa untuk hasil optimal

**3. Training Model AI**
Latih model dengan data wajah yang sudah dikumpulkan
Status training real-time
Auto-restart proses absensi setelah training

**4. Absensi Otomatis**
Deteksi wajah real-time melalui kamera
Pencatatan kehadiran otomatis ke database
Anti-spam (30 detik per siswa)
Log aktivitas lengkap

**5. Laporan Absensi**
Filter berdasarkan tanggal
Export ke Excel/CSV
Hapus data per tanggal atau semua
Statistik kehadiran

**6. Dashboard Monitoring**
Total siswa, hadir hari ini, total absensi
Log aktivitas terbaru
Status mesin absensi (running/stop)

# Struktur Database 
**Tabel siswa**
id	INTEGER (PK)	ID otomatis
nis	VARCHAR(20)	Nomor Induk Siswa
nama	VARCHAR(100)	Nama lengkap
kelas	VARCHAR(50)	Kelas siswa

**Tabel absensi**
id	INTEGER (PK)	ID otomatis
siswa_id	INTEGER (FK)	Referensi ke siswa.id
tanggal	VARCHAR(20)	Tanggal absensi (YYYY-MM-DD)
jam	VARCHAR(20)	Jam absensi (HH:MM:SS)

**Tabel log_absensi**
id	INTEGER (PK)	ID otomatis
pesan	VARCHAR(255)	Pesan log
waktu	VARCHAR(30)	Waktu kejadian

# Struktur Project
absensi-ai/
├── app.py                 # Aplikasi utama Flask
├── train.py              # Training model AI
├── absensi_ai.py         # Proses absensi real-time
├── requirements.txt      # Dependencies
├── dataset/              # Data wajah siswa
│   ├── 1/               # Folder ID siswa
│   │   ├── face_1.jpg
│   │   └── ...
│   └── 2/
├── models/               # Model AI yang sudah dilatih
│   ├── trainer.yml
│   └── label_map.npy
├── instance/             # Database SQLite
│   └── absensi.db
└── templates/            # Template HTML
    ├── index.html
    ├── siswa.html
    ├── kamera.html
    └── laporan.html

# Panduan Penggunaan
**1. Menambahkan Data Siswa**
Langkah-langkah:
Buka menu Data Siswa
Isi form tambah siswa:
NIS: Nomor Induk Siswa
Nama: Nama lengkap siswa
Kelas: Kelas siswa (contoh: X-RPL-1)
Klik Simpan

**2.Mengambil Sampel Wajah**
Syarat:
Minimal 10-20 foto per siswa
Pencahayaan yang cukup
Ekspresi wajah natural
Variasi sudut (depan, sedikit miring kiri/kanan)
Langkah-langkah:
Di halaman Data Siswa, klik tombol Ambil Wajah pada siswa yang dituju
Izinkan akses kamera browser
Posisikan wajah di tengah frame
Klik Capture Wajah (ulangi 10-20 kali)
Perhatikan jumlah foto yang tersimpan
Klik Kembali jika selesai

**3. Melatih Model AI**
Langkah-langkah:
Buka menu Data Siswa
Pastikan setiap siswa memiliki minimal 10 foto
Klik tombol Latih Semua Model
Tunggu proses training selesai (beberapa detik)
Status model akan berubah menjadi "Sudah dilatih"
Info:
Training hanya untuk siswa dengan foto ≥ 10
Model tersimpan di folder models/
Proses absensi akan otomatis restart dengan model baru

**4. Menjalankan Absensi**
Langkah-langkah:
Kembali ke Dashboard
Klik tombol Mulai Absensi AI
Jendela OpenCV akan terbuka dengan tampilan kamera
Siswa yang terdeteksi akan otomatis tercatat hadir
Tekan ESC atau klik Stop Absensi AI untuk berhenti
Aturan Absensi:
Wajah harus terdaftar dalam model
Satu siswa hanya bisa absen setiap 30 detik
Absensi dicatat dengan tanggal dan jam saat itu

**5. Melihat Laporan**
Langkah-langkah:
Buka menu Laporan Absensi
Gunakan filter tanggal untuk melihat data spesifik
Klik Export Excel untuk download laporan
Gunakan tombol Hapus untuk menghapus data tertentu
Fitur Laporan:
Filter berdasarkan tanggal
Export ke Excel (semua data atau per tanggal)
Hapus per hari atau semua data
Statistik total absensi

**6. Monitoring Dashboard**
Informasi yang ditampilkan:
Status Mesin Absensi: Running / Tidak Berjalan
Total Siswa: Jumlah siswa terdaftar
Hadir Hari Ini: Jumlah kehadiran hari ini
Total Absensi: Seluruh data absensi
Log Aktivitas: 10 kejadian terakhir
Fitur Log:
Show/Hide log
Hapus per tanggal
Hapus semua log
Auto-refresh setiap 10 detik

# DOKUMENTASI 
Halaman Index
<img width="1902" height="970" alt="image" src="https://github.com/user-attachments/assets/05844d7d-1314-451d-85fc-87f4a2bb03c6" />

Manajemen Siswa
<img width="1912" height="972" alt="image" src="https://github.com/user-attachments/assets/814cf9ab-9398-4918-abc0-4b156e357891" />

Rekam Foto 
<img width="1901" height="982" alt="image" src="https://github.com/user-attachments/assets/f32be312-4acf-4bab-a3d9-fad71b18c530" />

Laporan 
<img width="1906" height="977" alt="image" src="https://github.com/user-attachments/assets/8a3a6143-4082-44ac-851e-061b272a7f33" />

Absensi
<img width="1917" height="982" alt="image" src="https://github.com/user-attachments/assets/999b2da7-78c1-4b9b-934e-ad4ef97d4414" />





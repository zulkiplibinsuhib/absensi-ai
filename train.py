import cv2
import os
import numpy as np
import sqlite3
import sys
import traceback

def main():
    try:
        dataset_path = "dataset"

        # Cek folder dataset
        if not os.path.exists(dataset_path):
            print("[ERROR] Folder dataset tidak ditemukan!")
            return False

        # Hitung total file gambar
        total_files = 0
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    total_files += 1

        print(f"[INFO] Ditemukan {total_files} file gambar di folder dataset")

        if total_files == 0:
            print("[ERROR] Tidak ada file gambar! Silakan capture wajah terlebih dahulu.")
            return False

        faces = []
        labels = []
        label_map = {}

        # Load face detector
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        print(f"[INFO] Loading cascade from: {cascade_path}")
        
        detector = cv2.CascadeClassifier(cascade_path)

        if detector.empty():
            print("[ERROR] Gagal load face detector!")
            return False

        current_id = 0

        # Buat koneksi database
        db_path = 'instance/absensi.db'
        print(f"[INFO] Database path: {db_path}")
        
        if not os.path.exists(db_path):
            print(f"[ERROR] Database tidak ditemukan di {db_path}")
            return False
            
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Loop semua folder di dataset
        for folder_name in os.listdir(dataset_path):
            folder = os.path.join(dataset_path, folder_name)

            if not os.path.isdir(folder):
                continue

            print(f"\n[PROCESS] Memproses folder: {folder_name}")

            # Cek apakah folder berisi file gambar
            gambar_files = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
            gambar_count = len(gambar_files)
            
            if gambar_count == 0:
                print(f"[WARNING] Folder {folder_name} kosong, skip...")
                continue

            # Folder name harus berupa ID siswa (angka)
            try:
                siswa_id = int(folder_name)
                
                # Ambil nama siswa dari database berdasarkan ID
                cursor.execute("SELECT nama FROM siswa WHERE id = ?", (siswa_id,))
                result = cursor.fetchone()
                
                if result:
                    nama_siswa = result[0]
                    print(f"[TRAINING] {nama_siswa} (ID: {siswa_id}) - {gambar_count} gambar")
                else:
                    print(f"[WARNING] ID {siswa_id} tidak ditemukan di database, skip...")
                    continue
                    
            except ValueError:
                print(f"[WARNING] Folder {folder_name} bukan ID numerik, skip...")
                continue

            # Simpan ke label_map
            label_map[current_id] = nama_siswa
            
            wajah_terdeteksi = 0
            
            # Proses setiap gambar
            for file in gambar_files:
                path = os.path.join(folder, file)
                img = cv2.imread(path)

                if img is None:
                    print(f"  [ERROR] Gagal baca: {file}")
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Deteksi wajah
                wajah = detector.detectMultiScale(
                    gray,
                    scaleFactor=1.05,
                    minNeighbors=3,
                    minSize=(60, 60)
                )

                for (x, y, w, h) in wajah:
                    face = gray[y:y+h, x:x+w]
                    face_resized = cv2.resize(face, (200, 200))
                    faces.append(face_resized)
                    labels.append(current_id)
                    wajah_terdeteksi += 1

            print(f"  [OK] Wajah terdeteksi: {wajah_terdeteksi} dari {gambar_count} gambar")

            if wajah_terdeteksi > 0:
                current_id += 1
            else:
                print(f"  [WARNING] Tidak ada wajah terdeteksi untuk {nama_siswa}")

        conn.close()

        print(f"\n{'='*50}")
        print(f"HASIL TRAINING:")
        print(f"  Total wajah terdeteksi: {len(faces)}")
        print(f"  Total siswa: {current_id}")
        print(f"{'='*50}")

        if len(faces) < 2:
            print("\n[ERROR] Data training terlalu sedikit!")
            print("\nSOLUSI:")
            print("   1. Capture minimal 5-10 foto wajah untuk setiap siswa")
            print("   2. Pastikan wajah terlihat jelas di setiap foto")
            return False

        print("\n[TRAINING] Melatih model...")
        model = cv2.face.LBPHFaceRecognizer_create()

        model.train(faces, np.array(labels))
        print("[OK] Training berhasil!")

        os.makedirs("models", exist_ok=True)

        model.save("models/trainer.yml")
        np.save("models/label_map.npy", label_map)

        print("\n[OK] Model tersimpan di models/trainer.yml")
        print(f"Label map: {label_map}")

        # Test verifikasi
        print("\n[INFO] Verifikasi model:")
        for label_id, nama in label_map.items():
            print(f"   ID {label_id} -> {nama}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Detail: {e}")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    # Set encoding untuk Windows
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    success = main()
    sys.exit(0 if success else 1)
import cv2
import numpy as np
import sqlite3
import os
from datetime import datetime
import time

# =========================
# LOAD MODEL
# =========================

try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("models/trainer.yml")
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit()

try:
    label_map = np.load(
        "models/label_map.npy",
        allow_pickle=True
    ).item()
    print("✅ Label map loaded successfully")
except Exception as e:
    print(f"❌ Failed to load label map: {e}")
    exit()

# =========================
# FACE DETECTOR
# =========================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if face_detector.empty():
    print("❌ Failed to load face cascade classifier")
    exit()

# =========================
# DATABASE
# =========================

DB_PATH = "instance/absensi.db"

if not os.path.exists(DB_PATH):
    print(f"❌ Database tidak ditemukan: {DB_PATH}")
    exit()
else:
    print(f"✅ Database found: {DB_PATH}")

# =========================
# CAMERA
# =========================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ Failed to open camera")
    exit()
else:
    print("✅ Camera opened successfully")
    # Disable OpenCL warnings
    cv2.ocl.setUseOpenCL(False)

# mencegah spam absensi
last_attendance = {}

print("\n🎥 Starting attendance system...")
print("Press ESC to quit\n")

while True:

    ret, frame = camera.read()

    if not ret:
        print("❌ Failed to capture frame")
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:

        face = gray[
            y:y+h,
            x:x+w
        ]

        try:

            label, confidence = recognizer.predict(face)

            # Confidence threshold (lower is better for LBPH)
            # 0 = perfect match, >80 = poor match
            if confidence > 80:

                nama = "UNKNOWN"
                warna = (0, 0, 255)  # Red for unknown

            else:

                nama = label_map.get(
                    label,
                    "UNKNOWN"
                )

                warna = (0, 255, 0)  # Green for recognized

                now = datetime.now()

                tanggal = now.strftime(
                    "%Y-%m-%d"
                )

                jam = now.strftime(
                    "%H:%M:%S"
                )

                # anti spam 30 detik
                if (
                    nama not in last_attendance
                    or
                    (
                        now -
                        last_attendance[nama]
                    ).total_seconds() > 30
                ):

                    conn = sqlite3.connect(
                        DB_PATH
                    )

                    cursor = conn.cursor()

                    cursor.execute("""
                        SELECT id
                        FROM siswa
                        WHERE UPPER(nama)=?
                    """, (
                        nama.upper(),
                    ))

                    siswa = cursor.fetchone()

                    if siswa:

                        siswa_id = siswa[0]

                        cursor.execute("""
                            SELECT id
                            FROM absensi
                            WHERE siswa_id=?
                            AND tanggal=?
                        """, (
                            siswa_id,
                            tanggal
                        ))

                        sudah_absen = (
                            cursor.fetchone()
                        )

                        if not sudah_absen:

                            cursor.execute("""
                                INSERT INTO absensi
                                (
                                    siswa_id,
                                    tanggal,
                                    jam
                                )
                                VALUES
                                (
                                    ?,
                                    ?,
                                    ?
                                )
                            """, (
                                siswa_id,
                                tanggal,
                                jam
                            ))

                            cursor.execute("""
                                INSERT INTO log_absensi
                                (
                                    pesan,
                                    waktu
                                )
                                VALUES
                                (
                                    ?,
                                    ?
                                )
                            """, (
                                f"✅ {nama} hadir",
                                datetime.now().strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                )
                            ))

                            conn.commit()

                            print(f"✅ {nama} HADIR at {jam}")

                        else:

                            cursor.execute("""
                                INSERT INTO log_absensi
                                (
                                    pesan,
                                    waktu
                                )
                                VALUES
                                (
                                    ?,
                                    ?
                                )
                            """, (
                                f"ℹ️ {nama} sudah absen",
                                datetime.now().strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                )
                            ))

                            conn.commit()

                            print(f"ℹ️ {nama} SUDAH ABSEN today")

                    conn.close()

                    last_attendance[nama] = now

            # Draw rectangle and label
            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                warna,
                2
            )

            # Display confidence as percentage of match quality
            match_quality = max(0, min(100, 100 - (confidence / 2)))
            cv2.putText(
                frame,
                f"{nama} ({match_quality:.0f}%)",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                warna,
                2
            )

        except Exception as e:
            print(f"Error processing face: {e}")

    # Display FPS counter
    cv2.putText(
        frame,
        f"Faces detected: {len(faces)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Sistem Absensi AI - Press ESC to quit",
        frame
    )

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        print("\n👋 Exiting...")
        break
    elif key == ord('s'):  # 's' key to show stats
        print(f"\n📊 Active recognitions: {list(last_attendance.keys())}")

# Cleanup
camera.release()
cv2.destroyAllWindows()
print("✅ System shutdown complete")
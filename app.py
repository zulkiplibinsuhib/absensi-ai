from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import subprocess
from flask import jsonify, session
import threading
import os
import shutil
import sys 


app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///absensi.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

attendance_process = None

# ==================================================
# MODEL
# ==================================================

class Siswa(db.Model):

    __tablename__ = 'siswa'

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    nis = db.Column(
        db.String(20)
    )

    nama = db.Column(
        db.String(100)
    )

    kelas = db.Column(
        db.String(50)
    )


class Absensi(db.Model):

    __tablename__ = 'absensi'

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    siswa_id = db.Column(
        db.Integer
    )

    tanggal = db.Column(
        db.String(20)
    )

    jam = db.Column(
        db.String(20)
    )

class LogAbsensi(db.Model):

    __tablename__ = 'log_absensi'

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    pesan = db.Column(
        db.String(255)
    )

    waktu = db.Column(
        db.String(30)
    )
with app.app_context():
    db.create_all()

# ==================================================
# DASHBOARD
# ==================================================
from flask import render_template, request, jsonify
import cv2
import numpy as np
import base64
import os
from datetime import datetime
import sqlite3

# Route untuk halaman kamera
@app.route('/kamera/<int:siswa_id>')
def kamera(siswa_id):
    conn = sqlite3.connect('instance/absensi.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, nama FROM siswa WHERE id = ?", (siswa_id,))
    siswa = cursor.fetchone()
    conn.close()
    
    if siswa is None:
        return "Siswa tidak ditemukan", 404
    
    siswa_data = {
        'id': siswa[0],
        'nama': siswa[1]
    }
    
    return render_template('kamera.html', siswa=siswa_data)
# Route untuk menyimpan wajah
@app.route('/simpan_wajah', methods=['POST'])
def simpan_wajah():
    try:
        data = request.get_json()
        siswa_id = data.get('siswa_id')
        image_data = data.get('image')
        
        # Hapus header base64
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 ke binary
        image_binary = base64.b64decode(image_data)
        
        # Konversi ke numpy array
        nparr = np.frombuffer(image_binary, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Konversi ke grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Load face detector
        face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        
        # Deteksi wajah
        faces = face_detector.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))
        
        if len(faces) == 0:
            return jsonify({
                'success': False,
                'message': 'Tidak ada wajah terdeteksi. Pastikan wajah terlihat jelas.'
            }), 400
        
        # Buat direktori
        face_dir = f'dataset/{siswa_id}'
        if not os.path.exists(face_dir):
            os.makedirs(face_dir)
        
        # Hitung jumlah foto
        existing_photos = len([f for f in os.listdir(face_dir) if f.endswith('.jpg')])
        
        # Simpan semua wajah yang terdeteksi
        saved = 0
        for i, (x, y, w, h) in enumerate(faces):
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (200, 200))
            filename = f'{face_dir}/face_{existing_photos + i + 1}.jpg'
            cv2.imwrite(filename, face_resized)
            saved += 1
        
        new_count = existing_photos + saved
        
        return jsonify({
            'success': True,
            'message': f'{saved} foto berhasil disimpan',
            'total': new_count
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500
# Route untuk melatih model
@app.route('/train_model', methods=['POST'])
def train_model():
    try:
        # Cek apakah file train.py ada
        if not os.path.exists('train.py'):
            return jsonify({
                'success': False,
                'message': 'File train.py tidak ditemukan!'
            }), 400
        
        # Jalankan train.py sebagai subprocess (lebih aman)
        python_path = sys.executable
        result = subprocess.run(
            [python_path, 'train.py'],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            # Restart absensi jika berjalan
            restart_absensi_if_running()
            
            return jsonify({
                'success': True,
                'message': 'Model berhasil dilatih!',
                'output': result.stdout
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Training gagal!\n{result.stderr}',
                'error': result.stderr
            }), 400
            
    except subprocess.TimeoutExpired:
        return jsonify({
            'success': False,
            'message': 'Training timeout! Proses terlalu lama.'
        }), 400
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

def restart_absensi_if_running():
    global attendance_process
    if attendance_process and attendance_process.poll() is None:
        attendance_process.terminate()
        import time
        time.sleep(1)
        attendance_process = subprocess.Popen([sys.executable, 'absensi_ai.py'])
        print("✅ Proses absensi direstart")
@app.route('/reset_model', methods=['POST'])
def reset_model():
    """Menghapus semua file model dan mereset status"""
    global attendance_process
    
    try:
        # Matikan proses absensi jika berjalan
        if attendance_process and attendance_process.poll() is None:
            attendance_process.terminate()
            attendance_process = None
        
        # Hapus folder models
        models_path = 'models'
        if os.path.exists(models_path):
            # Hapus semua file di folder models
            for file in os.listdir(models_path):
                file_path = os.path.join(models_path, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Error hapus {file_path}: {e}")
            
            print(f"✅ Folder {models_path} telah dibersihkan")
        
        # Buat ulang folder models kosong
        os.makedirs(models_path, exist_ok=True)
        
        # Optional: Buat file dummy agar folder tidak kosong
        with open(os.path.join(models_path, '.gitkeep'), 'w') as f:
            pass
        
        return jsonify({
            'success': True,
            'message': 'Model berhasil direset. Silakan latih ulang model dengan data terbaru.'
        })
        
    except Exception as e:
        print(f"Error reset model: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/training_status')
def training_status():
    global training_process
    if training_process and training_process.is_alive():
        return jsonify({
            'status': 'running',
            'message': 'Training sedang berjalan...'
        })
    else:
        return jsonify({
            'status': 'idle',
            'message': 'Tidak ada training berjalan'
        })

def restart_absensi_if_running():
    global attendance_process
    if attendance_process and attendance_process.poll() is None:
        attendance_process.terminate()
        import time
        time.sleep(1)
        attendance_process = subprocess.Popen(['python', 'absensi_ai.py'])
        print("✅ Proses absensi direstart")

@app.route('/get_trained_students')
def get_trained_students():
    """Mendapatkan daftar siswa yang sudah ada di model"""
    try:
        import numpy as np
        
        # Cek apakah file model ada
        if not os.path.exists('models/label_map.npy'):
            return jsonify({'trained': []})
        
        label_map = np.load('models/label_map.npy', allow_pickle=True).item()
        
        # Jika label_map kosong
        if not label_map:
            return jsonify({'trained': []})
        
        trained_students = []
        
        conn = sqlite3.connect('instance/absensi.db')
        cursor = conn.cursor()
        
        for nama in label_map.values():
            cursor.execute("SELECT id FROM siswa WHERE nama = ?", (nama,))
            result = cursor.fetchone()
            if result:
                trained_students.append(str(result[0]))
        
        conn.close()
        return jsonify({'trained': trained_students})
        
    except Exception as e:
        print(f"Error get trained students: {e}")
        return jsonify({'trained': []})


@app.route('/get_absensi_by_date')
def get_absensi_by_date():
    tanggal = request.args.get('tanggal')
    
    data = db.session.execute(
        db.text("""
            SELECT
                a.id,
                s.nama,
                a.tanggal,
                a.jam
            FROM absensi a
            LEFT JOIN siswa s ON s.id = a.siswa_id
            WHERE a.tanggal = :tanggal
            ORDER BY a.id DESC
        """), {'tanggal': tanggal}
    ).fetchall()
    
    return jsonify({
        'success': True,
        'total': len(data),
        'data': [[row[0], row[1], row[2], row[3]] for row in data]
    })

@app.route('/hapus_log/<int:log_id>', methods=['DELETE'])
def hapus_log(log_id):
    try:
        log = LogAbsensi.query.get_or_404(log_id)
        db.session.delete(log)
        db.session.commit()
        return jsonify({'success': True, 'message': 'Log berhasil dihapus'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# Hapus semua log
@app.route('/hapus_semua_log', methods=['DELETE'])
def hapus_semua_log():
    try:
        deleted_count = LogAbsensi.query.delete()
        db.session.commit()
        return jsonify({
            'success': True, 
            'message': f'{deleted_count} log berhasil dihapus',
            'deleted_count': deleted_count
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# Hapus log berdasarkan tanggal (opsional - fitur tambahan)
@app.route('/hapus_log_per_tanggal', methods=['POST'])
def hapus_log_per_tanggal():
    try:
        tanggal = request.json.get('tanggal')
        if not tanggal:
            return jsonify({'success': False, 'message': 'Tanggal tidak ditemukan'}), 400
        
        deleted_count = LogAbsensi.query.filter(
            LogAbsensi.waktu.like(f'{tanggal}%')
        ).delete()
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'{deleted_count} log tanggal {tanggal} berhasil dihapus'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500
        
@app.route('/get_all_absensi')
def get_all_absensi():
    data = db.session.execute(
        db.text("""
            SELECT
                a.id,
                s.nama,
                a.tanggal,
                a.jam
            FROM absensi a
            LEFT JOIN siswa s ON s.id = a.siswa_id
            ORDER BY a.id DESC
        """)
    ).fetchall()
    
    return jsonify({
        'success': True,
        'total': len(data),
        'data': [[row[0], row[1], row[2], row[3]] for row in data]
    })

@app.route('/absensi/hapus_semua', methods=['POST'])
def hapus_semua_absensi():
    try:
        Absensi.query.delete()
        db.session.commit()
        return jsonify({'success': True, 'message': 'Semua data absensi dihapus'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/absensi/hapus_per_hari', methods=['POST'])
def hapus_per_hari():
    try:
        tanggal = request.args.get('tanggal')
        deleted = Absensi.query.filter_by(tanggal=tanggal).delete()
        db.session.commit()
        return jsonify({'success': True, 'deleted': deleted, 'message': f'Menghapus {deleted} data'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/export_excel')
def export_excel():
    import pandas as pd
    from io import BytesIO
    from flask import send_file
    
    tanggal = request.args.get('tanggal')
    
    if tanggal:
        data = db.session.execute(
            db.text("""
                SELECT s.nama, a.tanggal, a.jam
                FROM absensi a
                LEFT JOIN siswa s ON s.id = a.siswa_id
                WHERE a.tanggal = :tanggal
                ORDER BY a.jam DESC
            """), {'tanggal': tanggal}
        ).fetchall()
        filename = f'laporan_absensi_{tanggal}.xlsx'
    else:
        data = db.session.execute(
            db.text("""
                SELECT s.nama, a.tanggal, a.jam
                FROM absensi a
                LEFT JOIN siswa s ON s.id = a.siswa_id
                ORDER BY a.tanggal DESC, a.jam DESC
            """)
        ).fetchall()
        filename = 'laporan_absensi_semua.xlsx'
    
    df = pd.DataFrame(data, columns=['Nama Siswa', 'Tanggal', 'Jam'])
    
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Laporan Absensi', index=False)
    
    output.seek(0)
    return send_file(output, download_name=filename, as_attachment=True)

    
# Route untuk menampilkan daftar siswa (opsional, untuk memilih siswa)
@app.route('/daftar_siswa')
def daftar_siswa():
    conn = sqlite3.connect('instance/absensi.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, nama FROM siswa ORDER BY nama")
    siswa_list = cursor.fetchall()
    conn.close()
    
    return render_template('daftar_siswa.html', siswa_list=siswa_list)
@app.route('/get_latest_logs')
def get_latest_logs():
    logs = LogAbsensi.query\
        .order_by(LogAbsensi.id.desc())\
        .limit(10)\
        .all()
    
    logs_data = []
    for log in logs:
        logs_data.append({
            'id': log.id,
            'pesan': log.pesan,
            'waktu': log.waktu
        })
    
    return jsonify({'logs': logs_data})
@app.route('/get_photo_count/<int:siswa_id>')
def get_photo_count(siswa_id):
    # Gunakan ID sebagai folder name (bukan nama)
    face_dir = f'dataset/{siswa_id}'  # Ini sudah benar
    count = 0
    if os.path.exists(face_dir):
        count = len([f for f in os.listdir(face_dir) if f.endswith('.jpg')])
    print(f"📊 Siswa ID {siswa_id} memiliki {count} foto")  # Debug
    return jsonify({'count': count})
@app.route('/siswa/hapus/<int:id>')
def hapus_siswa(id):
    try:
        # Hapus dari database
        siswa = Siswa.query.get_or_404(id)
        nama = siswa.nama
        db.session.delete(siswa)
        db.session.commit()
        
        # Hapus folder dataset siswa
        import shutil
        folder_dataset = f'dataset/{id}'
        if os.path.exists(folder_dataset):
            shutil.rmtree(folder_dataset)
            print(f"✅ Folder {folder_dataset} dihapus")
        
        return jsonify({'success': True, 'message': f'Siswa {nama} berhasil dihapus'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})
@app.route('/')
def index():
    global attendance_process

    jumlah_siswa = Siswa.query.count()
    total_absensi = Absensi.query.count()
    hari_ini = datetime.now().strftime("%Y-%m-%d")
    hadir_hari_ini = Absensi.query.filter_by(tanggal=hari_ini).count()

    # Ambil 10 log terakhir untuk ditampilkan
    logs = LogAbsensi.query.order_by(LogAbsensi.id.desc()).limit(10).all()
    
    # Ambil semua log untuk filter tanggal (unique)
    all_logs = LogAbsensi.query.order_by(LogAbsensi.id.desc()).all()
    
    # Buat list tanggal unik di Python
    unique_dates = list(set([log.waktu[:10] for log in all_logs]))
    unique_dates.sort(reverse=True)  # Urutkan dari terbaru

    status_absensi = False
    if attendance_process:
        if attendance_process.poll() is None:
            status_absensi = True

    return render_template(
        'index.html',
        jumlah_siswa=jumlah_siswa,
        total_absensi=total_absensi,
        hadir_hari_ini=hadir_hari_ini,
        status_absensi=status_absensi,
        logs=logs,
        unique_dates=unique_dates  # Kirim list tanggal unik
    )
# ==================================================
# SISWA
# ==================================================

@app.route('/siswa')
def siswa():

    data = Siswa.query.all()

    return render_template(
        'siswa.html',
        siswa=data
    )


@app.route(
    '/siswa/tambah',
    methods=['POST']
)
def tambah_siswa():

    siswa = Siswa(

        nis=request.form['nis'],
        nama=request.form['nama'],
        kelas=request.form['kelas']

    )

    db.session.add(
        siswa
    )

    db.session.commit()

    return redirect(
        '/siswa'
    )

# ==================================================
# MULAI ABSENSI
# ==================================================

@app.route('/mulai_absensi')
def mulai_absensi():

    global attendance_process

    if (
        attendance_process is None
        or
        attendance_process.poll() is not None
    ):

        attendance_process = subprocess.Popen(
            ['python', 'absensi_ai.py']
        )

    return redirect('/')

# ==================================================
# STOP ABSENSI
# ==================================================

@app.route('/stop_absensi')
def stop_absensi():

    global attendance_process

    if attendance_process:

        attendance_process.terminate()

        attendance_process = None

    return redirect('/')

# ==================================================
# LAPORAN
# ==================================================

@app.route('/laporan')
def laporan():
    # Data untuk tabel (semua data)
    data = db.session.execute(
        db.text("""
            SELECT
                a.id,
                s.nama,
                a.tanggal,
                a.jam
            FROM absensi a
            LEFT JOIN siswa s
            ON s.id = a.siswa_id
            ORDER BY a.id DESC
        """)
    ).fetchall()  # Tambahkan .fetchall() agar bisa diiterasi

    total_absensi = Absensi.query.count()
    
    # Hitung total absensi hari ini
    hari_ini = datetime.now().strftime("%Y-%m-%d")
    total_hari_ini = Absensi.query.filter_by(tanggal=hari_ini).count()

    return render_template(
        'laporan.html',
        data=data,
        total_absensi=total_absensi,
        total_hari_ini=total_hari_ini,
        today=hari_ini  # Kirim tanggal hari ini untuk filter default
    )
# ==================================================
# HAPUS ABSENSI
# ==================================================

@app.route(
    '/absensi/hapus/<int:id>'
)
def hapus_absensi(id):

    data = Absensi.query.get_or_404(
        id
    )

    db.session.delete(
        data
    )

    db.session.commit()

    return redirect(
        '/laporan'
    )

# ==================================================
# HAPUS SEMUA ABSENSI
# ==================================================

@app.route(
    '/absensi/hapus_semua'
)
def hapus_semua():

    Absensi.query.delete()

    db.session.commit()

    return redirect(
        '/laporan'
    )

# ==================================================
# RUN
# ==================================================

if __name__ == '__main__':

    app.run(
        debug=True
    )
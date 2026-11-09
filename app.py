from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import os
import base64
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20MB

EXCEL_PATH = "dataset/dataset_buah_sayur_knn.xlsx"
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===============================
# LOAD DATASET FUNCTION
# ===============================
def load_dataset():
    if os.path.exists(EXCEL_PATH):
        df = pd.read_excel(EXCEL_PATH)
        df.columns = df.columns.str.strip().str.lower()
        if "rekomendasi bahan kimia" in df.columns:
            df = df.rename(columns={"rekomendasi bahan kimia": "rekomendasi"})

        # Pastikan kolom ada
        for col in ["jenis", "nama", "ph", "rekomendasi"]:
            if col not in df.columns:
                df[col] = ""

        # Ubah tipe data
        df["jenis"] = df["jenis"].astype(str)
        df["nama"] = df["nama"].astype(str)
        df["rekomendasi"] = df["rekomendasi"].astype(str)
        try:
            df["ph"] = pd.to_numeric(df["ph"], errors="coerce").fillna(0)
        except:
            df["ph"] = 0

        df = df.dropna(subset=["jenis", "nama", "rekomendasi"])
        return df
    else:
        # Kalau Excel belum ada, buat dataframe kosong
        return pd.DataFrame(columns=["jenis","nama","ph","rekomendasi"])

# ===============================
# INISIALISASI DATASET & MODEL
# ===============================
df = load_dataset()
data = df.to_dict(orient="records")

# Label encoding
le_jenis = LabelEncoder()
le_rekom = LabelEncoder()
if not df.empty:
    df["jenis_enc"] = le_jenis.fit_transform(df["jenis"])
    df["rekom_enc"] = le_rekom.fit_transform(df["rekomendasi"])
    X = df[["jenis_enc","ph"]]
    y = df["rekom_enc"]
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)
else:
    knn = None

# ===============================
# ROUTES
# ===============================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/kamera")
def kamera():
    return render_template("kamera.html")

@app.route("/api/data")
def api_data():
    safe_data = []
    for d in load_dataset().to_dict(orient="records"):
        safe_data.append({
            "jenis": d["jenis"],
            "nama": d["nama"],
            "ph": float(d["ph"]),
            "rekomendasi": d["rekomendasi"]
        })
    return jsonify(safe_data)

# ===============================
# PREDIKSI MANUAL
# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        jenis = request.form.get("jenis", "").lower()
        nama = request.form.get("nama", "Item")
        ph = float(request.form.get("ph", 5.5))
    except:
        return render_template("index.html", hasil="pH tidak valid")

    try:
        if knn is not None:
            jenis_enc = le_jenis.transform([jenis])[0]
            pred = knn.predict([[jenis_enc, ph]])
            hasil = le_rekom.inverse_transform(pred)[0]
        else:
            hasil = "Gunakan Pembersih Standar"
    except:
        hasil = "Gunakan Pembersih Standar"

    # Simpan ke Excel
    try:
        new_row = pd.DataFrame([{
            "jenis": jenis,
            "nama": nama,
            "ph": ph,
            "rekomendasi": hasil
        }])
        if os.path.exists(EXCEL_PATH):
            df_existing = pd.read_excel(EXCEL_PATH)
            df_updated = pd.concat([df_existing, new_row], ignore_index=True)
        else:
            df_updated = new_row
        df_updated.to_excel(EXCEL_PATH, index=False)
    except Exception as e:
        print("Gagal menyimpan ke Excel:", e)

    return render_template("index.html",
                           jenis=jenis,
                           nama=nama,
                           ph=ph,
                           hasil=hasil)

# ===============================
# PREDIKSI DARI KAMERA
# ===============================
@app.route("/predict_image_ajax", methods=["POST"])
def predict_image_ajax():
    image_data = request.form.get("image")
    if not image_data:
        return jsonify({"error":"Tidak ada gambar"}), 400

    # decode base64
    try:
        header, encoded = image_data.split(",", 1)
        data_img = base64.b64decode(encoded)
    except:
        return jsonify({"error":"Format gambar salah"}), 400

    filename = "capture.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    with open(filepath, "wb") as f:
        f.write(data_img)

    jenis = request.form.get("jenis", "buah").lower()
    nama = request.form.get("nama", "Capture")
    ph = 5.5
    hasil = "Gunakan Pembersih Standar"

    try:
        if knn is not None:
            jenis_enc = le_jenis.transform([jenis])[0]
            pred = knn.predict([[jenis_enc, ph]])
            hasil = le_rekom.inverse_transform(pred)[0]
    except Exception as e:
        print("Error prediksi:", e)

    # Simpan ke Excel
    try:
        new_row = pd.DataFrame([{
            "jenis": jenis,
            "nama": nama,
            "ph": ph,
            "rekomendasi": hasil
        }])
        if os.path.exists(EXCEL_PATH):
            df_existing = pd.read_excel(EXCEL_PATH)
            df_updated = pd.concat([df_existing, new_row], ignore_index=True)
        else:
            df_updated = new_row
        df_updated.to_excel(EXCEL_PATH, index=False)
    except Exception as e:
        print("Gagal menyimpan ke Excel:", e)

    kondisi = "Segar" if ph < 6 else "Busuk"

    return jsonify({
        "gambar_url": f"/{UPLOAD_FOLDER}/{filename}",
        "jenis": jenis,
        "nama": nama,
        "ph": ph,
        "kondisi": kondisi,
        "hasil": hasil
    })

# ===============================
# RUN APP
# ===============================
if __name__ == "__main__":
    app.run(debug=True)
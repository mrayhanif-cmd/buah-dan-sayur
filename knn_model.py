import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

def load_and_train():
    df = pd.read_excel("dataset/dataset_buah_sayur_knn.xlsx")

    enc_k = LabelEncoder()
    df["Kontaminan_enc"] = enc_k.fit_transform(df["Kontaminan"])

    enc_j = LabelEncoder()
    df["Jenis_enc"] = enc_j.fit_transform(df["Jenis"])

    enc_r = LabelEncoder()
    df["Rekomendasi_enc"] = enc_r.fit_transform(df["Rekomendasi Bahan Kimia"])

    X = df[["Jenis_enc", "Kontaminan_enc", "pH"]]
    y = df["Rekomendasi_enc"]

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)

    return model, {"Jenis": enc_j, "Kontaminan": enc_k, "Rekomendasi": enc_r}

def predict_pembersih(model, encoders, jenis, kontaminan, ph):
    enc_j = encoders["Jenis"]
    enc_k = encoders["Kontaminan"]
    enc_r = encoders["Rekomendasi"]

    j = enc_j.transform([jenis])[0]
    k = enc_k.transform([kontaminan])[0]

    pred = model.predict([[j, k, ph]])[0]
    return enc_r.inverse_transform([pred])[0]

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("dataset_prediksi_penyakit.csv")

# Encode kolom kategorikal
le_penyakit = LabelEncoder()
le_gejala = LabelEncoder()
le_jk = LabelEncoder()

df["Penyakit"] = le_penyakit.fit_transform(df["Penyakit"])
df["Gejala"] = le_gejala.fit_transform(df["Gejala"])
df["Jenis_Kelamin"] = le_jk.fit_transform(df["Jenis_Kelamin"])

# Fitur & target
X = df[["Usia", "Jenis_Kelamin", "Berat_Badan", "Tinggi_Badan", "Gejala"]]
y = df["Penyakit"]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Mapping penyakit → obat & pantangan
mapping = df[["Penyakit", "Obat", "Pantangan"]].drop_duplicates()
mapping["Penyakit_Label"] = le_penyakit.inverse_transform(mapping["Penyakit"])

# ================== Streamlit App ==================
st.title("🩺 Prediksi Penyakit & Rekomendasi Obat")

st.sidebar.header("Input Data Pasien")
usia = st.sidebar.number_input("Usia", min_value=1, max_value=100, value=25)
jenis_kelamin = st.sidebar.selectbox("Jenis Kelamin", le_jk.classes_)
berat = st.sidebar.number_input("Berat Badan (kg)", min_value=10, max_value=200, value=60)
tinggi = st.sidebar.number_input("Tinggi Badan (cm)", min_value=50, max_value=250, value=170)
gejala = st.sidebar.selectbox("Gejala", le_gejala.classes_)

if st.sidebar.button("Prediksi"):
    # Encode input
    jk_encoded = le_jk.transform([jenis_kelamin])[0]
    gejala_encoded = le_gejala.transform([gejala])[0]
    input_data = [[usia, jk_encoded, berat, tinggi, gejala_encoded]]
    pred = model.predict(input_data)[0]
    penyakit = le_penyakit.inverse_transform([pred])[0]

    # Ambil obat & pantangan
    info = mapping[mapping["Penyakit_Label"] == penyakit].iloc[0]
    obat = info["Obat"]
    pantangan = info["Pantangan"]

    st.subheader("Hasil Prediksi")
    st.write(f"**Penyakit Terdeteksi:** {penyakit}")
    st.write(f"**Obat Disarankan:** {obat}")
    st.write(f"**Pantangan:** {pantangan}")

    # Grafik hasil prediksi
    st.subheader("Grafik Prediksi")
    fig, ax = plt.subplots()
    sns.barplot(x=[penyakit], y=[1], ax=ax, palette="coolwarm")
    ax.set_ylabel("Deteksi (1 = terdeteksi)")
    st.pyplot(fig)

# Distribusi Penyakit di Dataset
st.subheader("📊 Distribusi Penyakit")
fig1, ax1 = plt.subplots(figsize=(8,5))
sns.countplot(x=le_penyakit.inverse_transform(df["Penyakit"]),
              order=pd.Series(le_penyakit.inverse_transform(df["Penyakit"])).value_counts().index,
              palette="viridis", ax=ax1)
plt.xticks(rotation=45)
st.pyplot(fig1)

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# Load Dataset
# ======================
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

# Training model
# Daftar kolom target
kolom_target = ["Penyakit", "Obat", "Pantangan"]

# Pastikan semua kolom ada, kalau tidak → bikin kosong
for kol in kolom_target:
    if kol not in df.columns:
        df[kol] = ""

# Buat mapping unik
mapping = df[kolom_target].drop_duplicates()

# Tampilkan hasil
print("✅ Kolom berhasil diproses!")
print("Jumlah data mapping:", len(mapping))
print("\nContoh isi mapping:")
print(mapping.head(10))  # tampilkan 10 data pertama

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

print("Kolom yang ada:", df.columns)

# biar aman, hanya ambil kolom yang benar-benar ada
kolom_ada = [col for col in ["Penyakit", "Obat", "Pantangan"] if col in df.columns]
mapping = df[kolom_ada].drop_duplicates()


# Mapping Obat & Pantangan
mapping = df[["Penyakit", "Obat", "Pantangan"]].drop_duplicates()
mapping["Penyakit_Label"] = le_penyakit.inverse_transform(mapping["Penyakit"])

# ======================
# Streamlit UI
# ======================
st.title("🩺 Sistem Prediksi Penyakit & Rekomendasi Obat")

st.sidebar.header("Input Data Pasien")
usia = st.sidebar.number_input("Usia", min_value=1, max_value=100, value=25)
jenis_kelamin = st.sidebar.selectbox("Jenis Kelamin", le_jk.classes_)
berat = st.sidebar.number_input("Berat Badan (kg)", min_value=10, max_value=200, value=60)
tinggi = st.sidebar.number_input("Tinggi Badan (cm)", min_value=50, max_value=250, value=170)
gejala = st.sidebar.selectbox("Gejala", le_gejala.classes_)

if st.sidebar.button("Prediksi Penyakit"):
    jk_encoded = le_jk.transform([jenis_kelamin])[0]
    gejala_encoded = le_gejala.transform([gejala])[0]
    input_data = [[usia, jk_encoded, berat, tinggi, gejala_encoded]]
    pred = model.predict(input_data)[0]
    penyakit = le_penyakit.inverse_transform([pred])[0]

    info = mapping[mapping["Penyakit_Label"] == penyakit].iloc[0]
    obat = info["Obat"]
    pantangan = info["Pantangan"]

    st.subheader("Hasil Prediksi")
    st.write(f"**Penyakit Terdeteksi:** {penyakit}")
    st.write(f"**Obat Disarankan:** {obat}")
    st.write(f"**Pantangan:** {pantangan}")

    # Grafik hasil prediksi user
    st.subheader("Grafik Hasil Prediksi")
    fig, ax = plt.subplots()
    sns.barplot(x=[penyakit], y=[1], ax=ax, palette="coolwarm")
    ax.set_ylabel("Deteksi (1 = terdeteksi)")
    st.pyplot(fig)

# ======================
# Grafik Distribusi Dataset
# ======================
st.subheader("📊 Statistik Dataset")

# Distribusi Penyakit
fig1, ax1 = plt.subplots(figsize=(8,5))
sns.countplot(x=le_penyakit.inverse_transform(df["Penyakit"]), 
              order=pd.Series(le_penyakit.inverse_transform(df["Penyakit"])).value_counts().index,
              palette="viridis", ax=ax1)
ax1.set_title("Distribusi Penyakit")
plt.xticks(rotation=45)
st.pyplot(fig1)

# Distribusi Obat
fig2, ax2 = plt.subplots(figsize=(8,5))
sns.countplot(x=df["Obat"], order=df["Obat"].value_counts().index, palette="magma", ax=ax2)
ax2.set_title("Distribusi Obat")
plt.xticks(rotation=45)
st.pyplot(fig2)

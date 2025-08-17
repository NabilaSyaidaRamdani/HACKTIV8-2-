
# 🌸 Acne Detection & Skincare Recommendation using YOLO & GPT

## 📌 Project Overview

Proyek ini bertujuan untuk membangun sistem **deteksi jerawat berbasis YOLO** yang dapat mengenali berbagai jenis jerawat (blackhead, whitehead, papule, pustule, dan nodule) dari foto wajah.
Selain mendeteksi dan menghitung jumlah jerawat, sistem ini juga terhubung dengan **Large Language Model (GPT-4o-mini)** untuk memberikan **rekomendasi skincare dasar** sesuai kondisi kulit yang terdeteksi.

Dengan adanya sistem ini, pengguna dapat memperoleh insight kondisi kulitnya secara otomatis dan mendapatkan edukasi skincare yang lebih personal.

---

## 📂 Raw Dataset Link

Dataset yang digunakan untuk melatih model YOLO berasal dari [Roboflow Acne Detection Dataset](https://roboflow.com/) dengan tambahan anotasi manual.
Dataset berisi ribuan gambar wajah dengan label jerawat:

* Blackhead
* Whitehead
* Papule
* Pustule
* Nodule

---

## 🔎 Insight & Findings

Dari proses training dan evaluasi model YOLO:

* Model dapat mendeteksi jerawat dengan akurasi yang cukup baik, meskipun performa tiap kelas berbeda.
* Kelas **Whitehead** memiliki nilai Average Precision (AP) tertinggi, sedangkan kelas **Papule** cenderung lebih sulit dideteksi.
* Hasil deteksi divisualisasikan dalam bentuk **bounding box** dan dihitung jumlahnya untuk setiap jenis jerawat.

Insight tambahan:

* Dengan informasi jumlah jerawat per tipe, pengguna dapat memahami kondisi kulit lebih detail.
* Integrasi LLM membantu menjelaskan hasil deteksi dengan bahasa yang mudah dipahami dan memberi rekomendasi perawatan dasar.

---

## 🤖 AI Support Explanation

Proyek ini memanfaatkan **kombinasi Computer Vision dan Natural Language Processing**:

1. **YOLO (You Only Look Once)**

   * Digunakan untuk deteksi jerawat pada gambar wajah.
   * Output berupa bounding box, label jenis jerawat, dan confidence score.
   * Model dilatih khusus dengan dataset jerawat untuk meningkatkan akurasi.

2. **GPT-4o-mini**

   * Digunakan untuk menghasilkan rekomendasi skincare berdasarkan hasil deteksi.
   * Memberikan penjelasan singkat mengenai kondisi kulit pengguna.
   * Memberikan rekomendasi sederhana seperti cleanser, serum, dan treatment yang sesuai.

---

✨ **Tujuan akhir:**
Membuat sistem **deteksi jerawat otomatis** dengan rekomendasi skincare berbasis AI yang **mudah digunakan**, **informatif**, dan **bermanfaat** untuk edukasi perawatan kulit.

---

👉 Bisa langsung dicoba dengan menjalankan aplikasi Streamlit yang sudah disediakan.
https://hacktiv8.streamlit.app/



<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/92422a9d-5734-4273-8dc5-e653e546418f" />


<img width="435" height="770" alt="image" src="https://github.com/user-attachments/assets/d3d35127-e0cc-4b34-9d4f-61c90fd2bef9" />


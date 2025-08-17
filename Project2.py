import os
import cv2
import streamlit as st
import tempfile
from ultralytics import YOLO

# ===============================
# Setup Streamlit UI
# ===============================
st.title("🧴 Acne Detection & Skincare Recommendation")
st.write("Upload foto wajahmu untuk deteksi jerawat dan rekomendasi skincare dasar.")

# Upload gambar
uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Simpan file sementara dengan ekstensi benar
    suffix = "." + uploaded_file.name.split(".")[-1]
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(uploaded_file.read())
    image_path = temp_file.name

    # ===============================
    # 1. Load YOLO Model
    # ===============================
    yolo_model = YOLO("best.pt")  # Pastikan best.pt ada di project folder
    results = yolo_model(image_path)[0]

    # ===============================
    # 2. Hitung jumlah jerawat
    # ===============================
    class_ids = results.boxes.cls.cpu().numpy()
    class_names = results.names
    detected_acne = [class_names[int(i)] for i in class_ids]

    summary = {}
    for acne in detected_acne:
        summary[acne] = summary.get(acne, 0) + 1

    summary_text = ", ".join([f"{count} {name}" for name, count in summary.items()])
    st.subheader("📊 Hasil Deteksi")
    st.write(summary_text if summary_text else "Tidak ada jerawat terdeteksi.")

    # ===============================
    # 3. Tampilkan gambar hasil deteksi
    # ===============================
    annotated_img = results.plot()
    st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
             caption="Hasil Deteksi",
             use_column_width=True)

    # ===============================
    # 4. Hubungkan ke LLM untuk rekomendasi skincare
    # ===============================
    if summary_text:
        openai.api_key = os.environ["github_pat_11A67J3OY0BaBrNx2t0A8P_O3xErnxZHUARep0t5w3bjHpM3zdQYsEN6CC4nGlHMwrFAKEXVX7gqNio6Nq"]
        openai.base_url = "https://models.inference.ai.azure.com"

        prompt = f"""
        Saya mendeteksi kondisi kulit dengan hasil: {summary_text}.
        Tolong berikan penjelasan singkat mengenai kondisi ini
        dan rekomendasi skincare dasar yang sesuai (misalnya cleanser, serum, treatment).
        Gunakan bahasa yang mudah dipahami.
        """

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )

        st.subheader("💡 Rekomendasi Skincare")
        st.write(response.choices[0].message["content"])

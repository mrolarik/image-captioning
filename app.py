import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import requests
from io import BytesIO

# ==== Load model and processor ====
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

# ==== ตัวอย่างภาพ ====
sample_images = {
    "สุนัข": "https://images.unsplash.com/photo-1552053831-71594a27632d?q=80&w=3062&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    "จักรยาน": "https://plus.unsplash.com/premium_photo-1663091740058-b07d3f6832c2?q=80&w=2940&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
}

st.title("🖼️ Image Captioning App")
st.write("เลือกรูปจากตัวอย่างด้านล่าง, อัปโหลด, หรือป้อน URL แล้วระบบจะอธิบายรูปภาพให้คุณ")

# ==== ตัวอย่างภาพ (แสดงใน main area) ====
st.subheader("📌 ตัวอย่างรูปภาพ")
cols = st.columns(len(sample_images))
selected_sample = None

for idx, (label, url) in enumerate(sample_images.items()):
    with cols[idx]:
        st.image(url, caption=label, use_column_width=True)
        if st.button(f"ใช้รูปภาพ: {label}", key=f"sample_{idx}"):
            selected_sample = url
            st.session_state['selected_sample_label'] = label

# ==== โหลดรูป ====
image = None

# ✅ จากตัวอย่าง
if selected_sample:
    try:
        response = requests.get(selected_sample)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        st.success(f"✅ โหลดรูปภาพตัวอย่าง ({st.session_state.get('selected_sample_label', '')}) สำเร็จ")
    except:
        st.error("❌ ไม่สามารถโหลดรูปภาพตัวอย่างได้")

# ✅ จากการอัปโหลด
uploaded_file = st.file_uploader("📁 หรืออัปโหลดรูปภาพของคุณ", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.success("✅ โหลดรูปภาพที่อัปโหลดสำเร็จ")

# ✅ จาก URL
image_url_input = st.text_input("🔗 หรือป้อน URL ของรูปภาพ (ลงท้ายด้วย .jpg, .png, .jpeg)")
if image_url_input and not uploaded_file and not selected_sample:
    try:
        if image_url_input.lower().endswith((".jpg", ".jpeg", ".png")):
            response = requests.get(image_url_input)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            st.success("✅ โหลดรูปภาพจาก URL สำเร็จ")
        else:
            st.warning("⚠️ URL ควรลงท้ายด้วย .jpg, .jpeg หรือ .png")
    except:
        st.error("❌ ไม่สามารถโหลดภาพจาก URL ได้ กรุณาตรวจสอบลิงก์อีกครั้ง")

# ==== สร้างคำอธิบาย ====
if image:
    st.image(image, caption="📸 รูปภาพที่เลือก", use_container_width=True)

    with st.spinner("🧠 กำลังวิเคราะห์และอธิบายรูปภาพ..."):
        inputs = processor(images=image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

    st.success("📝 คำอธิบายจากโมเดล:")
    st.markdown(f"**{caption}**")

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

# ==== Sidebar: Sample Images ====
st.sidebar.header("🖼️ ตัวอย่างรูปภาพ")
sample_images = {
    "สุนัข": "https://images.unsplash.com/photo-1601758123927-196d5f5fb692",
    "จักรยาน": "https://images.unsplash.com/photo-1589571894960-20bbe2828f3b"
}

selected_sample = None
for label, url in sample_images.items():
    st.sidebar.image(url, caption=label, use_column_width=True)
    if st.sidebar.button(f"ใช้รูปภาพ: {label}"):
        selected_sample = url
        st.session_state['selected_sample_label'] = label

# ==== Main UI ====
st.title("🖼️ Image Captioning App")
st.write("อัปโหลดรูปภาพ หรือเลือกรูปจากตัวอย่างทางด้านซ้าย แล้วระบบจะอธิบายภาพให้คุณ")

# ==== Load image ====
image = None

if selected_sample:
    try:
        response = requests.get(selected_sample)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        st.success(f"✅ โหลดรูปภาพตัวอย่าง ({st.session_state.get('selected_sample_label', '')}) สำเร็จ")
    except:
        st.error("❌ ไม่สามารถโหลดรูปภาพตัวอย่างได้")

uploaded_file = st.file_uploader("หรืออัปโหลดรูปภาพของคุณเอง", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.success("✅ โหลดรูปภาพที่อัปโหลดสำเร็จ")

# ==== Generate Caption ====
if image:
    st.image(image, caption="📸 รูปภาพที่เลือก", use_container_width=True)

    with st.spinner("🧠 กำลังวิเคราะห์และอธิบายรูปภาพ..."):
        inputs = processor(images=image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

    st.success("📝 คำอธิบายจากโมเดล:")
    st.markdown(f"**{caption}**")

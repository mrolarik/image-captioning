import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import requests
from io import BytesIO
import random

# ==== Load model and processor ====
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

# ==== Sample image collection (many entries possible) ====
sample_images = {
    "สุนัข": "https://images.unsplash.com/photo-1552053831-71594a27632d?q=80&w=3062&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    "จักรยาน": "https://plus.unsplash.com/premium_photo-1663091740058-b07d3f6832c2?q=80&w=2940&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    "แมว": "https://plus.unsplash.com/premium_photo-1677181729163-33e6b59d5c8f?q=80&w=3087&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    "รถยนต์": "https://images.unsplash.com/photo-1533473359331-0135ef1b58bf?q=80&w=2940&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    "ภูเขา": "https://images.unsplash.com/photo-1465056836041-7f43ac27dcb5?q=80&w=2942&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
}

st.title("🖼️ Image Captioning App")
st.write("เลือกรูปจากตัวอย่างด้านล่าง, อัปโหลด, หรือป้อน URL แล้วระบบจะอธิบายรูปภาพให้คุณ")

# ==== Randomly select 2 images ====
if 'random_keys' not in st.session_state:
    st.session_state.random_keys = random.sample(list(sample_images.keys()), 2)

if st.button("🔀 สุ่มรูปภาพตัวอย่างใหม่"):
    st.session_state.random_keys = random.sample(list(sample_images.keys()), 2)

# ==== Display the 2 random images ====
st.subheader("🖼️ ตัวอย่างรูปภาพ (สุ่ม 2 รูป)")
image = None
selected_sample = None

cols = st.columns(2)
for i, key in enumerate(st.session_state.random_keys):
    with cols[i]:
        st.image(sample_images[key], caption=key, use_container_width=True)
        if st.button(f"✅ ใช้รูปภาพ: {key}"):
            st.session_state.selected_sample_label = key
            st.session_state.selected_sample_url = sample_images[key]

# ==== Load selected sample ====
if 'selected_sample_url' in st.session_state:
    try:
        response = requests.get(st.session_state.selected_sample_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        st.success(f"✅ โหลดรูปภาพตัวอย่าง ({st.session_state.selected_sample_label}) สำเร็จ")
    except:
        st.error("❌ ไม่สามารถโหลดภาพตัวอย่างได้")

# ==== File upload ====
uploaded_file = st.file_uploader("📁 หรืออัปโหลดรูปภาพของคุณ", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.success("✅ โหลดรูปภาพที่อัปโหลดสำเร็จ")

# ==== URL input ====
image_url_input = st.text_input(
    "🔗 หรือป้อน URL ของรูปภาพ (ลงท้ายด้วย .jpg, .png, .jpeg)",
    placeholder="ตัวอย่าง: https://images.unsplash.com/photo-1601758123927-196d5f5fb692"
)
if image_url_input and not uploaded_file and 'selected_sample_url' not in st.session_state:
    try:
        if image_url_input.lower().endswith((".jpg", ".jpeg", ".png")):
            response = requests.get(image_url_input)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            st.success("✅ โหลดรูปภาพจาก URL สำเร็จ")
        else:
            st.warning("⚠️ URL ควรลงท้ายด้วย .jpg, .jpeg หรือ .png")
    except:
        st.error("❌ ไม่สามารถโหลดภาพจาก URL ได้")

# ==== Caption generation ====
if image:
    st.image(image, caption="📸 รูปภาพที่เลือก", use_container_width=True)

    with st.spinner("🧠 กำลังอธิบายรูปภาพ..."):
        inputs = processor(images=image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

    st.success("📝 คำอธิบายจากโมเดล:")
    st.markdown(f"**{caption}**")

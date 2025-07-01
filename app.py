import streamlit as st

# ✅ Must be first Streamlit command
st.set_page_config(page_title="🚗 Car Damage Classifier", layout="centered")

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import requests
import os
from io import StringIO

# ----------------------------------
# ✅ Config
class_names = ['F_Breakage', 'F_Crushed', 'F_Normal', 'R_Breakage', 'R_Crushed', 'R_Normal']
model_path = "car_damage_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------
# ✅ Robust Google Drive Downloader
def download_model_from_drive(file_id, filename):
    """Download a file from Google Drive using chunked streaming (handles confirmation tokens)."""
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    token = get_confirm_token(response)
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    with open(filename, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

# Download model before loading
download_model_from_drive("1b-vO7pxnKlnhbS_-0E6Rzzz-QxOC_jg5", model_path)

# ----------------------------------
# ✅ Load model
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model

model = load_model()

# ----------------------------------
# ✅ Grad-CAM setup
cam_extractor = SmoothGradCAMpp(model, target_layer="layer4")

# ----------------------------------
# ✅ Image Preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# ----------------------------------
# ✅ Prediction & Grad-CAM Heatmap
def predict_and_visualize(image):
    input_tensor = preprocess_image(image).to(device)

    model.zero_grad()
    output = model(input_tensor)

    pred_class = output.argmax().item()

    # Grad-CAM extraction
    activation_map = cam_extractor(pred_class, output)[0].squeeze(0).cpu()

    # Overlay heatmap
    result = overlay_mask(
        to_pil_image(input_tensor.squeeze().cpu()),
        to_pil_image(activation_map, mode='F'),
        alpha=0.5
    )

    return result, class_names[pred_class]

# ----------------------------------
# ✅ Streamlit UI
st.title("🚗 Car Damage Detection with Grad-CAM")

# 📤 Image input section
st.subheader("📸 Choose Image Source")
input_option = st.radio("Select image input mode:", ["Upload Image", "Use Camera"])

if input_option == "Upload Image":
    uploaded_file = st.file_uploader("Upload a car image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
    else:
        image = None
else:
    camera_input = st.camera_input("Take a photo of the car")
    if camera_input:
        image = Image.open(camera_input).convert("RGB")
    else:
        image = None

# ✅ Prediction and Grad-CAM
if image:
    with st.spinner("🔍 Analyzing..."):
        heatmap_img, predicted_label = predict_and_visualize(image)

    st.success(f"✅ Predicted: **{predicted_label}**")

    # 🔄 Side-by-side display
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="🖼️ Input Image", use_column_width=True)

    with col2:
        st.image(heatmap_img, caption="🔥 Grad-CAM Heatmap", use_column_width=True)

    # 📥 Download Prediction Report
    report_str = f"Predicted class: {predicted_label}\nModel: ResNet18\nGrad-CAM applied: Yes"
    st.download_button(
        label="📥 Download Prediction Report",
        data=report_str,
        file_name="car_damage_prediction.txt",
        mime="text/plain"
    )

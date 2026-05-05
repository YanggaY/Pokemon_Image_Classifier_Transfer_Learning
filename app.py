import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Setting
MODEL_NAME = "resnet34"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "PokemonData")

MODEL_PATH = os.path.join(
    BASE_DIR,
    "pokemon_resnet34_fixed_True_pre_True.pth"
)


# 클래스 이름 로드
if os.path.exists(DATA_DIR):
    class_names = sorted(os.listdir(DATA_DIR))
else:
    st.error(f"데이터 폴더를 찾을 수 없습니다: {DATA_DIR}")
    st.stop()


# 모델 로드 함수
@st.cache_resource
def load_model(model_path):
    model = models.resnet34(weights=None)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))

    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)

    model.eval()
    return model


if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

model = load_model(MODEL_PATH)


# 이미지 전처리
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])



# Streamlit UI
st.title("Pokemon Classifier")
st.write("Upload a Pokemon image and the model will predict its class.")

st.info(
    "Model: ResNet34 / Pretrained: True / Feature Extract: True"
)

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    img_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        top5_prob, top5_idx = torch.topk(probabilities, 5)

    st.subheader("Top-5 Predictions")

    for i in range(5):
        idx = top5_idx[0][i].item()
        prob = top5_prob[0][i].item() * 100
        name = class_names[idx]

        st.write(f"{i + 1}. **{name}** — {prob:.2f}%")

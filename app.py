import mlflow
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import streamlit as st
from PIL import Image
import datetime
from db import add_user, add_prediction

model_uri = "runs:/bc849e88bbd44802a391bd8a71ab6e42/model"
model = mlflow.pytorch.load_model(model_uri)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def infer(model, image):
    model.eval()
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        probs = torch.nn.functional.softmax(outputs, dim=1)
    return preds.item(), probs[0, preds].item()

st.title("Регистрация пользователя")
name = st.text_input("Ваше имя")
email = st.text_input("Ваша электронная почта")

if st.button("Зарегистрироваться"):
    if name and email:
        add_user(name, email)
        st.success(f"Пользователь {name} зарегистрирован!")
    else:
        st.error("Пожалуйста, заполните все поля.")

st.title("Загрузите изображение для предсказания")
uploaded_file = st.file_uploader("Выберите изображение...", type=["png", "jpg", "jpeg"])

st.title("Fashion MNIST Model Inference")
st.write("Загрузите изображение, и модель предскажет его класс.")

uploaded_file = st.file_uploader("Выберите изображение...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Загруженное изображение", use_column_width=True)

    class_id, prob = infer(model, image)
    
    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]

    st.write(f"Предсказанный класс: **{class_names[class_id]}**")
    st.write(f"Вероятность: **{prob * 100:.2f}%**")

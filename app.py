import mlflow
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import streamlit as st
from PIL import Image

# Загрузка модели из MLflow
model_uri = "runs:/bc849e88bbd44802a391bd8a71ab6e42/model"  # Замените на ваш run-id
model = mlflow.pytorch.load_model(model_uri)

# Перемещаем модель на устройство (GPU или CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()  # Устанавливаем модель в режим инференса

# Подготовка трансформации для данных (нормализация как для Fashion MNIST)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Преобразуем в 3 канала
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Нормализация для 3 каналов
])

# Функция для инференса
def infer(model, image):
    model.eval()  # Переводим модель в режим инференса
    image = transform(image).unsqueeze(0).to(device)  # Преобразуем изображение и добавляем батч-размер
    with torch.no_grad():  # Отключаем градиенты для инференса
        outputs = model(image)  # Прогоняем через модель
        _, preds = torch.max(outputs, 1)  # Получаем индексы максимальных значений
        probs = torch.nn.functional.softmax(outputs, dim=1)  # Вычисляем вероятности
    return preds.item(), probs[0, preds].item()

# Интерфейс Streamlit
st.title("Fashion MNIST Model Inference")
st.write("Загрузите изображение, и модель предскажет его класс.")

# Загрузка изображения пользователем
uploaded_file = st.file_uploader("Выберите изображение...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Открываем изображение
    image = Image.open(uploaded_file)
    st.image(image, caption="Загруженное изображение", use_column_width=True)

    # Инференс
    class_id, prob = infer(model, image)
    
    # Маппинг меток классов для Fashion MNIST
    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]

    st.write(f"Предсказанный класс: **{class_names[class_id]}**")
    st.write(f"Вероятность: **{prob * 100:.2f}%**")


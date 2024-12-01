import mlflow
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np

# Указываем путь к модели в MLflow (в данном случае, это имя эксперимента и модель)
model_uri = "runs:/<run-id>/model"  # Замените <run-id> на ID вашей сохранённой модели

# Загружаем модель из MLflow
model = mlflow.pytorch.load_model(model_uri)

# Перемещаем модель на устройство (GPU или CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()  # Устанавливаем модель в режим инференса

# Подготовка трансформации для данных (нормализация как для Fashion MNIST)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Для Fashion MNIST
])

# Загружаем тестовые данные
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Функция для инференса
def infer(model, data_loader):
    model.eval()  # Переводим модель в режим инференса
    all_preds = []

    with torch.no_grad():  # Отключаем градиенты для инференса
        for images, _ in data_loader:
            images = images.to(device)
            outputs = model(images)  # Прогоняем через модель
            _, preds = torch.max(outputs, 1)  # Получаем индексы максимальных значений
            all_preds.extend(preds.cpu().numpy())  # Переводим на CPU и добавляем к результатам

    return np.array(all_preds)

# Запускаем инференс
predictions = infer(model, test_loader)

# Выводим несколько предсказаний
print(predictions[:10])  # Выводим первые 10 предсказанных классов

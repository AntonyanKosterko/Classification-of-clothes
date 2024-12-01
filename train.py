import mlflow
import mlflow.pytorch
import optuna
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Трансформации для загрузки и нормализации данных
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Загружаем данные
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

experiment_name = "fashion_mnist_resnet_experiment2"
mlflow.set_experiment(experiment_name)

# Определение модели (ResNet-18)
def create_model():
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)  # Последний слой для 10 классов
    
    # Перемещаем модель на доступное устройство (GPU, если оно есть)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    return model

# Определение функции для обучения модели с учетом гиперпараметров
def train_model_with_optuna(trial):
    epochs = trial.suggest_int('epochs', 3, 10)
    batch_size = trial.suggest_int('batch_size', 32, 128, step=32)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Перезагружаем даталоадер с новым размером батча
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    mlflow.start_run()

    mlflow.log_param('epochs', epochs)
    mlflow.log_param('batch_size', batch_size)
    mlflow.log_param('learning_rate', learning_rate)

    model.train()  # Устанавливаем модель в режим обучения
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Логируем метрики по эпохе
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        
        # Логируем метрики для каждой эпохи в MLflow
        mlflow.log_metric('epoch_loss', epoch_loss, step=epoch)
        mlflow.log_metric('epoch_accuracy', epoch_accuracy, step=epoch)

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    # Логируем финальную модель
    mlflow.pytorch.log_model(model, "model")

    # Завершаем эксперимент
    mlflow.end_run()

    return epoch_loss  # Оптимизируем по минимальному loss

# Определение исследования Optuna
study = optuna.create_study(direction='minimize')
study.optimize(train_model_with_optuna, n_trials=5)  # Количество проб

print("Лучшие гиперпараметры:", study.best_params)

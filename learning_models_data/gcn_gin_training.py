# ===========================================
# 📦 Импорты
# ===========================================
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool
from sklearn.model_selection import KFold
from collections import Counter

# ===========================================
# 🔧 Настройки путей и параметров
# ===========================================
# Папка с данными
data_base_dir = r"C:\Users\Killua\PycharmProjects\pythonProject6\train_binary_classifier_27_04\data_pt"

# Папка для сохранения дефолтных моделей
default_save_dir = r"C:\Users\Killua\PycharmProjects\pythonProject6\train_binary_classifier_27_04\result_models\models_default"

# Папка для сохранения моделей после кросс-валидации
cv_base_dir = r"C:\Users\Killua\PycharmProjects\pythonProject6\train_binary_classifier_27_04\result_models\models_cv"

# Гиперпараметры
input_dim = 3
hidden_dim = 64
num_epochs = 30
learning_rate = 1e-3
batch_size = 32
n_splits = 5  # Для кросс-валидации

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Создание папок
os.makedirs(default_save_dir, exist_ok=True)
os.makedirs(cv_base_dir, exist_ok=True)

# ===========================================
# 📂 Dataset класс
# ===========================================
class GraphPairDiskDataset(torch.utils.data.Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.file_list = sorted(os.listdir(directory))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.file_list[idx])
        data = torch.load(file_path)
        return (
            data["x1"], data["edge_index1"], data["batch1"],
            data["x2"], data["edge_index2"], data["batch2"],
            data["y"]
        )

# ===========================================
# 🔥 Кастомная collate_fn для списков графов
# ===========================================
def collate_fn(batch):
    x1_list, edge_index1_list, batch1_list, x2_list, edge_index2_list, batch2_list, y_list = zip(*batch)
    return (
        x1_list, edge_index1_list, batch1_list,
        x2_list, edge_index2_list, batch2_list,
        torch.cat(y_list, dim=0)
    )

# ===========================================
# 📊 Подсчёт количества классов в выборках
# ===========================================
def count_labels(dataset):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    labels = []
    for batch in loader:
        _, _, _, _, _, _, y = batch
        labels.extend(y.tolist())
    counter = Counter(labels)
    return counter

# ===========================================
# 🧠 Модель
# ===========================================
class GCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return x

class GINEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        nn1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        nn2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return x

class GraphSimilarityModel(nn.Module):
    def __init__(self, encoder, hidden_dim):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x1, edge_index1, batch1, x2, edge_index2, batch2):
        z1 = self.encoder(x1, edge_index1, batch1)
        z2 = self.encoder(x2, edge_index2, batch2)
        combined = torch.cat([z1, z2, torch.abs(z1 - z2), z1 * z2], dim=1)
        out = self.classifier(combined)
        return out.squeeze(-1)

# ===========================================
# 🛠 Функции обучения и оценки
# ===========================================
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in loader:
        x1_list, edge_index1_list, batch1_list, x2_list, edge_index2_list, batch2_list, y = batch

        outputs = []
        for x1, edge_index1, batch1, x2, edge_index2, batch2 in zip(x1_list, edge_index1_list, batch1_list, x2_list, edge_index2_list, batch2_list):
            x1 = x1.to(device)
            edge_index1 = edge_index1.to(device)
            batch1 = batch1.to(device)
            x2 = x2.to(device)
            edge_index2 = edge_index2.to(device)
            batch2 = batch2.to(device)

            output = model(x1, edge_index1, batch1, x2, edge_index2, batch2)
            outputs.append(output)

        outputs = torch.stack(outputs)
        outputs = outputs.squeeze(-1)
        y = y.to(device)

        optimizer.zero_grad()
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.sigmoid(outputs) > 0.5
        correct += (preds.float() == y).sum().item()
        total += y.size(0)

    return total_loss / len(loader), correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for batch in loader:
        x1_list, edge_index1_list, batch1_list, x2_list, edge_index2_list, batch2_list, y = batch

        outputs = []
        for x1, edge_index1, batch1, x2, edge_index2, batch2 in zip(x1_list, edge_index1_list, batch1_list, x2_list, edge_index2_list, batch2_list):
            x1 = x1.to(device)
            edge_index1 = edge_index1.to(device)
            batch1 = batch1.to(device)
            x2 = x2.to(device)
            edge_index2 = edge_index2.to(device)
            batch2 = batch2.to(device)

            output = model(x1, edge_index1, batch1, x2, edge_index2, batch2)
            outputs.append(output)

        outputs = torch.stack(outputs)
        outputs = outputs.squeeze(-1)
        y = y.to(device)

        loss = criterion(outputs, y)
        total_loss += loss.item()

        preds = torch.sigmoid(outputs) > 0.5
        correct += (preds.float() == y).sum().item()
        total += y.size(0)

    return total_loss / len(loader), correct / total

# ===========================================
# 🔥 Загрузка датасетов
# ===========================================
train_dataset = GraphPairDiskDataset(os.path.join(data_base_dir, "train"))
val_dataset = GraphPairDiskDataset(os.path.join(data_base_dir, "val"))
test_dataset = GraphPairDiskDataset(os.path.join(data_base_dir, "test"))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# ===========================================
# 📊 Подсчёт количества классов
# ===========================================
print("\n🔎 Подсчёт количества примеров 1 и 0:")

train_counter = count_labels(train_dataset)
val_counter = count_labels(val_dataset)

print(f"Train dataset: {train_counter}")
print(f"Val dataset: {val_counter}")

# ===========================================
# 🚀 Дефолтное обучение GCN и GIN
# ===========================================
print("\n🚀 Старт дефолтного обучения (без кросс-валидации)")

gcn_model = GraphSimilarityModel(GCNEncoder(input_dim, hidden_dim), hidden_dim).to(device)
gin_model = GraphSimilarityModel(GINEncoder(input_dim, hidden_dim), hidden_dim).to(device)

gcn_optimizer = torch.optim.Adam(gcn_model.parameters(), lr=learning_rate)
gin_optimizer = torch.optim.Adam(gin_model.parameters(), lr=learning_rate)

criterion = nn.BCEWithLogitsLoss()

gcn_train_losses, gcn_val_losses, gcn_train_accs, gcn_val_accs = [], [], [], []
gin_train_losses, gin_val_losses, gin_train_accs, gin_val_accs = [], [], [], []

best_gcn_val_acc = 0.0
best_gin_val_acc = 0.0

for epoch in range(num_epochs):
    # GCN
    train_loss, train_acc = train_epoch(gcn_model, train_loader, gcn_optimizer, criterion, device)
    val_loss, val_acc = evaluate(gcn_model, val_loader, criterion, device)
    gcn_train_losses.append(train_loss)
    gcn_val_losses.append(val_loss)
    gcn_train_accs.append(train_acc)
    gcn_val_accs.append(val_acc)
    if val_acc > best_gcn_val_acc:
        best_gcn_val_acc = val_acc
        torch.save(gcn_model.state_dict(), os.path.join(default_save_dir, "best_model_gcn.pt"))

    # GIN
    train_loss, train_acc = train_epoch(gin_model, train_loader, gin_optimizer, criterion, device)
    val_loss, val_acc = evaluate(gin_model, val_loader, criterion, device)
    gin_train_losses.append(train_loss)
    gin_val_losses.append(val_loss)
    gin_train_accs.append(train_acc)
    gin_val_accs.append(val_acc)
    if val_acc > best_gin_val_acc:
        best_gin_val_acc = val_acc
        torch.save(gin_model.state_dict(), os.path.join(default_save_dir, "best_model_gin.pt"))

# ===========================================
# 📈 Построение графика дефолтного обучения
# ===========================================
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(gcn_train_losses, label='GCN Train Loss')
plt.plot(gcn_val_losses, label='GCN Val Loss')
plt.plot(gin_train_losses, label='GIN Train Loss')
plt.plot(gin_val_losses, label='GIN Val Loss')
plt.title('Default Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(gcn_train_accs, label='GCN Train Acc')
plt.plot(gcn_val_accs, label='GCN Val Acc')
plt.plot(gin_train_accs, label='GIN Train Acc')
plt.plot(gin_val_accs, label='GIN Val Acc')
plt.title('Default Training Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(default_save_dir, "plots.png"))
plt.close()

print("\n✅ Дефолтное обучение завершено!")

# ===========================================
# 🚀 Старт Кросс-валидации
# ===========================================
print("\n🚀 Старт 5-Fold Кросс-валидации")

all_data_files = sorted(os.listdir(os.path.join(data_base_dir, "train")))
all_data_paths = [os.path.join(data_base_dir, "train", f) for f in all_data_files]

kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

all_fold_gcn_acc = []
all_fold_gin_acc = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_data_paths)):
    print(f"\n🔵 Fold {fold_idx+1}/{n_splits}")

    fold_train_files = [all_data_paths[i] for i in train_idx]
    fold_val_files = [all_data_paths[i] for i in val_idx]

    class FoldDataset(torch.utils.data.Dataset):
        def __init__(self, file_list):
            self.file_list = file_list

        def __len__(self):
            return len(self.file_list)

        def __getitem__(self, idx):
            data = torch.load(self.file_list[idx])
            return (
                data["x1"], data["edge_index1"], data["batch1"],
                data["x2"], data["edge_index2"], data["batch2"],
                data["y"]
            )

    fold_train_dataset = FoldDataset(fold_train_files)
    fold_val_dataset = FoldDataset(fold_val_files)

    fold_train_loader = DataLoader(fold_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    fold_val_loader = DataLoader(fold_val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Новые модели
    fold_gcn_model = GraphSimilarityModel(GCNEncoder(input_dim, hidden_dim), hidden_dim).to(device)
    fold_gin_model = GraphSimilarityModel(GINEncoder(input_dim, hidden_dim), hidden_dim).to(device)

    fold_gcn_optimizer = torch.optim.Adam(fold_gcn_model.parameters(), lr=learning_rate)
    fold_gin_optimizer = torch.optim.Adam(fold_gin_model.parameters(), lr=learning_rate)

    fold_criterion = nn.BCEWithLogitsLoss()

    fold_gcn_train_losses, fold_gcn_val_losses = [], []
    fold_gcn_train_accs, fold_gcn_val_accs = [], []

    fold_gin_train_losses, fold_gin_val_losses = [], []
    fold_gin_train_accs, fold_gin_val_accs = [], []

    best_fold_gcn_val_acc = 0.0
    best_fold_gin_val_acc = 0.0

    fold_save_dir = os.path.join(cv_base_dir, f"fold_{fold_idx}")
    os.makedirs(fold_save_dir, exist_ok=True)

    # Обучение GCN
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(fold_gcn_model, fold_train_loader, fold_gcn_optimizer, fold_criterion, device)
        val_loss, val_acc = evaluate(fold_gcn_model, fold_val_loader, fold_criterion, device)

        fold_gcn_train_losses.append(train_loss)
        fold_gcn_val_losses.append(val_loss)
        fold_gcn_train_accs.append(train_acc)
        fold_gcn_val_accs.append(val_acc)

        if val_acc > best_fold_gcn_val_acc:
            best_fold_gcn_val_acc = val_acc
            torch.save(fold_gcn_model.state_dict(), os.path.join(fold_save_dir, "best_model_gcn.pt"))

    all_fold_gcn_acc.append(best_fold_gcn_val_acc)

    # Обучение GIN
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(fold_gin_model, fold_train_loader, fold_gin_optimizer, fold_criterion, device)
        val_loss, val_acc = evaluate(fold_gin_model, fold_val_loader, fold_criterion, device)

        fold_gin_train_losses.append(train_loss)
        fold_gin_val_losses.append(val_loss)
        fold_gin_train_accs.append(train_acc)
        fold_gin_val_accs.append(val_acc)

        if val_acc > best_fold_gin_val_acc:
            best_fold_gin_val_acc = val_acc
            torch.save(fold_gin_model.state_dict(), os.path.join(fold_save_dir, "best_model_gin.pt"))

    all_fold_gin_acc.append(best_fold_gin_val_acc)

    # 📈 Сохраняем графики
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(fold_gcn_train_losses, label='GCN Train Loss')
    plt.plot(fold_gcn_val_losses, label='GCN Val Loss')
    plt.plot(fold_gin_train_losses, label='GIN Train Loss')
    plt.plot(fold_gin_val_losses, label='GIN Val Loss')
    plt.title(f'Fold {fold_idx} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(fold_gcn_train_accs, label='GCN Train Acc')
    plt.plot(fold_gcn_val_accs, label='GCN Val Acc')
    plt.plot(fold_gin_train_accs, label='GIN Train Acc')
    plt.plot(fold_gin_val_accs, label='GIN Val Acc')
    plt.title(f'Fold {fold_idx} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(fold_save_dir, "plots.png"))
    plt.close()

# ===========================================
# 📊 Финальные результаты кросс-валидации
# ===========================================
mean_gcn_acc = sum(all_fold_gcn_acc) / len(all_fold_gcn_acc)
mean_gin_acc = sum(all_fold_gin_acc) / len(all_fold_gin_acc)

print("\n✅ Кросс-валидация завершена!")
print(f"🎯 Средняя GCN Accuracy по фолдам: {mean_gcn_acc:.4f}")
print(f"🎯 Средняя GIN Accuracy по фолдам: {mean_gin_acc:.4f}")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

# ===========================================
# 📊 Оценка на тестовом наборе (финальная)
# ===========================================

print("\n🚀 Финальная оценка на тесте")

# Загружаем лучшие модели дефолтного обучения
gcn_model.load_state_dict(torch.load(os.path.join(default_save_dir, "best_model_gcn.pt")))
gin_model.load_state_dict(torch.load(os.path.join(default_save_dir, "best_model_gin.pt")))

gcn_model.eval()
gin_model.eval()

all_preds_gcn = []
all_preds_gin = []
all_labels = []

for batch in test_loader:
    x1_list, edge_index1_list, batch1_list, x2_list, edge_index2_list, batch2_list, y = batch

    outputs_gcn = []
    outputs_gin = []

    for x1, edge_index1, batch1, x2, edge_index2, batch2 in zip(x1_list, edge_index1_list, batch1_list, x2_list, edge_index2_list, batch2_list):
        x1 = x1.to(device)
        edge_index1 = edge_index1.to(device)
        batch1 = batch1.to(device)
        x2 = x2.to(device)
        edge_index2 = edge_index2.to(device)
        batch2 = batch2.to(device)

        with torch.no_grad():
            out_gcn = gcn_model(x1, edge_index1, batch1, x2, edge_index2, batch2)
            out_gin = gin_model(x1, edge_index1, batch1, x2, edge_index2, batch2)

        outputs_gcn.append(out_gcn.cpu())
        outputs_gin.append(out_gin.cpu())

    outputs_gcn = torch.stack(outputs_gcn).squeeze(-1)
    outputs_gin = torch.stack(outputs_gin).squeeze(-1)
    y = y.cpu()

    all_preds_gcn.append(torch.sigmoid(outputs_gcn))
    all_preds_gin.append(torch.sigmoid(outputs_gin))
    all_labels.append(y)

# Склеиваем
all_preds_gcn = torch.cat(all_preds_gcn)
all_preds_gin = torch.cat(all_preds_gin)
all_labels = torch.cat(all_labels)

# Бинаризация по порогу 0.5
pred_labels_gcn = (all_preds_gcn > 0.5).long()
pred_labels_gin = (all_preds_gin > 0.5).long()

# Метрики
print("\n🎯 Метрики для GCN на тесте:")
print(f"Accuracy: {accuracy_score(all_labels, pred_labels_gcn):.4f}")
print(f"Precision: {precision_score(all_labels, pred_labels_gcn):.4f}")
print(f"Recall: {recall_score(all_labels, pred_labels_gcn):.4f}")
print(f"F1-Score: {f1_score(all_labels, pred_labels_gcn):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(all_labels, pred_labels_gcn))

print("\n🎯 Метрики для GIN на тесте:")
print(f"Accuracy: {accuracy_score(all_labels, pred_labels_gin):.4f}")
print(f"Precision: {precision_score(all_labels, pred_labels_gin):.4f}")
print(f"Recall: {recall_score(all_labels, pred_labels_gin):.4f}")
print(f"F1-Score: {f1_score(all_labels, pred_labels_gin):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(all_labels, pred_labels_gin))

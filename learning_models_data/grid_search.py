# ===========================================
# 📦 Импорты
# ===========================================
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import pandas as pd
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool
from sklearn.metrics import accuracy_score, precision_score, recall_score
from collections import Counter

# ===========================================
# 🔧 Пути и базовые настройки
# ===========================================
data_base_dir = r"C:\Users\Killua\PycharmProjects\pythonProject6\train_binary_classifier_27_04\data_pt"
results_save_path = r"C:\Users\Killua\PycharmProjects\pythonProject6\train_binary_classifier_27_04\grid_search_results.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 20

# ===========================================
# 📂 Dataset и collate_fn
# ===========================================
class GraphPairDiskDataset(torch.utils.data.Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        return (
            data["x1"], data["edge_index1"], data["batch1"],
            data["x2"], data["edge_index2"], data["batch2"],
            data["y"]
        )

def collate_fn(batch):
    x1_list, edge_index1_list, batch1_list, x2_list, edge_index2_list, batch2_list, y_list = zip(*batch)
    return (
        x1_list, edge_index1_list, batch1_list,
        x2_list, edge_index2_list, batch2_list,
        torch.cat(y_list, dim=0)
    )

# ===========================================
# 🧠 Модели
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
# 🛠 Функции обучения
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

        preds = torch.sigmoid(outputs) > 0.5
        correct += (preds.float() == y).sum().item()
        total += y.size(0)
        total_loss += loss.item()

    return total_loss / len(loader), correct / total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    for batch in loader:
        x1_list, edge_index1_list, batch1_list, x2_list, edge_index2_list, batch2_list, y = batch

        outputs = []
        for x1, edge_index1, batch1, x2, edge_index2, batch2 in zip(x1_list, edge_index1_list, batch1_list, x2_list, edge_index2_list, batch2_list):
            x1, edge_index1, batch1 = x1.to(device), edge_index1.to(device), batch1.to(device)
            x2, edge_index2, batch2 = x2.to(device), edge_index2.to(device), batch2.to(device)
            output = model(x1, edge_index1, batch1, x2, edge_index2, batch2)
            outputs.append(output)

        outputs = torch.stack(outputs).squeeze(-1)
        preds = torch.sigmoid(outputs) > 0.5
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    return all_preds, all_labels

# ===========================================
# 📂 Загрузка данных + Подсчёт классов
# ===========================================
train_dir = os.path.join(data_base_dir, "train")
val_dir = os.path.join(data_base_dir, "val")
test_dir = os.path.join(data_base_dir, "test")

train_files = sorted([os.path.join(train_dir, f) for f in os.listdir(train_dir)])
val_files = sorted([os.path.join(val_dir, f) for f in os.listdir(val_dir)])
test_files = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir)])

# Объединяем train + val
train_files += val_files

# Подсчёт классов на трейне
train_dataset = GraphPairDiskDataset(train_files)
train_loader_count = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

train_labels = []
for batch in train_loader_count:
    _, _, _, _, _, _, y = batch
    train_labels.extend(y.numpy().tolist())

train_counter = Counter(train_labels)

# Подсчёт классов на тесте
test_dataset = GraphPairDiskDataset(test_files)
test_loader_count = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

test_labels = []
for batch in test_loader_count:
    _, _, _, _, _, _, y = batch
    test_labels.extend(y.numpy().tolist())

test_counter = Counter(test_labels)

print("\n📊 Количество примеров:")
print(f"  Трейн - Класс 0: {train_counter.get(0.0, 0)}, Класс 1: {train_counter.get(1.0, 0)}")
print(f"  Тест  - Класс 0: {test_counter.get(0.0, 0)}, Класс 1: {test_counter.get(1.0, 0)}")

# ===========================================
# 📈 Сетка параметров
# ===========================================
param_grid = {
    'learning_rate': [0.001, 0.0008, 0.0006, 0.0004, 0.0002],
    'hidden_dim': [64, 96, 128, 160],
    'batch_size': [8, 16, 24, 32]
}

param_combinations = list(itertools.product(
    param_grid['learning_rate'],
    param_grid['hidden_dim'],
    param_grid['batch_size']
))

# ===========================================
# 🚀 Грид-сёрч
# ===========================================
results = []

print(f"\n🚀 Старт Грид-сёрча: всего {len(param_combinations)} комбинаций")

for idx, (lr, hidden_dim, batch_size) in enumerate(param_combinations):
    print(f"\n🔵 Комбинация {idx+1}/{len(param_combinations)}: lr={lr}, hidden_dim={hidden_dim}, batch_size={batch_size}")

    train_dataset = GraphPairDiskDataset(train_files)
    test_dataset = GraphPairDiskDataset(test_files)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    criterion = nn.BCEWithLogitsLoss()

    # --- GCN
    gcn_model = GraphSimilarityModel(GCNEncoder(3, hidden_dim), hidden_dim).to(device)
    gcn_optimizer = torch.optim.Adam(gcn_model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(gcn_model, train_loader, gcn_optimizer, criterion, device)

    preds_test_gcn, labels_test_gcn = evaluate(gcn_model, test_loader, device)
    gcn_test_acc = accuracy_score(labels_test_gcn, preds_test_gcn)
    gcn_test_precision = precision_score(labels_test_gcn, preds_test_gcn, zero_division=0)
    gcn_test_recall = recall_score(labels_test_gcn, preds_test_gcn, zero_division=0)

    # --- GIN
    gin_model = GraphSimilarityModel(GINEncoder(3, hidden_dim), hidden_dim).to(device)
    gin_optimizer = torch.optim.Adam(gin_model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(gin_model, train_loader, gin_optimizer, criterion, device)

    preds_test_gin, labels_test_gin = evaluate(gin_model, test_loader, device)
    gin_test_acc = accuracy_score(labels_test_gin, preds_test_gin)
    gin_test_precision = precision_score(labels_test_gin, preds_test_gin, zero_division=0)
    gin_test_recall = recall_score(labels_test_gin, preds_test_gin, zero_division=0)

    # --- Сохраняем всё
    results.append({
        "learning_rate": lr,
        "hidden_dim": hidden_dim,
        "batch_size": batch_size,
        "test_accuracy_gcn": gcn_test_acc,
        "test_precision_gcn": gcn_test_precision,
        "test_recall_gcn": gcn_test_recall,
        "test_accuracy_gin": gin_test_acc,
        "test_precision_gin": gin_test_precision,
        "test_recall_gin": gin_test_recall,
    })

# ===========================================
# 📊 Сохраняем в CSV
# ===========================================
df_results = pd.DataFrame(results)
df_results.to_csv(results_save_path, index=False)

print(f"\n✅ Грид-сёрч завершен! Результаты сохранены в: {results_save_path}")

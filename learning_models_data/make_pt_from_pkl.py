import os
import pickle
import torch
import random
from math import floor

# Путь к папке с исходными парами
source_dir = r"C:\Users\Killua\PycharmProjects\pythonProject6\train_binary_classifier_27_04\data_pairs"

# Путь куда сохранить обработанные .pt файлы
save_dir = r"C:\Users\Killua\PycharmProjects\pythonProject6\train_binary_classifier_27_04\data_pt"
os.makedirs(save_dir, exist_ok=True)

# Папки для train/val/test
train_dir = os.path.join(save_dir, "train")
val_dir = os.path.join(save_dir, "val")
test_dir = os.path.join(save_dir, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Считываем все папки с парами
pair_dirs = sorted(os.listdir(source_dir))
random.seed(42)
random.shuffle(pair_dirs)

# Делим
n_total = len(pair_dirs)
n_train = floor(0.8 * n_total)
n_val = floor(0.1 * n_total)
n_test = n_total - n_train - n_val

train_pairs = pair_dirs[:n_train]
val_pairs = pair_dirs[n_train:n_train+n_val]
test_pairs = pair_dirs[n_train+n_val:]

splits = {
    "train": train_pairs,
    "val": val_pairs,
    "test": test_pairs
}

# Функция конвертации одного графа
def graph_to_tensors(graph_dict):
    G = graph_dict["nx_graph"]

    if G.number_of_nodes() == 0:
        raise ValueError("Граф пустой!")

    # Узлы
    node_features = []
    node_idx_map = {}
    for idx, (node_id, attr) in enumerate(G.nodes(data=True)):
        x = attr.get("x", 0.0)
        y = attr.get("y", 0.0)
        type_str = attr.get("type", "unknown")
        type_encoding = {
            "line": 0,
            "circle": 1,
            "arc": 2
        }.get(type_str, -1)  # если неизвестный тип, кодируем как -1
        node_features.append([x, y, type_encoding])
        node_idx_map[node_id] = idx

    x = torch.tensor(node_features, dtype=torch.float)

    # Рёбра
    edge_index = []
    for u, v in G.edges():
        edge_index.append([node_idx_map[u], node_idx_map[v]])
        edge_index.append([node_idx_map[v], node_idx_map[u]])

    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Batch
    batch = torch.zeros(x.size(0), dtype=torch.long)

    return x, edge_index, batch

# Конвертация одной пары
def process_and_save(pair_folder, split_name, idx):
    pair_path = os.path.join(source_dir, pair_folder)

    with open(os.path.join(pair_path, "graph_a.pkl"), "rb") as f:
        graph_a = pickle.load(f)

    with open(os.path.join(pair_path, "graph_b.pkl"), "rb") as f:
        graph_b = pickle.load(f)

    with open(os.path.join(pair_path, "label.txt"), "r") as f:
        label = int(f.read().strip())

    # Конвертация графов
    x1, edge_index1, batch1 = graph_to_tensors(graph_a)
    x2, edge_index2, batch2 = graph_to_tensors(graph_b)

    pair_data = {
        "x1": x1, "edge_index1": edge_index1, "batch1": batch1,
        "x2": x2, "edge_index2": edge_index2, "batch2": batch2,
        "y": torch.tensor([label], dtype=torch.float)
    }

    save_subdir = os.path.join(save_dir, split_name)
    os.makedirs(save_subdir, exist_ok=True)
    save_path = os.path.join(save_subdir, f"pair_{idx:04d}.pt")
    torch.save(pair_data, save_path)

# Запуск с обработкой ошибок
good_counts = {"train": 0, "val": 0, "test": 0}
bad_counts = {"train": 0, "val": 0, "test": 0}

for split_name, pairs in splits.items():
    idx_good = 0
    for pair_folder in pairs:
        try:
            process_and_save(pair_folder, split_name, idx_good)
            idx_good += 1
            good_counts[split_name] += 1
        except Exception as e:
            print(f"❌ Пропущена пара {pair_folder} ({split_name}): {e}")
            bad_counts[split_name] += 1

# Вывод итогов
print("\n✅ Конвертация завершена!")
for split in ['train', 'val', 'test']:
    print(f"👉 {split}: сохранено {good_counts[split]}, пропущено {bad_counts[split]}")

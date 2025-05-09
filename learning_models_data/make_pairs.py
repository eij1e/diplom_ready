import itertools
import random
from pathlib import Path
from made_pkl_from_dxf import parse_and_visualize_primitives_dxf  # Импортируй свою функцию здесь

# Путь к исходным данным
base_dxf_dir = Path(r"C:\Users\Killua\PycharmProjects\pythonProject6\train_binary_classifier_27_04\data")
output_pairs_dir = Path(r"C:\Users\Killua\PycharmProjects\pythonProject6\train_binary_classifier_27_04\data_pairs")

# Убедись, что папка для пар существует
output_pairs_dir.mkdir(parents=True, exist_ok=True)

# Счётчик пар
pair_counter = 1

# Словарь: имя папки -> список файлов
folder_files = {}

# Проход по подпапкам (ex1, ex2, ...)
for subfolder in base_dxf_dir.iterdir():
    if not subfolder.is_dir():
        continue
    dxf_files = list(subfolder.glob("*.dxf"))
    if dxf_files:
        folder_files[subfolder.name] = dxf_files

# Список всех папок
folder_names = list(folder_files.keys())

# ======= Создаём похожие пары =======
positive_pairs = []

for folder, files in folder_files.items():
    for file_a, file_b in itertools.combinations(files, 2):
        positive_pairs.append((file_a, file_b, 1))  # метка 1 — похожие

print(f"🔵 Похожих пар: {len(positive_pairs)}")

# Обработаем похожие пары
for file_a, file_b, label in positive_pairs:
    pair_name = f"pair_{pair_counter}"
    pair_dir = output_pairs_dir / pair_name
    pair_dir.mkdir(parents=True, exist_ok=True)

    graph_a_pkl = pair_dir / "graph_a.pkl"
    graph_b_pkl = pair_dir / "graph_b.pkl"
    graph_a_png = pair_dir / "graph_a.png"
    graph_b_png = pair_dir / "graph_b.png"

    parse_and_visualize_primitives_dxf(
        str(file_a),
        output_png_path=str(graph_a_png),
        output_pkl_path=str(graph_a_pkl),
        image_width=1600,
        image_height=1200,
        dpi=100
    )
    parse_and_visualize_primitives_dxf(
        str(file_b),
        output_png_path=str(graph_b_png),
        output_pkl_path=str(graph_b_pkl),
        image_width=1600,
        image_height=1200,
        dpi=100
    )

    # Создание label.txt в папке пары
    (pair_dir / "label.txt").write_text(str(label))

    print(f"✅ Похожая пара: {pair_name}")
    pair_counter += 1

# ======= Создаём непохожие пары =======
negative_pairs = []
positive_count = len(positive_pairs)
used_negative_pairs = set()

while len(negative_pairs) < positive_count:
    folder1, folder2 = random.sample(folder_names, 2)
    file_a = random.choice(folder_files[folder1])
    file_b = random.choice(folder_files[folder2])

    pair_key = frozenset([file_a.stem, file_b.stem])

    if pair_key in used_negative_pairs:
        continue

    used_negative_pairs.add(pair_key)
    negative_pairs.append((file_a, file_b, 0))

print(f"🔴 Непохожих пар: {len(negative_pairs)}")

# Обработаем непохожие пары
for file_a, file_b, label in negative_pairs:
    pair_name = f"pair_{pair_counter}"
    pair_dir = output_pairs_dir / pair_name
    pair_dir.mkdir(parents=True, exist_ok=True)

    graph_a_pkl = pair_dir / "graph_a.pkl"
    graph_b_pkl = pair_dir / "graph_b.pkl"
    graph_a_png = pair_dir / "graph_a.png"
    graph_b_png = pair_dir / "graph_b.png"

    parse_and_visualize_primitives_dxf(
        str(file_a),
        output_png_path=str(graph_a_png),
        output_pkl_path=str(graph_a_pkl),
        image_width=1600,
        image_height=1200,
        dpi=100
    )
    parse_and_visualize_primitives_dxf(
        str(file_b),
        output_png_path=str(graph_b_png),
        output_pkl_path=str(graph_b_pkl),
        image_width=1600,
        image_height=1200,
        dpi=100
    )

    # Создание label.txt в папке пары
    (pair_dir / "label.txt").write_text(str(label))

    print(f"❌ Непохожая пара: {pair_name}")
    pair_counter += 1

print("🎯 Разметка завершена!")

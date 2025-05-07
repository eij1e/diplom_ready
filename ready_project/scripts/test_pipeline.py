import os
import torch
from core.convert_dxf_graphs import process_dxf_to_pt
from core.comparer_graphs import compare_graphs_and_store
from core.models import GCNEncoder, GINEncoder, GraphSimilarityModel
from app.database import SessionLocal, VersionInfo
from packaging import version as vparse
from datetime import datetime

# --- Вычисляем директории
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # .../ready_project/scripts
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))  # .../ready_project

# --- Автоинкремент версии
def get_next_version():
    session = SessionLocal()
    versions = session.query(VersionInfo.version).all()
    session.close()
    if not versions:
        return "1.0.0"
    latest = sorted([vparse.parse(ver[0]) for ver in versions])[-1]
    next_version = f"{latest.major}.{latest.minor}.{latest.micro + 1}"
    return next_version

version = get_next_version()

# Сохраняем версию в базу, если её ещё нет
def save_version_if_needed(version: str):
    session = SessionLocal()
    exists = session.query(VersionInfo).filter(VersionInfo.version == version).first()
    if not exists:
        new_version = VersionInfo(version=version, created_at=datetime.utcnow())
        session.add(new_version)
        session.commit()
    session.close()

save_version_if_needed(version)

print(f"🔢 Используется версия сравнения: v{version}")

# --- Пути
input_dxf_dir = os.path.join(BASE_DIR, "..", "train_binary_classifier_27_04", "validation_data", "dxf")
output_pkl_dir = os.path.join(BASE_DIR, "test", "pkl")
output_pt_dir = os.path.join(BASE_DIR, "test", "pt")
image_save_dir = os.path.join(BASE_DIR, "static", "images")
gcn_model_path = os.path.join(BASE_DIR, "models", "best_model_gcn.pt")
gin_model_path = os.path.join(BASE_DIR, "models", "best_model_gin.pt")

# --- Конвертация DXF → PKL, PT, PNG
process_dxf_to_pt(
    input_dxf_dir=input_dxf_dir,
    output_pkl_dir=output_pkl_dir,
    output_pt_dir=output_pt_dir,
    image_save_dir=image_save_dir
)

# --- Загрузка моделей
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gcn_model = GraphSimilarityModel(GCNEncoder(3, 64), 64).to(device)
gcn_model.load_state_dict(torch.load(gcn_model_path, map_location=device))
gcn_model.eval()

gin_model = GraphSimilarityModel(GINEncoder(3, 64), 64).to(device)
gin_model.load_state_dict(torch.load(gin_model_path, map_location=device))
gin_model.eval()

# --- Сравнение графов
compare_graphs_and_store(
    pkl_dir=output_pkl_dir,
    pt_dir=output_pt_dir,
    gcn_model=gcn_model,
    gin_model=gin_model,
    version=version,
    image_dir=image_save_dir
)

print("\n✅ Тестовый запуск пайплайна завершён.")

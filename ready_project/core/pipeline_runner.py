import os
import torch
from core.convert_dxf_graphs import process_dxf_to_pt
from core.comparer_graphs import compare_graphs_and_store
from core.models import GCNEncoder, GINEncoder, GraphSimilarityModel
from app.database import SessionLocal, VersionInfo
from packaging import version as vparse
from datetime import datetime

def get_next_version():
    session = SessionLocal()
    versions = session.query(VersionInfo.version).all()
    session.close()
    if not versions:
        return "1.0.0"
    latest = sorted([vparse.parse(ver[0]) for ver in versions])[-1]
    return f"{latest.major}.{latest.minor}.{latest.micro + 1}"

def save_version_if_needed(version):
    session = SessionLocal()
    exists = session.query(VersionInfo).filter(VersionInfo.version == version).first()
    if not exists:
        session.add(VersionInfo(version=version, created_at=datetime.utcnow()))
        session.commit()
    session.close()

def run_pipeline(selected_dxf_files):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    version = get_next_version()
    save_version_if_needed(version)

    version_dir = os.path.join(base_dir, "data", version.replace(".", "_"))
    os.makedirs(version_dir, exist_ok=True)
    input_dir = os.path.join(version_dir, "dxf")
    os.makedirs(input_dir, exist_ok=True)

    for file_path in selected_dxf_files:
        filename = os.path.basename(file_path)
        dest = os.path.join(input_dir, filename)
        with open(file_path, "rb") as src, open(dest, "wb") as dst:
            dst.write(src.read())

    output_pkl_dir = os.path.join(base_dir, "data", version.replace(".", "_"), "pkl")
    output_pt_dir = os.path.join(base_dir, "data", version.replace(".", "_"), "pt")
    image_dir = os.path.join(base_dir, "static", "images")
    os.makedirs(image_dir, exist_ok=True)

    process_dxf_to_pt(
        input_dxf_dir=input_dir,
        output_pkl_dir=output_pkl_dir,
        output_pt_dir=output_pt_dir,
        image_save_dir=image_dir
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gcn_model_path = os.path.join(base_dir, "models", "best_model_gcn.pt")
    gin_model_path = os.path.join(base_dir, "models", "best_model_gin.pt")

    gcn_model = GraphSimilarityModel(GCNEncoder(3, 64), 64).to(device)
    gcn_model.load_state_dict(torch.load(gcn_model_path, map_location=device))
    gcn_model.eval()

    gin_model = GraphSimilarityModel(GINEncoder(3, 64), 64).to(device)
    gin_model.load_state_dict(torch.load(gin_model_path, map_location=device))
    gin_model.eval()

    compare_graphs_and_store(
        pkl_dir=output_pkl_dir,
        pt_dir=output_pt_dir,
        gcn_model=gcn_model,
        gin_model=gin_model,
        version=version,
        image_dir=image_dir
    )

    print(f"✅ Сравнение завершено для версии: {version}")

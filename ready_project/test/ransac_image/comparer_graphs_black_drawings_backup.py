import os
import torch
import numpy as np
import pickle
import cv2
from tqdm import tqdm
from scipy.spatial.distance import cdist
from app.database import ComparisonResult, SessionLocal
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
matplotlib.use("Agg")  # отключает интерактивный GUI backend


# --- Примитивы ---
def extract_primitives(G):
    primitives = []
    for node, data in G.nodes(data=True):
        if data.get("type") in {"circle", "arc"}:
            radius = data.get("radius", 0.0)
            angle_span = data.get("end_angle", 0.0) - data.get("start_angle", 0.0)
            primitives.append(([data.get("x", 0.0), data.get("y", 0.0)], [radius, angle_span]))
    for u, v, data in G.edges(data=True):
        if data.get("type") in {"line", "arc", "polyline", "spline", "circle-edge"}:
            center = [(u[0] + v[0]) / 2, (u[1] + v[1]) / 2]
            dx, dy = v[0] - u[0], v[1] - u[1]
            length = np.hypot(dx, dy)
            angle = np.arctan2(dy, dx)
            primitives.append((center, [length, angle]))
    return primitives

def normalize_descriptors(primitives):
    lengths = np.array([desc[0] for _, desc in primitives])
    scale_factor = np.mean(lengths) if np.mean(lengths) != 0 else 1.0
    normalized = [(center, [desc[0] / scale_factor, desc[1]]) for center, desc in primitives]
    return normalized

def ransac_score(A, B, threshold):
    if not A or not B:
        return 0.0, []

    A_coords = np.array([desc for _, desc in A])
    B_coords = np.array([desc for _, desc in B])
    distances = cdist(A_coords, B_coords)

    matches = [(i, np.argmin(row)) for i, row in enumerate(distances) if np.min(row) < threshold]
    src_pts = np.array([A[i][0] for i, _ in matches], dtype=np.float32)
    dst_pts = np.array([B[j][0] for _, j in matches], dtype=np.float32)

    if len(src_pts) >= 3:
        M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
        if inliers is not None:
            num_inliers = int(np.sum(inliers))
            inlier_matches = [matches[i] for i in range(len(inliers)) if inliers[i]]
            return num_inliers / max(len(A), len(B)), inlier_matches
    return 0.0, []

def predict_ransac(G1, G2, threshold=10.0, alpha=0.7, return_matches=False):
    A = extract_primitives(G1)
    B = extract_primitives(G2)
    A_norm = normalize_descriptors(A)
    B_norm = normalize_descriptors(B)

    score_raw, matches_raw = ransac_score(A, B, threshold)
    score_norm, matches_norm = ransac_score(A_norm, B_norm, threshold)

    use_norm = score_norm >= score_raw
    score = alpha * max(score_norm, score_raw) + (1 - alpha) * min(score_norm, score_raw)
    matches = matches_norm if use_norm else matches_raw

    return (score, matches, use_norm) if return_matches else score

# --- Визуализация двух графов без совпадений ---
def draw_ransac_match(G1, G2, image_path, graph_name_a, graph_name_b, use_norm=False):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    ax1, ax2 = axs

    def draw(ax, G, title=None):
        for u, v, data in G.edges(data=True):
            if data.get("type") in {"line", "arc", "polyline", "spline", "circle-edge"}:
                ax.plot([u[0], v[0]], [u[1], v[1]], color="black", linewidth=1)

        for node, data in G.nodes(data=True):
            x, y = data.get("x"), data.get("y")
            if x is None or y is None:
                continue

            if data.get("type") == "circle":
                r = data.get("radius", 0)
                circle = patches.Circle((x, y), r, fill=False, edgecolor="black", linewidth=1.5)
                ax.add_patch(circle)
            elif data.get("type") == "arc":
                r = data.get("radius", 0)
                theta1 = np.rad2deg(data.get("start_angle", 0))
                theta2 = np.rad2deg(data.get("end_angle", 0))
                arc = patches.Arc((x, y), 2 * r, 2 * r, angle=0,
                                  theta1=theta1, theta2=theta2,
                                  edgecolor="black", linewidth=1.5)
                ax.add_patch(arc)

        if title:
            ax.set_title(title)
        ax.set_aspect("equal")
        ax.axis("off")

    draw(ax1, G1, graph_name_a)
    draw(ax2, G2, graph_name_b)

    plt.tight_layout()
    plt.savefig(image_path, dpi=150)
    plt.close()

# --- Предсказание через модель ---
def predict_similarity(model, data1, data2, device):
    with torch.no_grad():
        x1, edge_index1, batch1 = data1["x1"].to(device), data1["edge_index1"].to(device), data1["batch1"].to(device)
        x2, edge_index2, batch2 = data2["x2"].to(device), data2["edge_index2"].to(device), data2["batch2"].to(device)
        output = model(x1, edge_index1, batch1, x2, edge_index2, batch2)
        return int(torch.sigmoid(output).item() > 0.5)

# --- Основная функция сравнения и записи в БД ---
def compare_graphs_and_store(pkl_dir, pt_dir, gcn_model, gin_model, version, image_dir=None, use_ransac=True, use_gcn=True, use_gin=True, threshold=10.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    session = SessionLocal()

    pkl_files = sorted([f for f in os.listdir(pkl_dir) if f.endswith(".pkl")])

    for i in tqdm(range(len(pkl_files))):
        for j in range(i + 1, len(pkl_files)):
            name_a = os.path.splitext(pkl_files[i])[0]
            name_b = os.path.splitext(pkl_files[j])[0]

            path_a = os.path.join(pkl_dir, pkl_files[i])
            path_b = os.path.join(pkl_dir, pkl_files[j])
            G1 = pickle.load(open(path_a, "rb"))["nx_graph"]
            G2 = pickle.load(open(path_b, "rb"))["nx_graph"]

            result = ComparisonResult(
                graph_a=name_a,
                graph_b=name_b,
                version=version,
                created_at=datetime.utcnow(),
                image_path=""
            )

            if use_ransac:
                score, _, _ = predict_ransac(G1, G2, threshold, return_matches=True)
                result.ransac_predict = score

                if image_dir:
                    os.makedirs(image_dir, exist_ok=True)
                    image_filename = f"{name_a}__{name_b}__v{version.replace('.', '_')}.png"
                    image_path = os.path.join(image_dir, image_filename)
                    draw_ransac_match(G1, G2, image_path=image_path, graph_name_a=name_a, graph_name_b=name_b)
                    result.image_path = os.path.join("static", "images", image_filename).replace("\\", "/")


            data1 = torch.load(os.path.join(pt_dir, f"{name_a}.pt"))
            data2 = torch.load(os.path.join(pt_dir, f"{name_b}.pt"))

            if use_gcn and gcn_model:
                result.gcn_predict = predict_similarity(gcn_model, data1, data2, device)

            if use_gin and gin_model:
                result.gin_predict = predict_similarity(gin_model, data1, data2, device)

            session.add(result)

    session.commit()
    from core.statistics import compute_statistics_for_version

    compute_statistics_for_version(session, version=version, image_dir="static/histograms")
    session.close()
    print("✅ Все сравнения сохранены в базу данных.")

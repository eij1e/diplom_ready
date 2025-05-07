import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from scipy.spatial.distance import cdist
from pathlib import Path

# --- Настройка путей ---
graph_path_a = r"C:\Users\Killua\PycharmProjects\pythonProject6\ready_project\test\pkl\ex16_changed2.pkl"
graph_path_b = r"C:\Users\Killua\PycharmProjects\pythonProject6\ready_project\test\pkl\ex16_replace_rotate.pkl"
image_output_path = r"C:/Users/Killua/PycharmProjects/pythonProject6/ready_project/test/ransac_image/kek.png"


def load_graph(path):
    with open(path, "rb") as f:
        return pickle.load(f)["nx_graph"]

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

    return (score, matches, use_norm, A, B) if return_matches else score

def draw_ransac_match(G1, G2, image_path, graph_name_a, graph_name_b, matches=None, A=None, B=None):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    ax1, ax2 = axs

    matched_a = [tuple(np.round(A[i][0], 4)) for i, _ in matches] if matches else []
    matched_b = [tuple(np.round(B[j][0], 4)) for _, j in matches] if matches else []

    def draw(ax, G, title=None, matched_centers=None):
        for u, v, data in G.edges(data=True):
            if data.get("type") in {"line", "arc", "polyline", "spline", "circle-edge"}:
                center = ((u[0] + v[0]) / 2, (u[1] + v[1]) / 2)
                center_rounded = tuple(np.round(center, 4))
                col = "green" if matched_centers and center_rounded in matched_centers else "red"
                ax.plot([u[0], v[0]], [u[1], v[1]], color=col, linewidth=1)

        for node, data in G.nodes(data=True):
            x, y = data.get("x"), data.get("y")
            if x is None or y is None:
                continue

            center_rounded = tuple(np.round([x, y], 4))
            col = "green" if matched_centers and center_rounded in matched_centers else "red"

            if data.get("type") == "circle":
                r = data.get("radius", 0)
                circle = patches.Circle((x, y), r, fill=False, edgecolor=col, linewidth=1.5)
                ax.add_patch(circle)
            elif data.get("type") == "arc":
                r = data.get("radius", 0)
                theta1 = np.rad2deg(data.get("start_angle", 0))
                theta2 = np.rad2deg(data.get("end_angle", 0))
                arc = patches.Arc((x, y), 2 * r, 2 * r, angle=0,
                                  theta1=theta1, theta2=theta2,
                                  edgecolor=col, linewidth=1.5)
                ax.add_patch(arc)

        if title:
            ax.set_title(title)
        ax.set_aspect("equal")
        ax.axis("off")

    draw(ax1, G1, graph_name_a, matched_a)
    draw(ax2, G2, graph_name_b, matched_b)

    plt.tight_layout()
    plt.savefig(image_path, dpi=150)
    plt.close()

# --- Основной вызов без argparse ---
G1 = load_graph(graph_path_a)
G2 = load_graph(graph_path_b)

score, matches, use_norm, A, B = predict_ransac(G1, G2, return_matches=True)

draw_ransac_match(
    G1, G2,
    image_path=image_output_path,
    graph_name_a=Path(graph_path_a).stem,
    graph_name_b=Path(graph_path_b).stem,
    matches=matches,
    A=A,
    B=B
)

print(f"✅ Сравнение завершено. Score = {score:.3f}")
print(f"📷 Визуализация сохранена: {image_output_path}")

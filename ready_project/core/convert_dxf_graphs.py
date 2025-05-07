import os
import ezdxf
import networkx as nx
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pickle
from typing import Optional
import matplotlib
matplotlib.use("Agg")  # отключает интерактивный GUI backend

def process_dxf_to_pt(
    input_dxf_dir: str,
    output_pkl_dir: str,
    output_pt_dir: str,
    image_save_dir: Optional[str] = None,
    dpi: int = 100
):
    os.makedirs(output_pkl_dir, exist_ok=True)
    os.makedirs(output_pt_dir, exist_ok=True)
    if image_save_dir:
        os.makedirs(image_save_dir, exist_ok=True)

    def parse_and_save(dxf_path, pkl_save_path, png_save_path=None):
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
        G = nx.Graph()

        def add_circle_node(x, y, radius):
            node_id = f"circle_{x:.6f}_{y:.6f}_{radius:.6f}"
            G.add_node(node_id, x=x, y=y, type='circle', radius=radius)

        def add_arc_node(x, y, radius, start_angle, end_angle):
            node_id = f"arc_{x:.6f}_{y:.6f}_{radius:.6f}_{start_angle:.4f}_{end_angle:.4f}"
            G.add_node(node_id, x=x, y=y, type='arc',
                       radius=radius,
                       start_angle=start_angle,
                       end_angle=end_angle)

        def add_line_nodes(p1, p2):
            p1 = (round(p1[0], 6), round(p1[1], 6))
            p2 = (round(p2[0], 6), round(p2[1], 6))
            G.add_node(p1, x=p1[0], y=p1[1], type='line')
            G.add_node(p2, x=p2[0], y=p2[1], type='line')
            G.add_edge(p1, p2, type='line')

        for e in msp:
            t = e.dxftype()
            if t == 'LINE':
                start = e.dxf.start
                end = e.dxf.end
                add_line_nodes(start, end)
            elif t == 'CIRCLE':
                center = e.dxf.center
                radius = e.dxf.radius
                add_circle_node(center.x, center.y, radius)
            elif t == 'ARC':
                center = e.dxf.center
                radius = e.dxf.radius
                start_angle_deg = e.dxf.start_angle
                end_angle_deg = e.dxf.end_angle
                if end_angle_deg < start_angle_deg:
                    end_angle_deg += 360
                angle_span = end_angle_deg - start_angle_deg
                if abs(angle_span - 360) < 1e-1:
                    add_circle_node(center.x, center.y, radius)
                else:
                    add_arc_node(center.x, center.y, radius,
                                 np.deg2rad(start_angle_deg),
                                 np.deg2rad(end_angle_deg))

        if png_save_path:
            fig, ax = plt.subplots(figsize=(8, 8))
            all_x, all_y = [], []

            for u, v, data in G.edges(data=True):
                if data.get('type') == 'line':
                    p1 = G.nodes[u]
                    p2 = G.nodes[v]
                    xs = [p1['x'], p2['x']]
                    ys = [p1['y'], p2['y']]
                    ax.plot(xs, ys, color='gray', linewidth=1)
                    all_x.extend(xs)
                    all_y.extend(ys)

            for node_id, data in G.nodes(data=True):
                x, y = data.get("x"), data.get("y")
                if data.get('type') == 'circle':
                    r = data.get('radius', 0)
                    circle = patches.Circle((x, y), r, fill=False, edgecolor='blue', linewidth=1.5)
                    ax.add_patch(circle)
                    all_x.extend([x - r, x + r])
                    all_y.extend([y - r, y + r])
                elif data.get('type') == 'arc':
                    r = data.get('radius', 0)
                    start = np.rad2deg(data.get('start_angle', 0))
                    end = np.rad2deg(data.get('end_angle', 0))
                    arc = patches.Arc((x, y), 2 * r, 2 * r, angle=0, theta1=start, theta2=end,
                                      edgecolor='green', linewidth=1.5)
                    ax.add_patch(arc)
                    all_x.extend([x - r, x + r])
                    all_y.extend([y - r, y + r])

            if all_x and all_y:
                margin = 20
                ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
                ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

            ax.set_aspect('equal')
            ax.axis('off')
            plt.savefig(png_save_path, dpi=dpi)
            plt.close()

        with open(pkl_save_path, "wb") as f:
            pickle.dump({"nx_graph": G}, f)

    for file in os.listdir(input_dxf_dir):
        if file.endswith(".dxf"):
            dxf_path = os.path.join(input_dxf_dir, file)
            filename_wo_ext = os.path.splitext(file)[0]
            pkl_save_path = os.path.join(output_pkl_dir, f"{filename_wo_ext}.pkl")
            png_save_path = os.path.join(image_save_dir, f"{filename_wo_ext}.png") if image_save_dir else None
            parse_and_save(dxf_path, pkl_save_path, png_save_path)

    print(f"✅ Сконвертированы все DXF -> PKL")

    def graph_to_tensors(graph_dict):
        G = graph_dict["nx_graph"]
        node_features = []
        node_idx_map = {}

        for idx, (node_id, attr) in enumerate(G.nodes(data=True)):
            x = attr.get("x", 0.0)
            y = attr.get("y", 0.0)
            type_str = attr.get("type", "unknown")
            type_encoding = {"line": 0, "circle": 1, "arc": 2}.get(type_str, -1)
            node_features.append([x, y, type_encoding])
            node_idx_map[node_id] = idx

        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = []
        for u, v in G.edges():
            edge_index.append([node_idx_map[u], node_idx_map[v]])
            edge_index.append([node_idx_map[v], node_idx_map[u]])

        if len(edge_index) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        batch = torch.zeros(x.size(0), dtype=torch.long)
        return x, edge_index, batch

    for file in sorted(os.listdir(output_pkl_dir)):
        if file.endswith(".pkl"):
            filename_wo_ext = os.path.splitext(file)[0]
            with open(os.path.join(output_pkl_dir, file), "rb") as f:
                graph = pickle.load(f)
            x, edge_index, batch = graph_to_tensors(graph)
            data = {
                "x1": x,
                "edge_index1": edge_index,
                "batch1": batch,
                "x2": x,
                "edge_index2": edge_index,
                "batch2": batch,
                "y": torch.tensor([1], dtype=torch.float)
            }
            save_path = os.path.join(output_pt_dir, f"{filename_wo_ext}.pt")
            torch.save(data, save_path)

    print(f"✅ Сконвертированы все PKL -> PT")

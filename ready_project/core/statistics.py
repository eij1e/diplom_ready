import os
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy.orm import Session
from app.database import ComparisonResult, Statistics
import matplotlib
matplotlib.use("Agg")  # отключает интерактивный GUI backend


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTOGRAM_DIR = os.path.join(BASE_DIR, "static", "histograms")
os.makedirs(HISTOGRAM_DIR, exist_ok=True)

def compute_statistics_for_version(db: Session, version: str, image_dir: str):
    records = db.query(ComparisonResult).filter(ComparisonResult.version == version).all()
    if not records:
        return

    ransac_scores = [r.ransac_predict for r in records if r.ransac_predict is not None]
    gcn_count = sum(1 for r in records if r.gcn_predict == 1)
    gin_count = sum(1 for r in records if r.gin_predict == 1)
    ransac_above_0_5 = sum(1 for r in ransac_scores if r > 0.5)

    perc75 = float(np.percentile(ransac_scores, 75)) if ransac_scores else 0.0
    perc90 = float(np.percentile(ransac_scores, 90)) if ransac_scores else 0.0

    # --- Построение и сохранение гистограммы
    os.makedirs(image_dir, exist_ok=True)
    filename = f"histogram_v{version.replace('.', '_')}.png"
    hist_path = os.path.join(HISTOGRAM_DIR, filename)
    hist_relative = f"static/histograms/{filename}"

    import seaborn as sns

    plt.figure(figsize=(9, 4.5))
    sns.histplot(ransac_scores, bins=20, kde=True, color="cornflowerblue", edgecolor="black")
    plt.axvline(perc75, color='orange', linestyle='--', label='75-й перцентиль')
    plt.axvline(perc90, color='red', linestyle='--', label='90-й перцентиль')
    plt.title(f"RANSAC Score — v{version}", fontsize=14)
    plt.xlabel("RANSAC Score")
    plt.ylabel("Частота")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(hist_path)
    plt.close()

    # --- Запись в базу
    stat = Statistics(
        version=version,
        ransac_count_over_0_5=ransac_above_0_5,
        gcn_positive=gcn_count,
        gin_positive=gin_count,
        ransac_percentile_75=perc75,
        ransac_percentile_90=perc90,
        histogram_path=hist_relative
    )
    db.add(stat)
    db.commit()

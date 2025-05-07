from app.database import SessionLocal, ComparisonResult, Statistics

session = SessionLocal()
results = session.query(ComparisonResult).all()

#
for r in results:
    print(f"{r.graph_a} vs {r.graph_b}: RANSAC={r.ransac_predict:.2f}, GCN={r.gcn_predict}, GIN={r.gin_predict}")

# for r in results:
#     print(f"ransac_count_over_0_5: {r.ransac_count_over_0_5}\n"
#           f"gcn_positive: {r.gcn_positive}: "
#           f"RANSAC_75_perc ={r.ransac_percentile_75:.2f}"
#           f"ransac_percentile_90={r.ransac_percentile_90}, "
#           f"histogram_path={r.histogram_path}")

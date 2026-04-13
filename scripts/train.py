"""
PatrolIQ — ML Training Pipeline
Trains K-Means, DBSCAN, Hierarchical clustering + PCA + t-SNE.
All runs tracked in MLflow.
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore")

CLEAN_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned")
MLFLOW_URI = os.path.join(os.path.dirname(__file__), "..", "mlruns")
EXPERIMENT  = "PatrolIQ_Crime_Analysis"

GEO_SAMPLE  = 50_000   # for geo clustering
ML_SAMPLE   = 15_000   # for PCA / t-SNE

# ── helpers ───────────────────────────────────────────────────────────────────

def load_clean():
    path = os.path.join(CLEAN_DIR, "cleaned_crimes.csv")
    print(f"[train] Loading {path} …")
    df = pd.read_csv(path, low_memory=False)
    print(f"[train] Loaded shape: {df.shape}")
    return df


def cluster_stats(df, label_col, geo_sample_df=None):
    """Return per-cluster summary: crimes, top_type, arrest_rate."""
    src = geo_sample_df if geo_sample_df is not None else df
    src = src.copy()
    src["_label"] = src[label_col]
    stats = (src.groupby("_label").agg(
        total_crimes=("_label", "count"),
        arrest_rate=("Arrest", "mean"),
        top_crime=("Primary Type", lambda x: x.value_counts().index[0]),
    ).rename_axis("Cluster").reset_index())
    stats["arrest_rate"] = (stats["arrest_rate"] * 100).round(1)
    return stats


def save_json(obj, name):
    with open(os.path.join(CLEAN_DIR, name), "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[train] Saved {name}")


# ── Geographic Clustering ─────────────────────────────────────────────────────

def geo_clustering(df):
    print("\n[train] ── Geographic Clustering ────────────────────────────")
    geo = df[["Latitude", "Longitude", "Arrest", "Primary Type", "Crime_Severity_Score"]].dropna()
    geo_sample = geo.sample(n=min(GEO_SAMPLE, len(geo)), random_state=42).reset_index(drop=True)
    X = geo_sample[["Latitude", "Longitude"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    metrics = {}

    # ── K-Means ──────────────────────────────────────────────────────────
    with mlflow.start_run(run_name="KMeans_Geographic"):
        k = 8
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels, sample_size=5000, random_state=42)
        db  = davies_bouldin_score(X_scaled, labels)

        mlflow.log_params({"algorithm": "KMeans", "n_clusters": k, "type": "geographic"})
        mlflow.log_metrics({"silhouette_score": round(sil, 4), "davies_bouldin": round(db, 4)})
        mlflow.sklearn.log_model(km, "kmeans_geo_model")

        geo_sample["KMeans_Geo"] = labels
        centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_),
                               columns=["Latitude", "Longitude"])
        centers.to_csv(os.path.join(CLEAN_DIR, "kmeans_geo_centers.csv"), index=False)
        print(f"[train] K-Means Geo → sil={sil:.4f}, DB={db:.4f}")
        metrics["KMeans_Geo"] = {"silhouette": round(sil,4), "davies_bouldin": round(db,4), "n_clusters": k}
        mlflow.log_dict(metrics["KMeans_Geo"], "kmeans_geo_metrics.json")

    # ── DBSCAN ───────────────────────────────────────────────────────────
    with mlflow.start_run(run_name="DBSCAN_Geographic"):
        eps, min_s = 0.15, 10
        db_model = DBSCAN(eps=eps, min_samples=min_s)
        labels_db = db_model.fit_predict(X_scaled)
        n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)
        noise_pct = round(100 * (labels_db == -1).sum() / len(labels_db), 1)

        # silhouette only if > 1 real cluster
        if n_clusters_db > 1:
            mask = labels_db != -1
            sil_db = silhouette_score(X_scaled[mask], labels_db[mask], sample_size=5000, random_state=42)
            db_score = davies_bouldin_score(X_scaled[mask], labels_db[mask])
        else:
            sil_db, db_score = 0.0, 0.0

        mlflow.log_params({"algorithm": "DBSCAN", "eps": eps, "min_samples": min_s, "type": "geographic"})
        mlflow.log_metrics({"silhouette_score": round(sil_db,4), "davies_bouldin": round(db_score,4),
                            "n_clusters": n_clusters_db, "noise_pct": noise_pct})

        geo_sample["DBSCAN_Geo"] = labels_db
        print(f"[train] DBSCAN Geo → clusters={n_clusters_db}, noise={noise_pct}%, sil={sil_db:.4f}")
        metrics["DBSCAN_Geo"] = {"silhouette": round(sil_db,4), "davies_bouldin": round(db_score,4),
                                  "n_clusters": n_clusters_db, "noise_pct": noise_pct}

    # ── Hierarchical (Agglomerative) ──────────────────────────────────────
    with mlflow.start_run(run_name="Hierarchical_Geographic"):
        # Use smaller sample for dendrogram
        hier_sample = X_scaled[:3000]
        k_hier = 8
        hier = AgglomerativeClustering(n_clusters=k_hier, linkage="ward")
        labels_hier_small = hier.fit_predict(hier_sample)

        # Full prediction on geo_sample
        hier_full = AgglomerativeClustering(n_clusters=k_hier, linkage="ward")
        labels_hier = hier_full.fit_predict(X_scaled)
        sil_h = silhouette_score(X_scaled, labels_hier, sample_size=5000, random_state=42)
        db_h  = davies_bouldin_score(X_scaled, labels_hier)

        mlflow.log_params({"algorithm": "Hierarchical", "n_clusters": k_hier, "linkage": "ward", "type": "geographic"})
        mlflow.log_metrics({"silhouette_score": round(sil_h,4), "davies_bouldin": round(db_h,4)})

        geo_sample["Hierarchical_Geo"] = labels_hier

        # Dendrogram on 500 samples
        Z = linkage(X_scaled[:500], method="ward")
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.set_facecolor("#0e1117")
        fig.patch.set_facecolor("#0e1117")
        dendrogram(Z, ax=ax, truncate_mode="lastp", p=30,
                   leaf_font_size=8, color_threshold=None,
                   above_threshold_color="#4ade80")
        ax.set_title("Hierarchical Clustering Dendrogram — Chicago Crime Hotspots",
                     color="white", fontsize=13, fontweight="bold")
        ax.tick_params(colors="white")
        ax.spines[["top","right","left","bottom"]].set_color("#374151")
        plt.tight_layout()
        dend_path = os.path.join(CLEAN_DIR, "dendrogram_geo.png")
        plt.savefig(dend_path, dpi=120, bbox_inches="tight",
                    facecolor="#0e1117")
        plt.close()
        mlflow.log_artifact(dend_path)

        print(f"[train] Hierarchical Geo → sil={sil_h:.4f}, DB={db_h:.4f}")
        metrics["Hierarchical_Geo"] = {"silhouette": round(sil_h,4), "davies_bouldin": round(db_h,4), "n_clusters": k_hier}

    # Save metrics + cluster stats
    save_json(metrics, "geo_clustering_metrics.json")
    geo_sample.to_csv(os.path.join(CLEAN_DIR, "geo_sample.csv"), index=False)

    # Per-cluster stats for each algorithm
    for algo in ["KMeans_Geo", "DBSCAN_Geo", "Hierarchical_Geo"]:
        stats = cluster_stats(geo_sample, algo)
        stats.to_csv(os.path.join(CLEAN_DIR, f"{algo.lower()}_stats.csv"), index=False)

    return geo_sample


# ── Temporal Clustering ───────────────────────────────────────────────────────

def temporal_clustering(df):
    print("\n[train] ── Temporal Clustering ──────────────────────────────")
    temp = df[["Hour", "Day_Num", "Month", "Crime_Severity_Score", "Is_Weekend", "Arrest", "Primary Type"]].dropna()
    temp_sample = temp.sample(n=min(GEO_SAMPLE, len(temp)), random_state=42).reset_index(drop=True)
    X = temp_sample[["Hour", "Day_Num", "Month", "Crime_Severity_Score"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    with mlflow.start_run(run_name="KMeans_Temporal"):
        k = 4
        km_t = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_t = km_t.fit_predict(X_scaled)
        sil_t = silhouette_score(X_scaled, labels_t, sample_size=5000, random_state=42)
        db_t  = davies_bouldin_score(X_scaled, labels_t)

        mlflow.log_params({"algorithm": "KMeans", "n_clusters": k, "type": "temporal"})
        mlflow.log_metrics({"silhouette_score": round(sil_t,4), "davies_bouldin": round(db_t,4)})
        mlflow.sklearn.log_model(km_t, "kmeans_temporal_model")

        temp_sample["Temporal_Cluster"] = labels_t
        print(f"[train] Temporal K-Means → sil={sil_t:.4f}, DB={db_t:.4f}")

        temporal_metrics = {"KMeans_Temporal": {"silhouette": round(sil_t,4), "davies_bouldin": round(db_t,4), "n_clusters": k}}
        save_json(temporal_metrics, "temporal_clustering_metrics.json")
        temp_sample.to_csv(os.path.join(CLEAN_DIR, "temporal_sample.csv"), index=False)


# ── PCA ───────────────────────────────────────────────────────────────────────

def run_pca(df):
    print("\n[train] ── PCA ──────────────────────────────────────────────")
    feature_cols = ["Hour", "Day_Num", "Month", "Is_Weekend", "Is_Night",
                    "Crime_Severity_Score", "Latitude", "Longitude",
                    "District", "Ward", "Beat", "Community Area",
                    "Arrest", "Domestic"]
    ml_df = df[feature_cols].dropna()
    ml_sample = ml_df.sample(n=min(ML_SAMPLE, len(ml_df)), random_state=42)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(ml_sample)

    # Full PCA to find 70%+ components
    pca_full = PCA(random_state=42)
    pca_full.fit(X_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_70 = int(np.argmax(cumvar >= 0.70)) + 1
    n_80 = int(np.argmax(cumvar >= 0.80)) + 1
    print(f"[train] PCA: {n_70} components → 70% var, {n_80} components → 80% var")

    # Save scree data (first 14 components)
    scree_df = pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(len(pca_full.explained_variance_ratio_))],
        "Explained_Variance": pca_full.explained_variance_ratio_.round(4),
        "Cumulative_Variance": cumvar.round(4),
    })
    scree_df.to_csv(os.path.join(CLEAN_DIR, "pca_variance.csv"), index=False)

    # PCA 2D for visualization
    pca_2d = PCA(n_components=2, random_state=42)
    X_pca  = pca_2d.fit_transform(X_scaled)

    # Feature loadings (PC1-PC3)
    pca_3d = PCA(n_components=min(3, len(feature_cols)), random_state=42)
    pca_3d.fit(X_scaled)
    loadings_df = pd.DataFrame(pca_3d.components_.T,
                                index=feature_cols,
                                columns=[f"PC{i+1}" for i in range(pca_3d.n_components_)])
    loadings_df.to_csv(os.path.join(CLEAN_DIR, "pca_loadings.csv"))

    # PCA result sample
    pca_result = ml_sample.copy().reset_index(drop=True)
    pca_result["PC1"] = X_pca[:, 0]
    pca_result["PC2"] = X_pca[:, 1]

    # Merge Primary Type for coloring
    idx = df[feature_cols].dropna().index
    types = df.loc[idx, "Primary Type"].reset_index(drop=True)
    pca_result["Primary_Type"] = types[pca_result.index].values
    pca_result.to_csv(os.path.join(CLEAN_DIR, "pca_result.csv"), index=False)

    var2d = float((pca_2d.explained_variance_ratio_.sum() * 100).round(2))
    print(f"[train] PCA 2D → {var2d}% variance explained")

    with mlflow.start_run(run_name="PCA_DimensionalityReduction"):
        mlflow.log_params({"n_components_2d": 2, "n_components_70pct": n_70,
                            "n_components_80pct": n_80, "n_features": len(feature_cols)})
        mlflow.log_metrics({"variance_explained_2d": round(var2d, 2),
                             "n_components_for_70pct": n_70,
                             "n_components_for_80pct": n_80})
        mlflow.sklearn.log_model(pca_2d, "pca_2d_model")

    dim_metrics = {
        "PCA": {"variance_explained_2pc": round(var2d,2), "n_components_70pct": n_70,
                 "n_components_80pct": n_80, "n_features": len(feature_cols)},
    }
    return dim_metrics, X_scaled, pca_result


# ── t-SNE ─────────────────────────────────────────────────────────────────────

def run_tsne(df, X_pca_input_df):
    print("\n[train] ── t-SNE ────────────────────────────────────────────")
    feature_cols = ["Hour", "Day_Num", "Month", "Is_Weekend",
                    "Crime_Severity_Score", "Latitude", "Longitude",
                    "District", "Ward", "Beat", "Community Area",
                    "Arrest", "Domestic"]
    extra_cols   = ["Primary Type", "Is_Night"]
    all_cols = feature_cols + extra_cols
    ml_df = df[all_cols].dropna()
    tsne_sample_df = ml_df.sample(n=min(ML_SAMPLE, len(ml_df)), random_state=42).reset_index(drop=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(tsne_sample_df[feature_cols].values)

    # Use PCA to pre-reduce to 10 dims first (standard practice for t-SNE speed)
    pca_pre = PCA(n_components=min(10, X_scaled.shape[1]), random_state=42)
    X_pre = pca_pre.fit_transform(X_scaled)

    tsne = TSNE(n_components=2, perplexity=30, max_iter=300, random_state=42)
    X_tsne = tsne.fit_transform(X_pre)

    tsne_result = tsne_sample_df.reset_index(drop=True).copy()
    tsne_result["TSNE1"] = X_tsne[:, 0]
    tsne_result["TSNE2"] = X_tsne[:, 1]
    night_map = {1: "Night (8PM-6AM)", 0: "Day (6AM-8PM)"}
    is_night_series = tsne_result["Is_Night"].astype(int)
    tsne_result["Time_Period"] = is_night_series.map(night_map)
    tsne_result.to_csv(os.path.join(CLEAN_DIR, "tsne_result.csv"), index=False)
    print(f"[train] t-SNE done. KL-divergence: {tsne.kl_divergence_:.4f}")

    with mlflow.start_run(run_name="tSNE_Visualization"):
        mlflow.log_params({"perplexity": 30, "max_iter": 300, "n_components": 2, "type": "dimensionality_reduction"})
        mlflow.log_metrics({"kl_divergence": round(float(tsne.kl_divergence_), 4)})

    return {"tSNE": {"kl_divergence": round(float(tsne.kl_divergence_), 4)}}


# ── Main ──────────────────────────────────────────────────────────────────────

def main(resume=False):
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    df = load_clean()

    geo_sample_path = os.path.join(CLEAN_DIR, "geo_sample.csv")
    pca_path        = os.path.join(CLEAN_DIR, "pca_result.csv")
    tsne_path       = os.path.join(CLEAN_DIR, "tsne_result.csv")

    if resume and os.path.exists(geo_sample_path):
        print("[train] Resuming — geo clustering already done, loading cached.")
        geo_sample = pd.read_csv(geo_sample_path)
    else:
        geo_sample = geo_clustering(df)

    if resume and os.path.exists(os.path.join(CLEAN_DIR, "temporal_sample.csv")):
        print("[train] Resuming — temporal clustering already done.")
    else:
        temporal_clustering(df)

    if resume and os.path.exists(pca_path):
        print("[train] Resuming — PCA already done.")
        pca_result = pd.read_csv(pca_path)
        dim_metrics = {}
    else:
        dim_metrics, X_scaled, pca_result = run_pca(df)

    if resume and os.path.exists(tsne_path):
        print("[train] Resuming — t-SNE already done.")
    else:
        tsne_metrics = run_tsne(df, pca_result)
        if not (resume and os.path.exists(pca_path)):
            dim_metrics.update(tsne_metrics)
        else:
            dim_metrics = tsne_metrics
        save_json(dim_metrics, "dimensionality_reduction_summary.json")

    # Combined sample for app (join geo labels back)
    geo_small = geo_sample.sample(n=min(ML_SAMPLE, len(geo_sample)), random_state=42).reset_index(drop=True)
    geo_small.to_csv(os.path.join(CLEAN_DIR, "sample_clusters.csv"), index=False)

    # Silhouette comparison CSV
    geo_metrics_j   = json.load(open(os.path.join(CLEAN_DIR, "geo_clustering_metrics.json")))
    temp_metrics_raw = json.load(open(os.path.join(CLEAN_DIR, "temporal_clustering_metrics.json")))
    sil_rows = []
    for name, m in {**geo_metrics_j, **temp_metrics_raw}.items():
        sil_rows.append({"Algorithm": name, "Silhouette": m.get("silhouette",0),
                          "Davies_Bouldin": m.get("davies_bouldin",0),
                          "N_Clusters": m.get("n_clusters","N/A")})
    pd.DataFrame(sil_rows).to_csv(os.path.join(CLEAN_DIR, "silhouette_scores.csv"), index=False)

    print("\n[train] ✅ All done! Data saved to data/cleaned/")
    print(f"[train] MLflow runs saved in {MLFLOW_URI}")


if __name__ == "__main__":
    import sys
    resume = "--resume" in sys.argv
    main(resume=resume)

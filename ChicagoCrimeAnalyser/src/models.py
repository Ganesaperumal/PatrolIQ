import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import joblib

def run_models(input_path: str, output_dir: str):
    print("Loading model-ready data...")
    df = pd.read_csv(input_path)
    
    # Take a sub-sample for speed since DBSCAN and t-SNE are O(N^2) / O(N log N)
    print("Sub-sampling to 10,000 records for efficient training...")
    sample_df = df.sample(n=10000, random_state=42).copy()
    
    geo_features = ['Latitude_Scaled', 'Longitude_Scaled']
    temporal_features = ['Hour_Sin', 'Hour_Cos', 'Month_Sin', 'Month_Cos']
    
    X_geo = sample_df[geo_features]
    X_temp = sample_df[temporal_features]
    
    # All features for PCA
    X_all = sample_df.select_dtypes(include=[np.number]).dropna(axis=1)

    os.makedirs(output_dir, exist_ok=True)
    
    # ----------------------------------------------------
    # Experiment 1: Geographic Clustering
    # ----------------------------------------------------
    mlflow.set_experiment("PatrolIQ_Geographic_Clustering")
    
    # 1. K-Means
    with mlflow.start_run(run_name="KMeans_Geo"):
        print("Training Geographic K-Means...")
        kmeans = KMeans(n_clusters=8, random_state=42, n_init="auto")
        sample_df['Geo_Cluster_KMeans'] = kmeans.fit_predict(X_geo)
        
        score = silhouette_score(X_geo, sample_df['Geo_Cluster_KMeans'])
        mlflow.log_param("n_clusters", 8)
        mlflow.log_metric("silhouette_score", score)
        mlflow.sklearn.log_model(kmeans, "kmeans_geo_model")
        print(f"K-Means Geo Silhouette Score: {score:.3f}")

    # 2. DBSCAN
    with mlflow.start_run(run_name="DBSCAN_Geo"):
        print("Training Geographic DBSCAN...")
        dbscan = DBSCAN(eps=0.1, min_samples=20)
        sample_df['Geo_Cluster_DBSCAN'] = dbscan.fit_predict(X_geo)
        
        # Calculate silhouette only on non-noise points if there's more than 1 cluster
        non_noise = sample_df['Geo_Cluster_DBSCAN'] != -1
        if len(set(sample_df.loc[non_noise, 'Geo_Cluster_DBSCAN'])) > 1:
            score = silhouette_score(X_geo[non_noise], sample_df.loc[non_noise, 'Geo_Cluster_DBSCAN'])
        else:
            score = -1
            
        mlflow.log_param("eps", 0.1)
        mlflow.log_param("min_samples", 20)
        mlflow.log_metric("silhouette_score", score)
        print(f"DBSCAN Geo Silhouette Score: {score:.3f}")

    # 3. Hierarchical
    with mlflow.start_run(run_name="Hierarchical_Geo"):
        print("Training Geographic Agglomerative Clustering...")
        agg = AgglomerativeClustering(n_clusters=8)
        sample_df['Geo_Cluster_Agg'] = agg.fit_predict(X_geo)
        
        score = silhouette_score(X_geo, sample_df['Geo_Cluster_Agg'])
        mlflow.log_param("n_clusters", 8)
        mlflow.log_metric("silhouette_score", score)
        print(f"Agglomerative Geo Silhouette Score: {score:.3f}")

    # ----------------------------------------------------
    # Experiment 2: Temporal Clustering
    # ----------------------------------------------------
    mlflow.set_experiment("PatrolIQ_Temporal_Clustering")
    
    with mlflow.start_run(run_name="KMeans_Temporal"):
        print("Training Temporal K-Means...")
        kmeans_temp = KMeans(n_clusters=4, random_state=42, n_init="auto")
        sample_df['Temp_Cluster_KMeans'] = kmeans_temp.fit_predict(X_temp)
        
        score = silhouette_score(X_temp, sample_df['Temp_Cluster_KMeans'])
        mlflow.log_param("n_clusters", 4)
        mlflow.log_metric("silhouette_score", score)
        mlflow.sklearn.log_model(kmeans_temp, "kmeans_temp_model")
        print(f"K-Means Temporal Silhouette Score: {score:.3f}")

    # ----------------------------------------------------
    # Experiment 3: Dimensionality Reduction
    # ----------------------------------------------------
    mlflow.set_experiment("PatrolIQ_Dimensionality_Reduction")
    
    # 1. PCA
    with mlflow.start_run(run_name="PCA_Reduction"):
        print("Training PCA...")
        # Target ~20 numeric features for PCA to prevent sparse errors
        pca_cols = [c for c in X_all.columns if c not in ['ID', 'Case Number', 'Year', 'Geo_Cluster_KMeans', 'Geo_Cluster_DBSCAN', 'Geo_Cluster_Agg', 'Temp_Cluster_KMeans']]
        X_for_pca = sample_df[pca_cols].fillna(0)
        
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(X_for_pca)
        
        explained_var = pca.explained_variance_ratio_.sum()
        
        mlflow.log_param("n_components", 3)
        mlflow.log_metric("explained_variance_ratio", explained_var)
        mlflow.sklearn.log_model(pca, "pca_model")
        print(f"PCA total explained variance: {explained_var:.3f}")
        
        sample_df['PCA_1'] = pca_result[:, 0]
        sample_df['PCA_2'] = pca_result[:, 1]
        sample_df['PCA_3'] = pca_result[:, 2]

    # 2. t-SNE
    with mlflow.start_run(run_name="tSNE_Visualization"):
        print("Training t-SNE...")
        # Apply t-SNE on PCA results for speed
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        tsne_result = tsne.fit_transform(pca_result)
        
        mlflow.log_param("n_components", 2)
        mlflow.log_param("perplexity", 30)
        
        sample_df['tSNE_1'] = tsne_result[:, 0]
        sample_df['tSNE_2'] = tsne_result[:, 1]
        print("tSNE successfully transformed!")

    # Save final clustered output
    final_output = os.path.join(output_dir, 'clustered_crimes_sample.csv')
    sample_df.to_csv(final_output, index=False)
    print(f"Saved clustered results to {final_output}")
    print("Modeling phase complete!")

if __name__ == "__main__":
    input_file = "../data/processed/model_ready_crimes.csv"
    output_dir = "../data/processed"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, input_file)
    output_path = os.path.join(script_dir, output_dir)
    
    run_models(input_path, output_path)

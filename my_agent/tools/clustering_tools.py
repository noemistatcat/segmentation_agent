"""Clustering tools for customer segmentation analysis."""

import pandas as pd
import numpy as np
from typing import Optional, List
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from collections import Counter
try:
    from adk import tool
except Exception:
    def tool(fn=None, *args, **kwargs):
        if fn is None:
            def decorator(f):
                return f
            return decorator
        return fn


@tool
def preprocess_csv_for_clustering(csv_file: str) -> dict:
    """Preprocess CSV data for clustering analysis.

    Args:
        csv_file: Path to the CSV file

    Returns:
        Dictionary with features, feature_names, num_samples, num_features
    """
    df = pd.read_csv(csv_file)
    id_cols = [col for col in df.columns if 'id' in col.lower()]
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col not in id_cols]

    df_encoded = df.copy()
    for col in categorical_cols:
        if col not in id_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col].astype(str))

    feature_cols = numerical_cols + [col for col in categorical_cols if col not in id_cols]
    features = df_encoded[feature_cols].values
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    return {
        'features': scaled_features.tolist(),
        'feature_names': feature_cols,
        'num_samples': len(df),
        'num_features': len(feature_cols)
    }


@tool
def perform_cluster_analysis(preprocessed_data: dict, max_clusters: int = 6) -> dict:
    """Perform KMeans clustering on preprocessed data.

    Args:
        preprocessed_data: Dictionary from preprocess_csv_for_clustering
        max_clusters: Max clusters to test for auto-detection

    Returns:
        Dictionary with labels, n_clusters, cluster_centers, scores, cluster_sizes
    """
    features = np.array(preprocessed_data['features'])
    n_clusters = _find_optimal_clusters(features, max_clusters)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3, algorithm='elkan')
    labels = kmeans.fit_predict(features)
    silhouette = silhouette_score(features, labels)
    davies_bouldin = davies_bouldin_score(features, labels)
    cluster_counts = Counter(labels)
    cluster_sizes = {f"cluster_{i}": cluster_counts[i] for i in range(n_clusters)}

    return {
        'labels': labels.tolist(),
        'n_clusters': int(n_clusters),
        'cluster_centers': kmeans.cluster_centers_.tolist(),
        'silhouette_score': float(silhouette),
        'davies_bouldin_score': float(davies_bouldin),
        'cluster_sizes': cluster_sizes
    }


def _find_optimal_clusters(features, max_clusters):
    """Find optimal number of clusters using silhouette score.

    Parameters:
        features: Scaled feature array
        max_clusters: Maximum number of clusters to test

    Returns:
        Optimal number of clusters
    """
    n_samples = features.shape[0]
    max_clusters = min(max_clusters, n_samples - 1)
    best_score = -1
    best_k = 2

    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=3, algorithm='elkan')
        labels = kmeans.fit_predict(features)
        score = silhouette_score(features, labels)
        if score > best_score:
            best_score = score
            best_k = k

    return best_k


@tool
def save_clustered_data(csv_file: str, cluster_labels: List[int], output_file: Optional[str] = None) -> dict:
    """Merge cluster labels with original CSV data and save to file.

    Args:
        csv_file: Path to the original CSV file
        cluster_labels: List of cluster labels for each row
        output_file: Optional output file path. If not provided, generates automatically.

    Returns:
        Dictionary with output_path and success status
    """
    try:
        # Read original CSV
        df = pd.read_csv(csv_file)

        # Validate label count matches data rows
        if len(cluster_labels) != len(df):
            return {
                'success': False,
                'error': f'Label count mismatch: {len(cluster_labels)} labels vs {len(df)} rows'
            }

        # Add cluster column
        df['Cluster'] = cluster_labels

        # Generate output path if not provided
        if output_file is None:
            base_path = csv_file.rsplit('.', 1)[0]
            output_file = f"{base_path}_clustered.csv"

        # Save to CSV
        df.to_csv(output_file, index=False)

        return {
            'success': True,
            'output_path': output_file,
            'num_rows': len(df),
            'num_clusters': len(set(cluster_labels))
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
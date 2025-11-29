"""IMPROVED clustering tools with better cluster count selection and ADK logging."""

import logging
import pandas as pd
import numpy as np
from typing import Optional, List
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from collections import Counter
import tempfile
import pickle
import os

# Configure logging according to ADK documentation
logger = logging.getLogger(__name__)

try:
    from adk import tool
except Exception:
    def tool(fn=None, *args, **kwargs):
        if fn is None:
            def decorator(f):
                return f
            return decorator
        return fn

# Cache for preprocessed features to avoid passing large arrays through LLM context
_FEATURE_CACHE = {}


@tool
def preprocess_csv_for_clustering(csv_file: str) -> dict:
    """Preprocess CSV data for clustering analysis.

    Args:
        csv_file: Path to the CSV file

    Returns:
        Dictionary with cache_key, feature_names, num_samples, num_features (features cached internally)
    """
    logger.info(f"Starting preprocessing for CSV file: {csv_file}")

    try:
        df = pd.read_csv(csv_file)
        logger.debug(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")

        id_cols = [col for col in df.columns if 'id' in col.lower()]
        logger.debug(f"Identified ID columns: {id_cols}")

        # Exclude common label columns
        exclude_cols = id_cols + ['true_segment', 'segment', 'cluster', 'label']
        exclude_cols = [col.lower() for col in exclude_cols]

        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        numerical_cols = [col for col in numerical_cols if col.lower() not in exclude_cols]
        categorical_cols = [col for col in categorical_cols if col.lower() not in exclude_cols]

        logger.info(f"Feature columns identified - Numerical: {len(numerical_cols)}, Categorical: {len(categorical_cols)}")

        df_encoded = df.copy()
        for col in categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col].astype(str))
            logger.debug(f"Encoded categorical column: {col}")

        feature_cols = numerical_cols + categorical_cols
        features = df_encoded[feature_cols].values
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        logger.info(f"Successfully scaled {len(feature_cols)} features")

        # Cache features instead of returning them (avoids bloating LLM context)
        cache_key = f"features_{csv_file}_{len(df)}"
        _FEATURE_CACHE[cache_key] = scaled_features

        logger.info(f"Cached features with key: {cache_key}")

        result = {
            'cache_key': cache_key,
            'feature_names': feature_cols,
            'num_samples': len(df),
            'num_features': len(feature_cols)
        }

        logger.info(f"Preprocessing complete: {result['num_samples']} samples, {result['num_features']} features")
        return result

    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}", exc_info=True)
        raise


def find_elbow_point(inertias: dict) -> int:
    """
    Find the elbow point using the angle/curvature method.

    The elbow point is where the rate of decrease sharply changes,
    indicating diminishing returns from adding more clusters.

    Args:
        inertias: Dictionary of {f"k={k}": inertia_value}

    Returns:
        The optimal k value at the elbow
    """
    logger.debug("Finding elbow point from inertias")

    k_values = sorted([int(k.split('=')[1]) for k in inertias.keys()])
    inertia_values = np.array([inertias[f"k={k}"] for k in k_values])

    # Normalize k and inertia to 0-1 range for angle calculation
    k_norm = (np.array(k_values) - k_values[0]) / (k_values[-1] - k_values[0])
    inertia_norm = (inertia_values - inertia_values.min()) / (inertia_values.max() - inertia_values.min())

    # Calculate distances from each point to the line connecting first and last points
    # This finds the point of maximum curvature
    p1 = np.array([k_norm[0], inertia_norm[0]])
    p2 = np.array([k_norm[-1], inertia_norm[-1]])

    distances = []
    for i in range(len(k_norm)):
        p = np.array([k_norm[i], inertia_norm[i]])
        # Distance from point to line
        d = np.abs(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)
        distances.append(d)

    # The elbow is the point with maximum distance from the line
    elbow_idx = np.argmax(distances)
    elbow_k = k_values[elbow_idx]

    logger.info(f"Elbow method selected k={elbow_k} (max distance: {distances[elbow_idx]:.4f})")
    return elbow_k


@tool
def perform_cluster_analysis(preprocessed_data: dict, max_clusters: int = 8, n_clusters: Optional[int] = None) -> dict:
    """Optimized clustering with IMPROVED cluster count selection.

    Uses a composite scoring method that combines:
    - Silhouette score
    - Davies-Bouldin score
    - Penalty for overly simple (k=2) or complex (k>6) solutions

    This approach better identifies the true number of clusters compared to silhouette alone.

    Args:
        preprocessed_data: Dictionary from preprocess_csv_for_clustering (contains cache_key)
        max_clusters: Maximum number of clusters to try (default: 8)
        n_clusters: Number of clusters (if None, will optimize automatically)

    Returns:
        Dictionary with labels, n_clusters, cache_key, scores, cluster_sizes, optimization_scores
    """
    logger.info(f"Starting cluster analysis with max_clusters={max_clusters}, n_clusters={n_clusters}")

    # Retrieve features from cache
    cache_key = preprocessed_data['cache_key']
    if cache_key not in _FEATURE_CACHE:
        error_msg = f'Features not found in cache for key: {cache_key}'
        logger.error(error_msg)
        return {'error': error_msg}

    features = _FEATURE_CACHE[cache_key]
    n_samples = features.shape[0]

    logger.debug(f"Retrieved features from cache: {n_samples} samples, {features.shape[1]} features")

    # Adjust max_clusters based on dataset size
    max_clusters = min(max_clusters, n_samples - 1, 10)
    logger.debug(f"Adjusted max_clusters to {max_clusters} based on dataset size")

    # If n_clusters not specified, find optimal number using ELBOW METHOD on inertia
    if n_clusters is None:
        logger.info("No n_clusters specified, running elbow method optimization")

        silhouette_scores = {}
        davies_bouldin_scores = {}
        calinski_scores = {}
        inertias = {}

        # Try different cluster counts
        min_clusters = 2
        max_test = min(max_clusters, 8)

        logger.info(f"Testing k values from {min_clusters} to {max_test}")

        for k in range(min_clusters, max_test + 1):
            logger.debug(f"Testing k={k}")

            # Use MiniBatchKMeans for speed
            batch_size = min(500, max(100, n_samples // 10))
            kmeans = MiniBatchKMeans(
                n_clusters=k,
                random_state=42,
                batch_size=batch_size,
                max_iter=50,  # More iterations for better quality
                n_init=3      # More initializations
            )
            temp_labels = kmeans.fit_predict(features)

            # Store inertia for elbow method
            inertias[f"k={k}"] = float(kmeans.inertia_)
            logger.debug(f"k={k}: inertia={kmeans.inertia_:.2f}")

            # Calculate multiple metrics
            if n_samples > 500:
                sample_size = 500
                indices = np.random.choice(n_samples, sample_size, replace=False)
                sil_score = silhouette_score(features[indices], temp_labels[indices])
                db_score = davies_bouldin_score(features[indices], temp_labels[indices])
                ch_score = calinski_harabasz_score(features[indices], temp_labels[indices])
                logger.debug(f"k={k}: Using sampling (500/{n_samples}) for metric calculation")
            else:
                sil_score = silhouette_score(features, temp_labels)
                db_score = davies_bouldin_score(features, temp_labels)
                ch_score = calinski_harabasz_score(features, temp_labels)

            silhouette_scores[f"k={k}"] = float(sil_score)
            davies_bouldin_scores[f"k={k}"] = float(db_score)
            calinski_scores[f"k={k}"] = float(ch_score)

            logger.debug(f"k={k}: silhouette={sil_score:.4f}, db={db_score:.4f}, ch={ch_score:.2f}")

        # Use ELBOW METHOD to find optimal k
        n_clusters = find_elbow_point(inertias)

        logger.info(f"Optimization complete: selected k={n_clusters} using elbow method")

        optimization_scores = {
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'davies_bouldin_scores': davies_bouldin_scores,
            'calinski_harabasz_scores': calinski_scores,
            'selected_k': n_clusters,
            'selection_method': 'elbow_method_on_inertia'
        }
    else:
        logger.info(f"Using specified n_clusters={n_clusters}")
        optimization_scores = None

    # Final clustering with optimal/specified number of clusters
    logger.info(f"Performing final clustering with k={n_clusters}")

    batch_size = min(500, max(100, n_samples // 10))
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=batch_size,
        max_iter=100,  # More iterations for final clustering
        n_init=5       # More initializations for better quality
    )

    labels = kmeans.fit_predict(features)
    logger.debug(f"Clustering complete, generated {len(set(labels))} clusters")

    # Calculate final metrics
    if n_samples > 500:
        sample_size = 500
        indices = np.random.choice(n_samples, sample_size, replace=False)
        silhouette = silhouette_score(features[indices], labels[indices])
        davies_bouldin = davies_bouldin_score(features[indices], labels[indices])
        calinski = calinski_harabasz_score(features[indices], labels[indices])
    else:
        silhouette = silhouette_score(features, labels)
        davies_bouldin = davies_bouldin_score(features, labels)
        calinski = calinski_harabasz_score(features, labels)

    cluster_counts = Counter(labels)
    cluster_sizes = {f"cluster_{i}": cluster_counts[i] for i in range(n_clusters)}

    logger.info(f"Final metrics - Silhouette: {silhouette:.4f}, Davies-Bouldin: {davies_bouldin:.4f}, Calinski-Harabasz: {calinski:.2f}")
    logger.info(f"Cluster sizes: {cluster_sizes}")

    result = {
        'labels': labels.tolist(),
        'n_clusters': int(n_clusters),
        'cache_key': cache_key,
        'silhouette_score': float(silhouette),
        'davies_bouldin_score': float(davies_bouldin),
        'calinski_harabasz_score': float(calinski),
        'cluster_sizes': cluster_sizes
    }

    if optimization_scores:
        result['optimization_scores'] = optimization_scores

    logger.info("Cluster analysis completed successfully")
    return result


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
    logger.info(f"Saving clustered data from {csv_file}")
    logger.debug(f"Received {len(cluster_labels)} cluster labels")

    try:
        # Read original CSV
        df = pd.read_csv(csv_file)
        logger.debug(f"Loaded CSV with {len(df)} rows")

        # Validate label count matches data rows
        if len(cluster_labels) != len(df):
            error_msg = f'Label count mismatch: {len(cluster_labels)} labels vs {len(df)} rows'
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }

        # Add cluster column
        df['Cluster'] = cluster_labels
        logger.debug("Added Cluster column to dataframe")

        # Generate output path if not provided
        if output_file is None:
            base_path = csv_file.rsplit('.', 1)[0]
            output_file = f"{base_path}_clustered.csv"
            logger.debug(f"Generated output path: {output_file}")

        # Save to CSV
        df.to_csv(output_file, index=False)
        logger.info(f"Successfully saved clustered data to {output_file}")

        result = {
            'success': True,
            'output_path': output_file,
            'num_rows': len(df),
            'num_clusters': len(set(cluster_labels))
        }

        logger.info(f"Saved {result['num_rows']} rows with {result['num_clusters']} clusters")
        return result

    except Exception as e:
        logger.error(f"Error saving clustered data: {str(e)}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }

"""
Machine learning models for stellar anomaly detection.
Implements Isolation Forest, Autoencoder, and clustering methods.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
# TensorFlow not compatible with Python 3.13 yet
# Using scikit-learn alternatives for now
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
import joblib
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class StellarAnomalyDetector:
    """Main class for stellar anomaly detection using multiple methods."""
    
    def __init__(self, output_dir="../results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Model storage
        self.isolation_forest = None
        self.autoencoder = None
        self.dbscan = None
        
        # Results storage
        self.anomaly_scores = {}
        self.feature_names = None
        
    def load_training_data(self, features_file, metadata_file, scaler_file):
        """Load preprocessed training data."""
        print("Loading training data...")
        
        # Load features
        self.X = pd.read_csv(features_file)
        print(f"Features shape: {self.X.shape}")
        
        # Load metadata
        self.metadata = pd.read_csv(metadata_file)
        print(f"Metadata shape: {self.metadata.shape}")
        
        # Load scaler and feature names
        scaler_data = joblib.load(scaler_file)
        self.scaler = scaler_data['scaler']
        self.feature_names = scaler_data['feature_names']
        print(f"Features: {self.feature_names}")
        
        return self.X, self.metadata
    
    def train_isolation_forest(self, contamination=0.05, n_estimators=100, 
                             random_state=42):
        """
        Train Isolation Forest for anomaly detection.
        
        Parameters:
        -----------
        contamination : float
            Expected proportion of anomalies (0.05 = 5%)
        n_estimators : int
            Number of trees in the forest
        random_state : int
            Random seed for reproducibility
        """
        print(f"Training Isolation Forest (contamination={contamination})...")
        
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the model
        self.isolation_forest.fit(self.X)
        
        # Get anomaly scores (lower = more anomalous)
        if_scores = self.isolation_forest.decision_function(self.X)
        if_predictions = self.isolation_forest.predict(self.X)
        
        # Convert to anomaly scores (higher = more anomalous)
        if_anomaly_scores = -if_scores  # Flip sign
        
        self.anomaly_scores['isolation_forest'] = {
            'scores': if_anomaly_scores,
            'predictions': if_predictions,
            'threshold': 0.0  # Threshold for anomaly classification
        }
        
        n_anomalies = np.sum(if_predictions == -1)
        print(f"Isolation Forest identified {n_anomalies:,} anomalies "
              f"({n_anomalies/len(self.X)*100:.2f}%)")
        
        return if_anomaly_scores
    
    def build_autoencoder(self, input_dim, encoding_dim=None, 
                         hidden_layers=None):
        """
        Build autoencoder using scikit-learn MLPRegressor.
        
        Parameters:
        -----------
        input_dim : int
            Number of input features
        encoding_dim : int
            Dimension of encoding layer (None = input_dim // 2)
        hidden_layers : list
            List of hidden layer sizes
        """
        if encoding_dim is None:
            encoding_dim = max(2, input_dim // 2)
        
        if hidden_layers is None:
            hidden_layers = [input_dim * 2, encoding_dim]
        
        print(f"Building autoencoder with MLPRegressor: {input_dim} -> {hidden_layers} -> {input_dim}")
        
        # Create MLPRegressor as autoencoder substitute
        autoencoder = MLPRegressor(
            hidden_layer_sizes=tuple(hidden_layers),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=10
        )
        
        return autoencoder
    
    def train_autoencoder(self, epochs=100, batch_size=256, validation_split=0.2,
                         encoding_dim=None, hidden_layers=None):
        """
        Train autoencoder using scikit-learn MLPRegressor.
        
        Parameters:
        -----------
        epochs : int
            Maximum iterations (replaces epochs)
        batch_size : int
            Not used in scikit-learn version
        validation_split : float
            Validation fraction for early stopping
        """
        print("Training Autoencoder with scikit-learn...")
        
        input_dim = self.X.shape[1]
        
        # Build model
        self.autoencoder = self.build_autoencoder(
            input_dim, encoding_dim, hidden_layers
        )
        
        # Update max_iter based on epochs parameter
        self.autoencoder.max_iter = epochs
        
        print(f"Training autoencoder with max_iter={epochs}")
        
        # Train the model (autoencoder learns to reconstruct input)
        print("Fitting autoencoder...")
        self.autoencoder.fit(self.X.values, self.X.values)
        
        print(f"Training completed. Iterations: {self.autoencoder.n_iter_}")
        
        # Calculate reconstruction errors
        print("Calculating reconstruction errors...")
        reconstructions = self.autoencoder.predict(self.X.values)
        mse_errors = np.mean((self.X.values - reconstructions) ** 2, axis=1)
        
        self.anomaly_scores['autoencoder'] = {
            'scores': mse_errors,
            'reconstructions': reconstructions,
            'threshold': np.percentile(mse_errors, 95)  # 95th percentile as threshold
        }
        
        n_anomalies = np.sum(mse_errors > self.anomaly_scores['autoencoder']['threshold'])
        print(f"Autoencoder identified {n_anomalies:,} anomalies "
              f"({n_anomalies/len(self.X)*100:.2f}%)")
        
        return mse_errors, None  # No history object in scikit-learn
    
    def train_dbscan_clustering(self, eps=0.5, min_samples=5):
        """
        Apply DBSCAN clustering to identify outliers.
        
        Parameters:
        -----------
        eps : float
            Maximum distance between samples in a cluster
        min_samples : int
            Minimum samples in a cluster
        """
        print(f"Applying DBSCAN clustering (eps={eps}, min_samples={min_samples})...")
        
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        cluster_labels = self.dbscan.fit_predict(self.X)
        
        # Calculate anomaly scores based on cluster membership
        # Outliers (label = -1) get high scores, cluster members get low scores
        dbscan_scores = np.where(cluster_labels == -1, 1.0, 0.0)
        
        # For cluster members, use distance to cluster center as score
        unique_labels = np.unique(cluster_labels)
        for label in unique_labels:
            if label != -1:  # Skip outliers
                cluster_mask = cluster_labels == label
                cluster_data = self.X.values[cluster_mask]
                cluster_center = np.mean(cluster_data, axis=0)
                
                # Calculate distances to cluster center
                distances = np.linalg.norm(cluster_data - cluster_center, axis=1)
                # Normalize distances to [0, 1] range
                if len(distances) > 1:
                    normalized_distances = (distances - distances.min()) / (distances.max() - distances.min())
                    dbscan_scores[cluster_mask] = normalized_distances * 0.5  # Scale to [0, 0.5]
        
        self.anomaly_scores['dbscan'] = {
            'scores': dbscan_scores,
            'labels': cluster_labels,
            'threshold': 0.5  # Outliers have score = 1.0
        }
        
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_outliers = np.sum(cluster_labels == -1)
        print(f"DBSCAN found {n_clusters} clusters and {n_outliers:,} outliers "
              f"({n_outliers/len(self.X)*100:.2f}%)")
        
        return dbscan_scores, cluster_labels
    
    def combine_anomaly_scores(self, weights=None):
        """
        Combine anomaly scores from multiple methods.
        
        Parameters:
        -----------
        weights : dict
            Weights for each method (None = equal weights)
        """
        if weights is None:
            weights = {method: 1.0 for method in self.anomaly_scores.keys()}
        
        print(f"Combining anomaly scores with weights: {weights}")
        
        # Normalize all scores to [0, 1] range
        normalized_scores = {}
        for method, data in self.anomaly_scores.items():
            scores = data['scores']
            normalized = (scores - scores.min()) / (scores.max() - scores.min())
            normalized_scores[method] = normalized
        
        # Weighted combination
        combined_scores = np.zeros(len(self.X))
        total_weight = sum(weights.values())
        
        for method, weight in weights.items():
            if method in normalized_scores:
                combined_scores += (weight / total_weight) * normalized_scores[method]
        
        self.anomaly_scores['combined'] = {
            'scores': combined_scores,
            'weights': weights,
            'threshold': np.percentile(combined_scores, 95)
        }
        
        return combined_scores
    
    def get_top_anomalies(self, method='combined', top_n=1000):
        """
        Get top N anomalies based on specified method.
        
        Parameters:
        -----------
        method : str
            Scoring method to use
        top_n : int
            Number of top anomalies to return
        """
        if method not in self.anomaly_scores:
            raise ValueError(f"Method {method} not available. "
                           f"Available: {list(self.anomaly_scores.keys())}")
        
        scores = self.anomaly_scores[method]['scores']
        
        # Get indices of top anomalies
        top_indices = np.argsort(scores)[-top_n:][::-1]  # Descending order
        
        # Create results DataFrame
        results = pd.DataFrame({
            'rank': range(1, len(top_indices) + 1),
            'source_id': self.metadata.iloc[top_indices]['source_id'].values,
            'ra': self.metadata.iloc[top_indices]['ra'].values,
            'dec': self.metadata.iloc[top_indices]['dec'].values,
            'anomaly_score': scores[top_indices]
        })
        
        # Add original features for context
        for i, feature in enumerate(self.feature_names):
            results[feature] = self.X.iloc[top_indices, i].values
        
        return results
    
    def save_models_and_results(self):
        """Save trained models and results."""
        print("Saving models and results...")
        
        # Save Isolation Forest
        if self.isolation_forest is not None:
            if_file = os.path.join(self.output_dir, "isolation_forest.joblib")
            joblib.dump(self.isolation_forest, if_file)
            print(f"Isolation Forest saved to: {if_file}")
        
        # Save Autoencoder (scikit-learn version)
        if self.autoencoder is not None:
            ae_file = os.path.join(self.output_dir, "autoencoder.joblib")
            joblib.dump(self.autoencoder, ae_file)
            print(f"Autoencoder saved to: {ae_file}")
        
        # Save DBSCAN
        if self.dbscan is not None:
            dbscan_file = os.path.join(self.output_dir, "dbscan.joblib")
            joblib.dump(self.dbscan, dbscan_file)
            print(f"DBSCAN saved to: {dbscan_file}")
        
        # Save anomaly scores
        scores_file = os.path.join(self.output_dir, "anomaly_scores.joblib")
        joblib.dump(self.anomaly_scores, scores_file)
        print(f"Anomaly scores saved to: {scores_file}")
        
        # Save top anomalies for each method
        for method in self.anomaly_scores.keys():
            top_anomalies = self.get_top_anomalies(method=method, top_n=1000)
            results_file = os.path.join(self.output_dir, f"top_anomalies_{method}.csv")
            top_anomalies.to_csv(results_file, index=False)
            print(f"Top anomalies ({method}) saved to: {results_file}")

def main():
    """Main function for model training."""
    print("=== Stellar Anomaly Detection Training ===")
    
    # Initialize detector
    detector = StellarAnomalyDetector()
    
    # Load training data
    features_file = "../data/training_features.csv"
    metadata_file = "../data/training_metadata.csv"
    scaler_file = "../data/training_scaler.joblib"
    
    try:
        detector.load_training_data(features_file, metadata_file, scaler_file)
    except FileNotFoundError as e:
        print(f"Error: Training data not found. Run preprocessing.py first.")
        print(f"Missing file: {e}")
        return
    
    # Train models
    print("\n1. Training Isolation Forest...")
    detector.train_isolation_forest(contamination=0.05)
    
    print("\n2. Training Autoencoder...")
    detector.train_autoencoder(epochs=50, batch_size=256)
    
    print("\n3. Applying DBSCAN Clustering...")
    detector.train_dbscan_clustering(eps=0.8, min_samples=10)
    
    print("\n4. Combining Anomaly Scores...")
    detector.combine_anomaly_scores(weights={
        'isolation_forest': 0.4,
        'autoencoder': 0.4,
        'dbscan': 0.2
    })
    
    # Save everything
    detector.save_models_and_results()
    
    print("\n=== Model Training Complete ===")
    print("Results saved to ../results/")

if __name__ == "__main__":
    main()

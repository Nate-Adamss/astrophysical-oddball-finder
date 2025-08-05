"""
Data preprocessing module for stellar anomaly detection.
Handles cleaning, normalization, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

class StellarDataPreprocessor:
    """Class for preprocessing stellar data for anomaly detection."""
    
    def __init__(self, output_dir="../data"):
        self.output_dir = output_dir
        self.scaler = None
        self.feature_names = None
        
    def load_data(self, filepath):
        """Load processed Gaia data."""
        print(f"Loading data from: {filepath}")
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df):,} sources with {df.shape[1]} features")
        return df
    
    def remove_outliers(self, df, columns=None, method='iqr', factor=3.0):
        """
        Remove statistical outliers from the dataset.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input data
        columns : list
            Columns to check for outliers (None = all numeric columns)
        method : str
            'iqr' for interquartile range, 'zscore' for z-score
        factor : float
            Multiplier for outlier detection threshold
            
        Returns:
        --------
        pandas.DataFrame : Data with outliers removed
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        print(f"Removing outliers using {method} method (factor={factor})...")
        initial_count = len(df)
        
        if method == 'iqr':
            for col in columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        elif method == 'zscore':
            from scipy import stats
            for col in columns:
                z_scores = np.abs(stats.zscore(df[col]))
                df = df[z_scores < factor]
        
        final_count = len(df)
        removed = initial_count - final_count
        print(f"Removed {removed:,} outliers ({removed/initial_count*100:.1f}%)")
        print(f"Remaining: {final_count:,} sources")
        
        return df
    
    def select_features_for_ml(self, df):
        """
        Select and prepare features for machine learning.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Processed Gaia data
            
        Returns:
        --------
        pandas.DataFrame : Features ready for ML
        """
        print("Selecting features for anomaly detection...")
        
        # Core features for anomaly detection
        feature_columns = [
            # Astrometric features
            'parallax',
            'pmra', 'pmdec', 'pm_total',
            
            # Photometric features  
            'phot_g_mean_mag', 'abs_g_mag',
            'bp_rp',
            
            # Derived kinematic features
            'distance_pc', 'v_tan_km_s',
            'reduced_pm',
            
            # Quality indicators
            'ruwe'
        ]
        
        # Check which features are available
        available_features = [col for col in feature_columns if col in df.columns]
        missing_features = [col for col in feature_columns if col not in df.columns]
        
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
        
        print(f"Selected {len(available_features)} features: {available_features}")
        
        # Create feature matrix
        X = df[available_features].copy()
        
        # Store metadata (source_id, coordinates) separately
        metadata = df[['source_id', 'ra', 'dec']].copy()
        
        return X, metadata, available_features
    
    def handle_missing_values(self, X, strategy='median'):
        """
        Handle missing values in the feature matrix.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Feature matrix
        strategy : str
            Imputation strategy ('mean', 'median', 'most_frequent')
            
        Returns:
        --------
        pandas.DataFrame : Data with missing values handled
        """
        print(f"Handling missing values with strategy: {strategy}")
        
        # Check for missing values
        missing_counts = X.isnull().sum()
        if missing_counts.sum() > 0:
            print("Missing values per column:")
            for col, count in missing_counts.items():
                if count > 0:
                    print(f"  {col}: {count} ({count/len(X)*100:.1f}%)")
            
            # Impute missing values
            imputer = SimpleImputer(strategy=strategy)
            X_imputed = pd.DataFrame(
                imputer.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            
            print(f"Imputed {missing_counts.sum()} missing values")
            return X_imputed
        else:
            print("No missing values found")
            return X
    
    def normalize_features(self, X, method='standard'):
        """
        Normalize features for machine learning.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Feature matrix
        method : str
            'standard' for StandardScaler, 'robust' for RobustScaler
            
        Returns:
        --------
        pandas.DataFrame : Normalized features
        """
        print(f"Normalizing features using {method} scaling...")
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError("Method must be 'standard' or 'robust'")
        
        # Fit and transform
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert back to DataFrame
        X_scaled_df = pd.DataFrame(
            X_scaled,
            columns=X.columns,
            index=X.index
        )
        
        self.feature_names = list(X.columns)
        print(f"Normalized {len(self.feature_names)} features")
        
        return X_scaled_df
    
    def create_training_dataset(self, df, remove_outliers_flag=True, 
                              outlier_method='iqr', scaling_method='robust'):
        """
        Create a complete training dataset for anomaly detection.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Raw processed Gaia data
        remove_outliers_flag : bool
            Whether to remove statistical outliers
        outlier_method : str
            Method for outlier removal
        scaling_method : str
            Method for feature scaling
            
        Returns:
        --------
        tuple : (X_scaled, metadata, feature_names)
        """
        print("=== Creating Training Dataset ===")
        
        # Remove outliers if requested
        if remove_outliers_flag:
            df_clean = self.remove_outliers(df, method=outlier_method, factor=2.5)
        else:
            df_clean = df.copy()
        
        # Select features
        X, metadata, feature_names = self.select_features_for_ml(df_clean)
        
        # Handle missing values
        X_clean = self.handle_missing_values(X)
        
        # Normalize features
        X_scaled = self.normalize_features(X_clean, method=scaling_method)
        
        print(f"\nFinal training dataset:")
        print(f"  Shape: {X_scaled.shape}")
        print(f"  Features: {feature_names}")
        
        return X_scaled, metadata, feature_names
    
    def save_training_data(self, X_scaled, metadata, feature_names, 
                          filename_prefix="training"):
        """Save training data and metadata."""
        
        # Save scaled features
        features_file = os.path.join(self.output_dir, f"{filename_prefix}_features.csv")
        X_scaled.to_csv(features_file, index=False)
        
        # Save metadata
        metadata_file = os.path.join(self.output_dir, f"{filename_prefix}_metadata.csv")
        metadata.to_csv(metadata_file, index=False)
        
        # Save feature names and scaler
        import joblib
        scaler_file = os.path.join(self.output_dir, f"{filename_prefix}_scaler.joblib")
        joblib.dump({
            'scaler': self.scaler,
            'feature_names': feature_names
        }, scaler_file)
        
        print(f"Training data saved:")
        print(f"  Features: {features_file}")
        print(f"  Metadata: {metadata_file}")
        print(f"  Scaler: {scaler_file}")
        
        return features_file, metadata_file, scaler_file

def main():
    """Main function for data preprocessing."""
    print("=== Stellar Data Preprocessing ===")
    
    # Initialize preprocessor
    preprocessor = StellarDataPreprocessor()
    
    # Load processed data
    data_file = "../data/gaia_dr3_processed.csv"
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found. Run data_acquisition.py first.")
        return
    
    df = preprocessor.load_data(data_file)
    
    # Create training dataset
    X_scaled, metadata, feature_names = preprocessor.create_training_dataset(
        df, 
        remove_outliers_flag=True,
        outlier_method='iqr',
        scaling_method='robust'
    )
    
    # Save training data
    preprocessor.save_training_data(X_scaled, metadata, feature_names)
    
    print("\n=== Preprocessing Complete ===")

if __name__ == "__main__":
    main()

"""
Data Loading and Preprocessing Utilities
This file handles loading datasets and preprocessing data
"""

import pandas as pd
import numpy as np
import os


class DataLoader:
    """
    Load and preprocess exoplanet datasets
    """
    
    def __init__(self, data_dir='data'):
        """
        Initialize the data loader
        
        Args:
            data_dir: Directory containing dataset files
        """
        self.data_dir = data_dir
        self.datasets = {
            'kepler': 'kepler_dataset.csv',
            'k2': 'k2_dataset.csv',
            'tess': 'tess_dataset.csv',
            'custom': 'custom_dataset.csv'
        }
        
    def load_dataset(self, dataset_name='kepler'):
        """
        Load a dataset by name
        
        Args:
            dataset_name: Name of dataset ('kepler', 'k2', 'tess', or 'custom')
            
        Returns:
            pandas.DataFrame: Loaded dataset
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(self.datasets.keys())}")
        
        filepath = os.path.join(self.data_dir, self.datasets[dataset_name])
        
        if not os.path.exists(filepath):
            print(f"⚠️  Dataset file not found: {filepath}")
            print(f"   Using sample data instead...")
            return self._generate_sample_data()
        
        print(f"📂 Loading dataset: {filepath}")
        df = pd.read_csv(filepath)
        print(f"   Loaded {len(df)} samples with {len(df.columns)} columns")
        
        return df
    
    def load_file(self, filepath):
        """
        Load a dataset from a file path (for user uploads)
        
        Args:
            filepath: Path to CSV or JSON file
            
        Returns:
            pandas.DataFrame: Loaded dataset
        """
        print(f"📂 Loading file: {filepath}")
        
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.json'):
            df = pd.read_json(filepath)
        else:
            raise ValueError(f"Unsupported file format. Use .csv or .json")
        
        print(f"   Loaded {len(df)} samples with {len(df.columns)} columns")
        return df
    
    def preprocess_data(self, df, remove_missing=True, remove_outliers=True, 
                       remove_false_positives=True):
        """
        Preprocess the dataset
        
        Args:
            df: Input DataFrame
            remove_missing: Remove rows with missing values
            remove_outliers: Remove statistical outliers
            remove_false_positives: Remove known false positives (if column exists)
            
        Returns:
            pandas.DataFrame: Preprocessed dataset
        """
        print(f"🔧 Preprocessing dataset...")
        print(f"   Initial samples: {len(df)}")
        
        df_clean = df.copy()
        
        # Handle missing values
        if remove_missing:
            initial_count = len(df_clean)
            df_clean = df_clean.dropna()
            removed = initial_count - len(df_clean)
            if removed > 0:
                print(f"   Removed {removed} rows with missing values")
        
        # Remove outliers using IQR method
        if remove_outliers:
            df_clean = self._remove_outliers(df_clean)
        
        # Remove false positives if column exists
        if remove_false_positives and 'is_false_positive' in df_clean.columns:
            initial_count = len(df_clean)
            df_clean = df_clean[df_clean['is_false_positive'] != 1]
            removed = initial_count - len(df_clean)
            if removed > 0:
                print(f"   Removed {removed} false positive samples")
        
        print(f"   Final samples: {len(df_clean)}")
        
        return df_clean
    
    def _remove_outliers(self, df, columns=None):
        """
        Remove outliers using IQR method
        
        Args:
            df: Input DataFrame
            columns: List of columns to check (None = all numeric columns)
            
        Returns:
            pandas.DataFrame: Dataset with outliers removed
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        df_clean = df.copy()
        initial_count = len(df_clean)
        
        for col in columns:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df_clean = df_clean[
                    (df_clean[col] >= lower_bound) & 
                    (df_clean[col] <= upper_bound)
                ]
        
        removed = initial_count - len(df_clean)
        if removed > 0:
            print(f"   Removed {removed} outlier samples")
        
        return df_clean
    
    def extract_features(self, df):
        """
        Extract feature matrix (X) and target (y) from DataFrame
        
        Args:
            df: Input DataFrame
            
        Returns:
            tuple: (X, y) where X is feature matrix and y is target
        """
        # Define expected feature columns (10 features)
        feature_columns = [
            'orbital_period',
            'transit_duration',
            'transit_depth',
            'planetary_radius',
            'planet_equilibrium_temp',
            'stellar_effective_temp',
            'stellar_log_g',
            'stellar_radius',
            'ra',
            'dec'
        ]
        
        # Try common target column names
        target_columns = ['label', 'is_exoplanet', 'exoplanet', 'target', 'class']
        
        # Find which feature columns exist
        available_features = [col for col in feature_columns if col in df.columns]
        
        if not available_features:
            raise ValueError(f"No feature columns found. Expected: {feature_columns}")
        
        # Extract features
        X = df[available_features].values
        
        # Find target column
        y = None
        for target_col in target_columns:
            if target_col in df.columns:
                y = df[target_col].values
                print(f"   Using target column: {target_col}")
                break
        
        if y is None:
            raise ValueError(f"No target column found. Expected one of: {target_columns}")
        
        print(f"   Extracted {X.shape[1]} features and {len(y)} samples")
        
        return X, y
    
    def _generate_sample_data(self, n_samples=1000):
        """
        Generate sample exoplanet data for testing
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            pandas.DataFrame: Sample dataset
        """
        print(f"🔬 Generating {n_samples} sample exoplanet records...")
        
        np.random.seed(42)
        
        # Generate features with realistic distributions (10 features)
        data = {
            'orbital_period': np.random.exponential(10, n_samples),  # days
            'transit_duration': np.random.exponential(3, n_samples),  # hours
            'transit_depth': np.random.exponential(0.01, n_samples),  # percentage
            'planetary_radius': np.random.lognormal(0, 0.5, n_samples),  # Earth radii
            'planet_equilibrium_temp': np.random.normal(1200, 400, n_samples),  # Kelvin
            'stellar_effective_temp': np.random.normal(5778, 800, n_samples),  # Kelvin (like Sun)
            'stellar_log_g': np.random.normal(4.44, 0.2, n_samples),  # cgs (like Sun)
            'stellar_radius': np.random.normal(1.0, 0.2, n_samples),  # Solar radii
            'ra': np.random.uniform(0, 360, n_samples),  # degrees (0-360)
            'dec': np.random.uniform(-90, 90, n_samples),  # degrees (-90 to 90)
        }
        
        # Generate labels (roughly 25% exoplanets)
        data['is_exoplanet'] = np.random.choice([0, 1], n_samples, p=[0.75, 0.25])
        
        df = pd.DataFrame(data)
        
        # Ensure positive values (except dec which can be negative)
        for col in df.columns:
            if col not in ['is_exoplanet', 'dec']:
                df[col] = df[col].abs()
        
        return df
    
    def save_dataset(self, df, filename, output_dir='data'):
        """
        Save dataset to CSV file
        
        Args:
            df: DataFrame to save
            filename: Output filename
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"💾 Dataset saved to: {filepath}")


# Preprocessing utilities
def normalize_features(X):
    """
    Normalize features to 0-1 range
    
    Args:
        X: Feature matrix (numpy array)
        
    Returns:
        numpy.ndarray: Normalized features
    """
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    return scaler.fit_transform(X)


def standardize_features(X):
    """
    Standardize features to mean=0, std=1
    
    Args:
        X: Feature matrix (numpy array)
        
    Returns:
        numpy.ndarray: Standardized features
    """
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    return scaler.fit_transform(X)


# Example usage
if __name__ == '__main__':
    """
    Test the data loader
    """
    print("Testing DataLoader...")
    
    # Initialize loader
    loader = DataLoader()
    
    # Generate sample data
    df = loader._generate_sample_data(100)
    print(f"\nSample data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Preprocess
    df_clean = loader.preprocess_data(df)
    
    # Extract features
    X, y = loader.extract_features(df_clean)
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Exoplanet ratio: {y.sum() / len(y):.2%}")
    
    print("\n✅ DataLoader test complete!")

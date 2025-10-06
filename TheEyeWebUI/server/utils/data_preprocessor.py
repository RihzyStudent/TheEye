"""
Data Preprocessor for Exoplanet Detection
Handles CSV data loading and preprocessing for the ML model
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExoplanetDataPreprocessor:
    """
    Preprocessor for exoplanet detection data
    Handles loading, cleaning, and preparing data for ML models
    """
    
    # Required features in exact order for the model (training data format)
    REQUIRED_FEATURES = [
        'Orbital Period',
        'Transition Duration',
        'Transition Depth',
        'Planet Rad',
        'Planet Eqbm Temp',
        'Stellar Effective Temp',
        'Stellar log g',
        'Stellar Rad',
        'ra',
        'dec'
    ]
    
    # Column name mapping (frontend/API names -> training data names)
    COLUMN_MAPPING = {
        'orbital_period': 'Orbital Period',
        'transit_duration': 'Transition Duration',
        'transit_depth': 'Transition Depth',
        'planetary_radius': 'Planet Rad',
        'planetaryRadius': 'Planet Rad',
        'planet_equilibrium_temp': 'Planet Eqbm Temp',
        'planetEquilibriumTemp': 'Planet Eqbm Temp',
        'stellar_effective_temp': 'Stellar Effective Temp',
        'stellarEffectiveTemp': 'Stellar Effective Temp',
        'stellar_log_g': 'Stellar log g',
        'stellarLogG': 'Stellar log g',
        'stellar_radius': 'Stellar Rad',
        'stellarRadius': 'Stellar Rad',
        'transitDuration': 'Transition Duration',
        'transitDepth': 'Transition Depth',
        'orbitalPeriod': 'Orbital Period',
    }
    
    # Optional label column for training
    LABEL_COLUMN = 'Output'  # Changed to match training data
    
    def __init__(self, feature_columns: List[str] = None):
        """
        Initialize preprocessor
        
        Args:
            feature_columns: List of feature column names (default: REQUIRED_FEATURES)
        """
        self.feature_columns = feature_columns or self.REQUIRED_FEATURES
        self.feature_stats = None  # For normalization if needed
    
    def normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names using the mapping
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with normalized column names
        """
        df_normalized = df.copy()
        
        # Create a reverse mapping for columns that need renaming
        rename_map = {}
        for old_name, new_name in self.COLUMN_MAPPING.items():
            if old_name in df_normalized.columns and old_name != new_name:
                rename_map[old_name] = new_name
        
        if rename_map:
            df_normalized = df_normalized.rename(columns=rename_map)
            logger.info(f"✅ Normalized column names: {list(rename_map.keys())} -> {list(rename_map.values())}")
        
        return df_normalized
    
    def load_csv(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load CSV file
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            pandas DataFrame
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"✅ Loaded {len(df)} records from {file_path}")
            # Normalize column names
            df = self.normalize_column_names(df)
            return df
        except FileNotFoundError:
            logger.error(f"❌ File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"❌ Error loading CSV: {e}")
            raise
    
    def validate_data(self, df: pd.DataFrame, require_labels: bool = False) -> bool:
        """
        Validate that DataFrame has required columns
        
        Args:
            df: Input DataFrame
            require_labels: Whether to require label column (for training)
            
        Returns:
            True if valid, raises ValueError if not
        """
        # Check for required features
        missing_features = [col for col in self.REQUIRED_FEATURES if col not in df.columns]
        
        if missing_features:
            logger.error(f"❌ Missing required features: {missing_features}")
            logger.info(f"Available columns: {list(df.columns)}")
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Check for labels if required
        if require_labels and self.LABEL_COLUMN not in df.columns:
            logger.error(f"❌ Missing label column: {self.LABEL_COLUMN}")
            raise ValueError(f"Missing label column for training: {self.LABEL_COLUMN}")
        
        logger.info("✅ Data validation passed")
        return True
    
    def clean_data(self, df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
        """
        Clean data by handling missing values and outliers
        
        Args:
            df: Input DataFrame
            strategy: 'drop' to drop rows with NaN, 'fill' to fill with median
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Get feature columns that exist
        feature_cols = [col for col in self.REQUIRED_FEATURES if col in df_clean.columns]
        
        # Count missing values
        missing_counts = df_clean[feature_cols].isnull().sum()
        if missing_counts.any():
            logger.warning(f"⚠️  Missing values found:\n{missing_counts[missing_counts > 0]}")
        
        if strategy == 'drop':
            # Drop rows with any missing values in feature columns
            df_clean = df_clean.dropna(subset=feature_cols)
            logger.info(f"✅ Dropped rows with missing values. Remaining: {len(df_clean)}")
        
        elif strategy == 'fill':
            # Fill missing values with median
            for col in feature_cols:
                if df_clean[col].isnull().any():
                    median_val = df_clean[col].median()
                    df_clean[col].fillna(median_val, inplace=True)
                    logger.info(f"✅ Filled {col} missing values with median: {median_val:.2f}")
        
        # Remove duplicate rows
        duplicates = df_clean.duplicated().sum()
        if duplicates > 0:
            df_clean = df_clean.drop_duplicates()
            logger.info(f"✅ Removed {duplicates} duplicate rows")
        
        return df_clean
    
    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract feature array in correct order
        
        Args:
            df: Input DataFrame with feature columns
            
        Returns:
            numpy array of shape (n_samples, n_features)
        """
        # Extract features in exact order
        features = df[self.REQUIRED_FEATURES].values
        
        logger.info(f"✅ Extracted features: shape {features.shape}")
        return features
    
    def extract_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract labels for training
        
        Args:
            df: Input DataFrame with label column
            
        Returns:
            numpy array of labels
        """
        if self.LABEL_COLUMN not in df.columns:
            raise ValueError(f"Label column '{self.LABEL_COLUMN}' not found")
        
        labels = df[self.LABEL_COLUMN].values
        logger.info(f"✅ Extracted labels: {len(labels)} samples")
        logger.info(f"   - Class distribution: {np.bincount(labels.astype(int))}")
        
        return labels
    
    def preprocess_for_training(
        self, 
        file_path: Union[str, Path],
        test_size: float = 0.2,
        clean_strategy: str = 'drop'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Full preprocessing pipeline for training
        
        Args:
            file_path: Path to CSV file
            test_size: Fraction of data for testing (0.0 to 1.0)
            clean_strategy: 'drop' or 'fill' for missing values
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        logger.info("📊 Starting preprocessing for training...")
        
        # Load data
        df = self.load_csv(file_path)
        
        # Validate
        self.validate_data(df, require_labels=True)
        
        # Clean
        df = self.clean_data(df, strategy=clean_strategy)
        
        # Extract features and labels
        X = self.extract_features(df)
        y = self.extract_labels(df)
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"✅ Train set: {X_train.shape[0]} samples")
        logger.info(f"✅ Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_for_prediction(
        self, 
        data: Union[str, Path, Dict, pd.DataFrame],
        clean_strategy: str = 'fill'
    ) -> np.ndarray:
        """
        Preprocessing pipeline for prediction
        
        Args:
            data: Can be:
                - Path to CSV file
                - Dictionary with feature values
                - pandas DataFrame
            clean_strategy: 'drop' or 'fill' for missing values
            
        Returns:
            numpy array ready for prediction, shape (1, n_features) for single sample
        """
        logger.info("🔮 Starting preprocessing for prediction...")
        
        # Convert input to DataFrame
        if isinstance(data, (str, Path)):
            df = self.load_csv(data)
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
            # Normalize column names for dictionary input
            df = self.normalize_column_names(df)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
            # Normalize column names
            df = self.normalize_column_names(df)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        # Validate (no labels required for prediction)
        self.validate_data(df, require_labels=False)
        
        # Clean
        df = self.clean_data(df, strategy=clean_strategy)
        
        # Extract features
        X = self.extract_features(df)
        
        logger.info(f"✅ Preprocessed {X.shape[0]} sample(s) for prediction")
        
        return X
    
    def normalize_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Normalize features (optional, if your model needs it)
        
        Args:
            X: Feature array
            fit: If True, fit normalizer on this data; if False, use previously fitted
            
        Returns:
            Normalized feature array
        """
        from sklearn.preprocessing import StandardScaler
        
        if fit:
            self.scaler = StandardScaler()
            X_normalized = self.scaler.fit_transform(X)
            logger.info("✅ Fitted and normalized features")
        else:
            if not hasattr(self, 'scaler'):
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            X_normalized = self.scaler.transform(X)
            logger.info("✅ Normalized features using fitted scaler")
        
        return X_normalized
    
    def get_feature_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Get statistics about the features
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with feature statistics
        """
        stats = {}
        
        for col in self.REQUIRED_FEATURES:
            if col in df.columns:
                stats[col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'median': float(df[col].median()),
                    'missing': int(df[col].isnull().sum())
                }
        
        return stats
    
    def save_preprocessed_data(
        self, 
        X: np.ndarray, 
        y: np.ndarray = None,
        output_path: Union[str, Path] = None
    ):
        """
        Save preprocessed data to CSV
        
        Args:
            X: Feature array
            y: Label array (optional)
            output_path: Path to save CSV
        """
        # Convert to DataFrame
        df = pd.DataFrame(X, columns=self.REQUIRED_FEATURES)
        
        if y is not None:
            df[self.LABEL_COLUMN] = y
        
        # Save
        df.to_csv(output_path, index=False)
        logger.info(f"💾 Saved preprocessed data to {output_path}")


# Convenience functions for quick use

def load_and_preprocess_for_training(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Quick function to load and preprocess data for training
    
    Args:
        csv_path: Path to CSV file with features and labels
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    preprocessor = ExoplanetDataPreprocessor()
    return preprocessor.preprocess_for_training(csv_path)


def load_and_preprocess_for_prediction(data: Union[str, Dict]) -> np.ndarray:
    """
    Quick function to load and preprocess data for prediction
    
    Args:
        data: CSV path or dictionary with features
        
    Returns:
        Feature array ready for prediction
    """
    preprocessor = ExoplanetDataPreprocessor()
    return preprocessor.preprocess_for_prediction(data)


# Example usage
if __name__ == "__main__":
    print("🧪 Testing Exoplanet Data Preprocessor...\n")
    
    # Test with sample data
    sample_data = {
        'orbital_period': 3.52,
        'transit_duration': 2.8,
        'transit_depth': 0.01,
        'planetary_radius': 1.2,
        'planet_equilibrium_temp': 1450,
        'stellar_effective_temp': 5778,
        'stellar_log_g': 4.4,
        'stellar_radius': 1.0,
        'ra': 290.5,
        'dec': 44.2
    }
    
    preprocessor = ExoplanetDataPreprocessor()
    
    try:
        # Test preprocessing for prediction
        X = preprocessor.preprocess_for_prediction(sample_data)
        print(f"\n✅ Preprocessed features shape: {X.shape}")
        print(f"✅ Feature values: {X[0]}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
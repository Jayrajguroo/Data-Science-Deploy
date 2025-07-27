import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
import logging
import os

class DataLoader:
    def __init__(self):
        self.dataset_name = "Wine Quality"
        self.X = None
        self.y = None
        self.feature_names = []
        self.target_name = ""
        self.dataset_info = {}
        
    def load_data(self):
        """Load and return the dataset"""
        try:
            # Load Wine dataset (classification converted to regression)
            wine = load_wine()
            
            # Convert to DataFrame for easier handling
            self.X = pd.DataFrame(wine.data, columns=wine.feature_names)
            
            # Use alcohol content as target for regression
            # (In real scenario, this would be a different target variable)
            self.y = self.X['alcohol'].values
            self.X = self.X.drop('alcohol', axis=1)
            
            self.feature_names = list(self.X.columns)
            self.target_name = "alcohol_content"
            
            # Create dataset info
            self.dataset_info = {
                'name': 'Wine Quality Dataset',
                'description': 'Wine characteristics dataset used to predict alcohol content',
                'n_samples': len(self.X),
                'n_features': len(self.feature_names),
                'target': self.target_name,
                'features': self.feature_names[:5],  # Show first 5 features
                'target_range': {
                    'min': float(self.y.min()),
                    'max': float(self.y.max()),
                    'mean': float(self.y.mean())
                }
            }
            
            logging.info(f"Loaded dataset with {len(self.X)} samples and {len(self.feature_names)} features")
            return self.X, self.y
            
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            # Fallback to synthetic data if real dataset fails
            return self._create_synthetic_data()
    
    def _create_synthetic_data(self):
        """Create synthetic dataset as fallback"""
        logging.info("Creating synthetic dataset as fallback...")
        
        np.random.seed(42)
        n_samples = 500
        n_features = 8
        
        # Create correlated features
        X = np.random.randn(n_samples, n_features)
        
        # Create target with some relationship to features
        y = (X[:, 0] * 2 + X[:, 1] * 1.5 + X[:, 2] * 0.8 + 
             np.random.randn(n_samples) * 0.5 + 10)
        
        # Create feature names
        self.feature_names = [f'feature_{i+1}' for i in range(n_features)]
        self.target_name = 'target_value'
        
        # Convert to DataFrame
        self.X = pd.DataFrame(X, columns=self.feature_names)
        self.y = y
        
        # Create dataset info
        self.dataset_info = {
            'name': 'Synthetic Regression Dataset',
            'description': 'Synthetic dataset for demonstration purposes',
            'n_samples': len(self.X),
            'n_features': len(self.feature_names),
            'target': self.target_name,
            'features': self.feature_names,
            'target_range': {
                'min': float(self.y.min()),
                'max': float(self.y.max()),
                'mean': float(self.y.mean())
            }
        }
        
        return self.X, self.y
    
    def get_feature_names(self):
        """Return feature names"""
        if not self.feature_names:
            self.load_data()
        return self.feature_names
    
    def get_dataset_info(self):
        """Return dataset information"""
        if not self.dataset_info:
            self.load_data()
        return self.dataset_info
    
    def get_sample_data(self, n_samples=5):
        """Get sample data for display"""
        if self.X is None:
            self.load_data()
        
        sample_df = self.X.head(n_samples).copy()
        sample_df[self.target_name] = self.y[:n_samples]
        
        return sample_df.to_dict('records')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import logging
import json

class MLPipeline:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.target_name = ''
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.results = {}
        self.best_model = None
        self.best_model_name = ''
        
    def train_models(self, X, y):
        """Train multiple models and compare performance"""
        logging.info("Starting model training pipeline...")
        
        # Store feature names
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        else:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Define models to train
        models_to_train = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Support Vector Regression': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        
        self.results = {}
        best_score = -np.inf
        
        for name, model in models_to_train.items():
            logging.info(f"Training {name}...")
            
            try:
                # Use scaled data for SVR, original data for tree-based models
                if name == 'Support Vector Regression' or name == 'Linear Regression':
                    X_train_model = self.X_train_scaled
                    X_test_model = self.X_test_scaled
                else:
                    X_train_model = self.X_train
                    X_test_model = self.X_test
                
                # Train the model
                model.fit(X_train_model, self.y_train)
                
                # Make predictions
                y_pred_train = model.predict(X_train_model)
                y_pred_test = model.predict(X_test_model)
                
                # Calculate metrics
                train_r2 = r2_score(self.y_train, y_pred_train)
                test_r2 = r2_score(self.y_test, y_pred_test)
                train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
                train_mae = mean_absolute_error(self.y_train, y_pred_train)
                test_mae = mean_absolute_error(self.y_test, y_pred_test)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train_model, self.y_train, cv=5, scoring='r2')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                # Store results
                self.results[name] = {
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'predictions_test': y_pred_test.tolist()
                }
                
                # Store the model
                self.models[name] = model
                
                # Track best model
                if test_r2 > best_score:
                    best_score = test_r2
                    self.best_model = model
                    self.best_model_name = name
                
                logging.info(f"{name} - Test R²: {test_r2:.4f}, Test RMSE: {test_rmse:.4f}")
                
            except Exception as e:
                logging.error(f"Error training {name}: {str(e)}")
                continue
        
        # Save models and scaler
        self._save_models()
        
        logging.info(f"Best model: {self.best_model_name} with R² score: {best_score:.4f}")
        return self.results
    
    def _save_models(self):
        """Save trained models and scaler to disk"""
        try:
            os.makedirs('saved_models', exist_ok=True)
            
            # Save all models
            for name, model in self.models.items():
                filename = f"saved_models/{name.replace(' ', '_').lower()}_model.joblib"
                joblib.dump(model, filename)
            
            # Save scaler
            joblib.dump(self.scaler, 'saved_models/scaler.joblib')
            
            # Save metadata
            metadata = {
                'feature_names': self.feature_names,
                'best_model_name': self.best_model_name,
                'results': self.results
            }
            
            with open('saved_models/metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logging.info("Models saved successfully")
            
        except Exception as e:
            logging.error(f"Error saving models: {str(e)}")
    
    def _load_models(self):
        """Load saved models from disk"""
        try:
            # Load metadata
            with open('saved_models/metadata.json', 'r') as f:
                metadata = json.load(f)
            
            self.feature_names = metadata['feature_names']
            self.best_model_name = metadata['best_model_name']
            self.results = metadata['results']
            
            # Load scaler
            self.scaler = joblib.load('saved_models/scaler.joblib')
            
            # Load best model
            filename = f"saved_models/{self.best_model_name.replace(' ', '_').lower()}_model.joblib"
            self.best_model = joblib.load(filename)
            
            logging.info("Models loaded successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")
            return False
    
    def predict_single(self, input_data):
        """Make a single prediction"""
        try:
            # Load models if not already loaded
            if self.best_model is None:
                if not self._load_models():
                    raise Exception("No trained models found. Please train models first.")
            
            # Convert input to DataFrame
            if isinstance(input_data, dict):
                df = pd.DataFrame([input_data])
            else:
                df = pd.DataFrame(input_data.reshape(1, -1), columns=self.feature_names)
            
            # Ensure all features are present
            for feature in self.feature_names:
                if feature not in df.columns:
                    raise ValueError(f"Missing feature: {feature}")
            
            # Reorder columns to match training data
            df = df[self.feature_names]
            
            # Scale if necessary
            if self.best_model_name in ['Support Vector Regression', 'Linear Regression']:
                input_scaled = self.scaler.transform(df)
                prediction = self.best_model.predict(input_scaled)[0]
            else:
                prediction = self.best_model.predict(df)[0]
            
            return {
                'prediction': float(prediction),
                'model_used': self.best_model_name,
                'confidence': 'High' if self.results[self.best_model_name]['test_r2'] > 0.8 else 'Medium' if self.results[self.best_model_name]['test_r2'] > 0.6 else 'Low'
            }
            
        except Exception as e:
            logging.error(f"Error making prediction: {str(e)}")
            raise
    
    def predict_batch(self, df):
        """Make batch predictions"""
        try:
            # Load models if not already loaded
            if self.best_model is None:
                if not self._load_models():
                    raise Exception("No trained models found. Please train models first.")
            
            # Ensure all features are present
            missing_features = set(self.feature_names) - set(df.columns)
            if missing_features:
                raise ValueError(f"Missing features: {', '.join(missing_features)}")
            
            # Reorder columns to match training data
            df_ordered = df[self.feature_names]
            
            # Scale if necessary
            if self.best_model_name in ['Support Vector Regression', 'Linear Regression']:
                input_scaled = self.scaler.transform(df_ordered)
                predictions = self.best_model.predict(input_scaled)
            else:
                predictions = self.best_model.predict(df_ordered)
            
            return predictions.tolist()
            
        except Exception as e:
            logging.error(f"Error making batch predictions: {str(e)}")
            raise
    
    def get_model_results(self):
        """Get model training results"""
        if not self.results:
            self._load_models()
        return self.results
    
    def get_feature_importance(self):
        """Get feature importance for the best model"""
        try:
            if self.best_model is None:
                if not self._load_models():
                    return None
            
            if hasattr(self.best_model, 'feature_importances_'):
                importance_scores = self.best_model.feature_importances_
                feature_importance = [
                    {'feature': feature, 'importance': float(score)}
                    for feature, score in zip(self.feature_names, importance_scores)
                ]
                # Sort by importance
                feature_importance.sort(key=lambda x: x['importance'], reverse=True)
                return feature_importance
            elif hasattr(self.best_model, 'coef_'):
                # For linear models, use absolute coefficient values
                coef_abs = np.abs(self.best_model.coef_)
                feature_importance = [
                    {'feature': feature, 'importance': float(score)}
                    for feature, score in zip(self.feature_names, coef_abs)
                ]
                # Sort by importance
                feature_importance.sort(key=lambda x: x['importance'], reverse=True)
                return feature_importance
            else:
                return None
                
        except Exception as e:
            logging.error(f"Error getting feature importance: {str(e)}")
            return None

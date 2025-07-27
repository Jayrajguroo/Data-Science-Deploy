import os
import logging
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
import pandas as pd
import numpy as np
from ml_pipeline import MLPipeline
from data_loader import DataLoader
import joblib
import json
from werkzeug.utils import secure_filename
import io

# Set up logging
logging.basicConfig(level=logging.DEBUG)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure upload settings
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_EXTENSIONS = ['.csv']

# Initialize ML Pipeline
ml_pipeline = MLPipeline()
data_loader = DataLoader()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['csv']

@app.route('/')
def index():
    """Home page with project overview"""
    try:
        # Get basic dataset info
        dataset_info = data_loader.get_dataset_info()
        return render_template('index.html', dataset_info=dataset_info)
    except Exception as e:
        logging.error(f"Error loading index page: {str(e)}")
        flash(f"Error loading dataset information: {str(e)}", "error")
        return render_template('index.html', dataset_info=None)

@app.route('/train_models')
def train_models():
    """Train all models and redirect to performance page"""
    try:
        # Load and preprocess data
        X, y = data_loader.load_data()
        
        # Train models
        results = ml_pipeline.train_models(X, y)
        
        flash("Models trained successfully!", "success")
        return redirect(url_for('model_performance'))
        
    except Exception as e:
        logging.error(f"Error training models: {str(e)}")
        flash(f"Error training models: {str(e)}", "error")
        return redirect(url_for('index'))

@app.route('/model_performance')
def model_performance():
    """Display model performance metrics and visualizations"""
    try:
        # Get model results
        results = ml_pipeline.get_model_results()
        
        if not results:
            flash("No trained models found. Please train models first.", "warning")
            return redirect(url_for('index'))
        
        # Get feature importance for the best model
        feature_importance = ml_pipeline.get_feature_importance()
        
        return render_template('model_performance.html', 
                             results=results, 
                             feature_importance=feature_importance)
        
    except Exception as e:
        logging.error(f"Error loading model performance: {str(e)}")
        flash(f"Error loading model performance: {str(e)}", "error")
        return redirect(url_for('index'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Single prediction interface"""
    if request.method == 'GET':
        # Get feature names for the form
        feature_names = data_loader.get_feature_names()
        return render_template('predict.html', feature_names=feature_names)
    
    try:
        # Get prediction data from form
        feature_names = data_loader.get_feature_names()
        input_data = {}
        
        for feature in feature_names:
            value = request.form.get(feature)
            if value is None or value == '':
                flash(f"Please provide a value for {feature}", "error")
                return render_template('predict.html', feature_names=feature_names)
            
            try:
                input_data[feature] = float(value)
            except ValueError:
                flash(f"Invalid value for {feature}. Please enter a number.", "error")
                return render_template('predict.html', feature_names=feature_names)
        
        # Make prediction
        prediction = ml_pipeline.predict_single(input_data)
        
        return render_template('predict.html', 
                             feature_names=feature_names,
                             prediction=prediction,
                             input_data=input_data)
        
    except Exception as e:
        logging.error(f"Error making prediction: {str(e)}")
        flash(f"Error making prediction: {str(e)}", "error")
        return render_template('predict.html', feature_names=data_loader.get_feature_names())

@app.route('/batch_predict', methods=['GET', 'POST'])
def batch_predict():
    """Batch prediction from CSV upload"""
    if request.method == 'GET':
        return render_template('batch_predict.html')
    
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            flash("No file uploaded", "error")
            return render_template('batch_predict.html')
        
        file = request.files['file']
        
        if file.filename == '':
            flash("No file selected", "error")
            return render_template('batch_predict.html')
        
        if not allowed_file(file.filename):
            flash("Invalid file type. Please upload a CSV file.", "error")
            return render_template('batch_predict.html')
        
        # Read CSV file
        df = pd.read_csv(file)
        
        # Make batch predictions
        predictions = ml_pipeline.predict_batch(df)
        
        # Prepare results
        results_df = df.copy()
        results_df['Prediction'] = predictions
        
        # Convert to list of dictionaries for template
        results = results_df.to_dict('records')
        
        return render_template('batch_predict.html', 
                             results=results,
                             total_predictions=len(results))
        
    except Exception as e:
        logging.error(f"Error in batch prediction: {str(e)}")
        flash(f"Error processing file: {str(e)}", "error")
        return render_template('batch_predict.html')

@app.route('/api/model_results')
def api_model_results():
    """API endpoint for model results (for charts)"""
    try:
        results = ml_pipeline.get_model_results()
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/feature_importance')
def api_feature_importance():
    """API endpoint for feature importance data"""
    try:
        feature_importance = ml_pipeline.get_feature_importance()
        return jsonify(feature_importance)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

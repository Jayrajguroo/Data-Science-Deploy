# ML Pipeline Dashboard

## Overview

This is a Flask-based machine learning web application that provides a complete end-to-end ML pipeline. The application demonstrates data preprocessing, model training, evaluation, and serving through a user-friendly web interface. It uses the Wine dataset to predict alcohol content and compares multiple regression models to find the best performer.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Flask with Jinja2 templating
- **UI Framework**: Bootstrap 5 with dark theme
- **JavaScript Libraries**: Chart.js for data visualization
- **Styling**: Custom CSS with dark theme support
- **Static Assets**: Organized in `/static` directory with separate folders for CSS and JS

### Backend Architecture
- **Framework**: Flask (Python web framework)
- **Database ORM**: SQLAlchemy with Flask-SQLAlchemy extension
- **ML Libraries**: scikit-learn for machine learning, pandas/numpy for data manipulation
- **Model Persistence**: joblib for saving/loading trained models
- **File Handling**: Werkzeug for secure file uploads

### Data Architecture
- **Primary Dataset**: Wine dataset from scikit-learn (modified for regression)
- **Target Variable**: Alcohol content prediction
- **Data Processing**: StandardScaler for feature normalization
- **Train/Test Split**: 80/20 split with stratified sampling

## Key Components

### Core Application (`app.py`)
- Main Flask application setup and configuration
- Route definitions and request handling
- Integration between ML pipeline and web interface
- File upload handling with security measures

### ML Pipeline (`ml_pipeline.py`)
- Model training orchestration
- Multiple algorithm comparison (Linear Regression, Random Forest, Gradient Boosting, SVR)
- Cross-validation and performance evaluation
- Model persistence and loading

### Data Management (`data_loader.py`)
- Dataset loading and preprocessing
- Feature engineering and target variable creation
- Dataset information and statistics generation

### Database Models (`models.py`)
- `PredictionLog`: Tracks individual predictions for monitoring
- `ModelMetrics`: Stores model performance metrics over time
- SQLAlchemy declarative base implementation

### Web Interface Templates
- **Base Template**: Common layout with Bootstrap navigation
- **Index**: Dashboard overview and dataset information
- **Model Performance**: Comprehensive model comparison with charts
- **Single Prediction**: Form-based individual predictions
- **Batch Prediction**: CSV upload for bulk predictions

## Data Flow

1. **Data Loading**: Wine dataset loaded and preprocessed with alcohol content as target
2. **Model Training**: Multiple regression models trained with cross-validation
3. **Performance Evaluation**: Models compared using R², RMSE, and MAE metrics
4. **Model Selection**: Best performing model automatically selected
5. **Prediction Serving**: Web interface allows single and batch predictions
6. **Logging**: All predictions and model metrics stored in database

## External Dependencies

### Python Libraries
- **Flask**: Web framework and templating
- **SQLAlchemy**: Database ORM and management
- **scikit-learn**: Machine learning algorithms and utilities
- **pandas/numpy**: Data manipulation and numerical computing
- **joblib**: Model serialization and persistence

### Frontend Libraries
- **Bootstrap 5**: UI framework with dark theme
- **Chart.js**: Interactive data visualization
- **Font Awesome**: Icon library for UI enhancement

### Development Tools
- **Werkzeug**: WSGI utilities and security features
- **Flask-SQLAlchemy**: Flask-specific SQLAlchemy integration

## Deployment Strategy

### Development Setup
- **Entry Point**: `main.py` runs Flask development server
- **Configuration**: Environment-based configuration with development defaults
- **Debug Mode**: Enabled for development with detailed error logging
- **Host/Port**: Configured for 0.0.0.0:5000 for containerized environments

### Production Considerations
- **WSGI Integration**: ProxyFix middleware for reverse proxy compatibility
- **Security**: Secret key management via environment variables
- **File Upload Limits**: 16MB maximum file size with CSV validation
- **Error Handling**: Comprehensive logging and user-friendly error messages

### Database Strategy
- **Development**: SQLite (default SQLAlchemy configuration)
- **Production Ready**: Designed to work with PostgreSQL when needed
- **Schema Management**: Declarative base for easy migrations
- **Data Persistence**: Model artifacts saved to filesystem, metadata in database

The application is designed to be easily deployable on platforms like Replit, with environment-based configuration and containerization support.
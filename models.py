from app import db

class PredictionLog(db.Model):
    """Log predictions for monitoring and analysis"""
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
    model_name = db.Column(db.String(100), nullable=False)
    input_features = db.Column(db.Text)  # JSON string of input features
    prediction = db.Column(db.Float, nullable=False)
    confidence_score = db.Column(db.Float)
    
    def __repr__(self):
        return f'<PredictionLog {self.id}: {self.model_name}>'

class ModelMetrics(db.Model):
    """Store model performance metrics"""
    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(100), nullable=False)
    metric_name = db.Column(db.String(50), nullable=False)
    metric_value = db.Column(db.Float, nullable=False)
    training_timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
    
    def __repr__(self):
        return f'<ModelMetrics {self.model_name}: {self.metric_name}={self.metric_value}>'

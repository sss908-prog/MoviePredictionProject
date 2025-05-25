import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

def format_currency(amount: float) -> str:
    """
    Format a number as currency.
    
    Args:
        amount: Numeric amount to format
        
    Returns:
        Formatted currency string
    """
    return f"${amount:,.2f}"

def format_runtime(minutes: int) -> str:
    """
    Format runtime in minutes to hours and minutes.
    
    Args:
        minutes: Runtime in minutes
        
    Returns:
        Formatted runtime string
    """
    hours = minutes // 60
    mins = minutes % 60
    
    if hours > 0:
        return f"{hours}h {mins}m"
    else:
        return f"{mins}m"

def format_rating(rating: float) -> str:
    """
    Format rating with appropriate precision.
    
    Args:
        rating: Numeric rating
        
    Returns:
        Formatted rating string
    """
    return f"{rating:.1f}/10"

def get_rating_category(rating: float) -> Dict[str, str]:
    """
    Get rating category and color based on numeric rating.
    
    Args:
        rating: Numeric rating (1-10)
        
    Returns:
        Dictionary with category and color class
    """
    if rating >= 8.0:
        return {'category': 'Excellent', 'color': 'success'}
    elif rating >= 7.0:
        return {'category': 'Good', 'color': 'primary'}
    elif rating >= 6.0:
        return {'category': 'Average', 'color': 'warning'}
    elif rating >= 4.0:
        return {'category': 'Below Average', 'color': 'secondary'}
    else:
        return {'category': 'Poor', 'color': 'danger'}

def validate_file_upload(file, allowed_extensions: List[str], max_size: int = 5 * 1024 * 1024) -> bool:
    """
    Validate uploaded file.
    
    Args:
        file: Uploaded file object
        allowed_extensions: List of allowed file extensions
        max_size: Maximum file size in bytes
        
    Returns:
        True if file is valid, False otherwise
    """
    if not file or not file.filename:
        return False
    
    # Check file extension
    if '.' not in file.filename:
        return False
    
    extension = file.filename.rsplit('.', 1)[1].lower()
    if extension not in allowed_extensions:
        return False
    
    # Check file size (if possible)
    try:
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(0)
        
        if size > max_size:
            return False
    except:
        # If we can't check size, allow the file
        pass
    
    return True

def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert value to float.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Float value or default
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value: Any, default: int = 0) -> int:
    """
    Safely convert value to integer.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Integer value or default
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def log_user_action(action: str, details: Dict[str, Any] = None):
    """
    Log user actions for analytics.
    
    Args:
        action: Action name
        details: Additional details dictionary
    """
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'action': action,
        'details': details or {}
    }
    
    logger.info(f"User action: {json.dumps(log_entry)}")

def calculate_confidence_interval(predictions: List[float], confidence: float = 0.95) -> Dict[str, float]:
    """
    Calculate confidence interval for predictions.
    
    Args:
        predictions: List of prediction values
        confidence: Confidence level (default 0.95)
        
    Returns:
        Dictionary with lower and upper bounds
    """
    import numpy as np
    from scipy import stats
    
    if not predictions:
        return {'lower': 0, 'upper': 0}
    
    mean = np.mean(predictions)
    std_err = stats.sem(predictions)
    
    # Calculate confidence interval
    h = std_err * stats.t.ppf((1 + confidence) / 2., len(predictions) - 1)
    
    return {
        'lower': max(1.0, mean - h),
        'upper': min(10.0, mean + h)
    }

def generate_chart_data(model_performance: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """
    Generate data for performance charts.
    
    Args:
        model_performance: Model performance metrics
        
    Returns:
        Chart data dictionary
    """
    models = list(model_performance.keys())
    
    # Format model names for display
    model_labels = [name.replace('_', ' ').title() for name in models]
    
    # Extract metrics
    r2_scores = [model_performance[model]['r2'] for model in models]
    rmse_scores = [model_performance[model]['rmse'] for model in models]
    
    return {
        'labels': model_labels,
        'r2_scores': r2_scores,
        'rmse_scores': rmse_scores,
        'models': models
    }

def create_prediction_summary(predictions: Dict[str, float], best_model: str) -> Dict[str, Any]:
    """
    Create a summary of predictions from multiple models.
    
    Args:
        predictions: Dictionary of model predictions
        best_model: Name of the best performing model
        
    Returns:
        Prediction summary dictionary
    """
    if not predictions:
        return {}
    
    valid_predictions = {k: v for k, v in predictions.items() if v is not None}
    
    if not valid_predictions:
        return {}
    
    values = list(valid_predictions.values())
    
    summary = {
        'best_prediction': predictions.get(best_model, 0),
        'average_prediction': sum(values) / len(values),
        'min_prediction': min(values),
        'max_prediction': max(values),
        'prediction_range': max(values) - min(values),
        'model_count': len(valid_predictions)
    }
    
    # Add rating categories
    summary['best_category'] = get_rating_category(summary['best_prediction'])
    summary['avg_category'] = get_rating_category(summary['average_prediction'])
    
    return summary

# Template filters for Jinja2
def register_template_filters(app):
    """
    Register custom template filters.
    
    Args:
        app: Flask application instance
    """
    app.jinja_env.filters['currency'] = format_currency
    app.jinja_env.filters['runtime'] = format_runtime
    app.jinja_env.filters['rating'] = format_rating
    app.jinja_env.filters['safe_float'] = safe_float
    app.jinja_env.filters['safe_int'] = safe_int
    app.jinja_env.filters['get_rating_category'] = get_rating_category

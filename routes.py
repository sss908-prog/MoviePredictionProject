from flask import render_template, request, flash, redirect, url_for, jsonify
from app import app
from ml_models import predictor
from data_processor import data_processor
import logging

logger = logging.getLogger(__name__)

@app.route('/')
def index():
    """
    Home page with overview of the movie rating prediction system.
    """
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Movie rating prediction page.
    """
    if request.method == 'GET':
        # Display the prediction form
        feature_options = data_processor.get_feature_options()
        feature_stats = data_processor.get_feature_statistics()
        return render_template('predict.html', 
                             feature_options=feature_options,
                             feature_stats=feature_stats)
    
    elif request.method == 'POST':
        try:
            # Extract features from form
            movie_features = data_processor.extract_features_from_form(request.form)
            
            # Ensure models are trained
            if not predictor.is_trained:
                logger.info("Models not trained, training now...")
                success = predictor.train_models()
                if not success:
                    flash('Error training models. Please try again.', 'error')
                    return redirect(url_for('predict'))
            
            # Get predictions from all models
            predictions = {}
            for model_name in predictor.trained_models.keys():
                try:
                    pred = predictor.predict(movie_features, model_name)
                    predictions[model_name] = round(pred, 2)
                except Exception as e:
                    logger.error(f"Error predicting with {model_name}: {str(e)}")
                    predictions[model_name] = None
            
            # Get best model prediction
            best_prediction = predictions.get(predictor.best_model_name, 0)
            
            # Create feature summary
            feature_summary = data_processor.create_feature_summary(movie_features)
            
            # Get similar movies
            similar_movies = data_processor.suggest_similar_movies(movie_features)
            
            # Get model performance
            model_performance = predictor.get_model_performance()
            
            return render_template('predict.html',
                                 feature_options=data_processor.get_feature_options(),
                                 feature_stats=data_processor.get_feature_statistics(),
                                 predictions=predictions,
                                 best_prediction=best_prediction,
                                 best_model=predictor.best_model_name,
                                 feature_summary=feature_summary,
                                 similar_movies=similar_movies,
                                 model_performance=model_performance,
                                 show_results=True)
            
        except ValueError as e:
            flash(f'Input validation error: {str(e)}', 'error')
            return redirect(url_for('predict'))
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            flash('An error occurred during prediction. Please try again.', 'error')
            return redirect(url_for('predict'))

@app.route('/dashboard')
def dashboard():
    """
    Model performance dashboard.
    """
    try:
        # Ensure models are trained
        if not predictor.is_trained:
            logger.info("Models not trained, training now...")
            success = predictor.train_models()
            if not success:
                flash('Error training models. Please check the logs.', 'error')
                return redirect(url_for('index'))
        
        # Get model performance metrics
        model_performance = predictor.get_model_performance()
        best_model = predictor.best_model_name
        
        # Calculate some additional statistics
        stats = {
            'total_models': len(model_performance),
            'best_model': best_model,
            'best_r2': model_performance[best_model]['r2'] if best_model else 0,
            'avg_rmse': sum(perf['rmse'] for perf in model_performance.values()) / len(model_performance) if model_performance else 0
        }
        
        return render_template('dashboard.html',
                             model_performance=model_performance,
                             stats=stats)
        
    except Exception as e:
        logger.error(f"Dashboard error: {str(e)}")
        flash('Error loading dashboard. Please try again.', 'error')
        return redirect(url_for('index'))

@app.route('/retrain', methods=['POST'])
def retrain_models():
    """
    Retrain all models with fresh data.
    """
    try:
        logger.info("Retraining models...")
        success = predictor.train_models()
        
        if success:
            flash('Models retrained successfully!', 'success')
        else:
            flash('Error during model retraining.', 'error')
            
    except Exception as e:
        logger.error(f"Retraining error: {str(e)}")
        flash('Error during model retraining.', 'error')
    
    return redirect(url_for('dashboard'))

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors."""
    return render_template('index.html', error_message="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal error: {str(error)}")
    return render_template('index.html', error_message="Internal server error"), 500

# Initialize models on startup with real data
try:
    logger.info("Initializing movie rating predictor with real dataset...")
    if not predictor.is_trained:
        predictor.train_models()
except Exception as e:
    logger.error(f"Error during initialization: {str(e)}")

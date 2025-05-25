import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class MovieRatingPredictor:
    """
    A comprehensive movie rating prediction system using multiple ML models.
    """
    
    def __init__(self):
        self.models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        self.model_performance = {}
        self.best_model_name = None
        self.trained_models = {}
        
    def create_synthetic_dataset(self, n_samples=1000):
        """
        Create a synthetic movie dataset for demonstration purposes.
        In a real application, this would load actual movie data.
        """
        np.random.seed(42)
        
        # Define possible values for categorical features
        genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller']
        directors = ['Christopher Nolan', 'Steven Spielberg', 'Martin Scorsese', 'Quentin Tarantino', 
                    'James Cameron', 'Tim Burton', 'Ridley Scott', 'David Fincher']
        studios = ['Warner Bros', 'Disney', 'Universal', 'Paramount', 'Sony', 'Fox', 'MGM']
        
        data = []
        for _ in range(n_samples):
            # Generate features
            budget = np.random.uniform(1, 300)  # Million dollars
            runtime = np.random.uniform(80, 180)  # Minutes
            year = np.random.randint(1990, 2024)
            genre = np.random.choice(genres)
            director = np.random.choice(directors)
            studio = np.random.choice(studios)
            
            # Create realistic rating based on features (with some noise)
            base_rating = 5.0
            
            # Budget impact (diminishing returns)
            base_rating += min(budget / 100, 2.0)
            
            # Runtime impact (optimal around 120 minutes)
            runtime_factor = 1 - abs(runtime - 120) / 100
            base_rating += runtime_factor * 0.5
            
            # Year impact (slight modern bias)
            year_factor = (year - 1990) / 34
            base_rating += year_factor * 0.5
            
            # Genre impact
            genre_bonus = {'Action': 0.2, 'Comedy': -0.1, 'Drama': 0.3, 'Horror': -0.2, 
                          'Romance': 0.1, 'Sci-Fi': 0.15, 'Thriller': 0.05}
            base_rating += genre_bonus.get(genre, 0)
            
            # Director impact
            director_bonus = {'Christopher Nolan': 1.0, 'Steven Spielberg': 0.8, 'Martin Scorsese': 0.9,
                             'Quentin Tarantino': 0.7, 'James Cameron': 0.6, 'Tim Burton': 0.4,
                             'Ridley Scott': 0.5, 'David Fincher': 0.6}
            base_rating += director_bonus.get(director, 0)
            
            # Add noise and clamp between 1-10
            rating = max(1.0, min(10.0, base_rating + np.random.normal(0, 0.5)))
            
            data.append({
                'budget_millions': budget,
                'runtime_minutes': runtime,
                'release_year': year,
                'genre': genre,
                'director': director,
                'studio': studio,
                'rating': rating
            })
        
        return pd.DataFrame(data)
    
    def prepare_features(self, df):
        """
        Prepare features for training/prediction.
        """
        # Create feature matrix
        features = pd.DataFrame()
        
        # Numerical features
        features['budget_millions'] = df['budget_millions']
        features['runtime_minutes'] = df['runtime_minutes']
        features['release_year'] = df['release_year']
        
        # One-hot encode categorical features
        genre_dummies = pd.get_dummies(df['genre'], prefix='genre')
        director_dummies = pd.get_dummies(df['director'], prefix='director')
        studio_dummies = pd.get_dummies(df['studio'], prefix='studio')
        
        features = pd.concat([features, genre_dummies, director_dummies, studio_dummies], axis=1)
        
        return features
    
    def train_models(self, df=None):
        """
        Train all models on the provided dataset.
        """
        try:
            if df is None:
                logger.info("No dataset provided, creating synthetic dataset")
                df = self.create_synthetic_dataset()
            
            logger.info(f"Training models on dataset with {len(df)} samples")
            
            # Prepare features and target
            X = self.prepare_features(df)
            y = df['rating']
            
            self.feature_names = X.columns.tolist()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train and evaluate each model
            for name, model in self.models.items():
                logger.info(f"Training {name}...")
                
                # Train model
                if name == 'linear_regression':
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Store trained model
                self.trained_models[name] = model
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                # Cross-validation score
                if name == 'linear_regression':
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
                else:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                
                self.model_performance[name] = {
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,
                    'mae': mae,
                    'cv_score_mean': cv_scores.mean(),
                    'cv_score_std': cv_scores.std()
                }
                
                logger.info(f"{name} - RMSE: {rmse:.3f}, R²: {r2:.3f}")
            
            # Find best model based on R² score
            best_r2 = -np.inf
            for name, perf in self.model_performance.items():
                if perf['r2'] > best_r2:
                    best_r2 = perf['r2']
                    self.best_model_name = name
            
            self.is_trained = True
            logger.info(f"Training completed. Best model: {self.best_model_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return False
    
    def predict(self, movie_features, model_name=None):
        """
        Predict rating for a single movie.
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found")
        
        try:
            # Create DataFrame from features
            df = pd.DataFrame([movie_features])
            
            # Prepare features (ensure all expected columns are present)
            X = self.prepare_features(df)
            
            # Ensure all feature columns are present
            missing_cols = set(self.feature_names) - set(X.columns)
            for col in missing_cols:
                X[col] = 0
            
            # Reorder columns to match training data
            X = X[self.feature_names]
            
            model = self.trained_models[model_name]
            
            # Scale features for linear regression
            if model_name == 'linear_regression':
                X_scaled = self.scaler.transform(X)
                prediction = model.predict(X_scaled)[0]
            else:
                prediction = model.predict(X)[0]
            
            # Clamp prediction between 1 and 10
            prediction = max(1.0, min(10.0, prediction))
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def get_model_performance(self):
        """
        Get performance metrics for all trained models.
        """
        return self.model_performance
    
    def save_models(self, filepath='models'):
        """
        Save trained models to disk.
        """
        if not self.is_trained:
            raise ValueError("No trained models to save")
        
        try:
            os.makedirs(filepath, exist_ok=True)
            
            # Save models
            for name, model in self.trained_models.items():
                joblib.dump(model, os.path.join(filepath, f'{name}.joblib'))
            
            # Save scaler and metadata
            joblib.dump(self.scaler, os.path.join(filepath, 'scaler.joblib'))
            joblib.dump({
                'feature_names': self.feature_names,
                'model_performance': self.model_performance,
                'best_model_name': self.best_model_name
            }, os.path.join(filepath, 'metadata.joblib'))
            
            logger.info(f"Models saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise
    
    def load_models(self, filepath='models'):
        """
        Load trained models from disk.
        """
        try:
            # Load metadata
            metadata = joblib.load(os.path.join(filepath, 'metadata.joblib'))
            self.feature_names = metadata['feature_names']
            self.model_performance = metadata['model_performance']
            self.best_model_name = metadata['best_model_name']
            
            # Load scaler
            self.scaler = joblib.load(os.path.join(filepath, 'scaler.joblib'))
            
            # Load models
            for name in self.models.keys():
                model_path = os.path.join(filepath, f'{name}.joblib')
                if os.path.exists(model_path):
                    self.trained_models[name] = joblib.load(model_path)
            
            self.is_trained = True
            logger.info(f"Models loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

# Global predictor instance
predictor = MovieRatingPredictor()

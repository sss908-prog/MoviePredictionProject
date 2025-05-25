import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class MovieDataProcessor:
    """
    Handles data preprocessing and feature engineering for movie rating prediction.
    """
    
    def __init__(self):
        self.genre_options = [
            'Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 
            'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 
            'History', 'Horror', 'Music', 'Mystery', 'Romance', 
            'Sci-Fi', 'Sport', 'Thriller', 'War', 'Western'
        ]
        
        self.director_options = [
            'Christopher Nolan', 'Steven Spielberg', 'Martin Scorsese', 
            'Quentin Tarantino', 'James Cameron', 'Tim Burton', 
            'Ridley Scott', 'David Fincher', 'Peter Jackson', 
            'Denis Villeneuve', 'Jordan Peele', 'Greta Gerwig',
            'Chloe Zhao', 'Rian Johnson', 'Patty Jenkins', 'Other'
        ]
        
        self.studio_options = [
            'Warner Bros', 'Disney', 'Universal Pictures', 'Paramount Pictures',
            'Sony Pictures', '20th Century Fox', 'MGM', 'Lionsgate',
            'A24', 'Netflix', 'Amazon Studios', 'Apple Studios', 'Other'
        ]
        
    def validate_movie_input(self, movie_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean movie input data.
        
        Args:
            movie_data: Dictionary containing movie features
            
        Returns:
            Dictionary with validated and cleaned data
            
        Raises:
            ValueError: If required fields are missing or invalid
        """
        required_fields = ['budget_millions', 'runtime_minutes', 'release_year', 'genre', 'director', 'studio']
        
        # Check for required fields
        for field in required_fields:
            if field not in movie_data:
                raise ValueError(f"Missing required field: {field}")
        
        validated_data = {}
        
        try:
            # Validate numerical fields
            budget = float(movie_data['budget_millions'])
            if budget < 0 or budget > 1000:
                raise ValueError("Budget must be between 0 and 1000 million dollars")
            validated_data['budget_millions'] = budget
            
            runtime = float(movie_data['runtime_minutes'])
            if runtime < 30 or runtime > 300:
                raise ValueError("Runtime must be between 30 and 300 minutes")
            validated_data['runtime_minutes'] = runtime
            
            year = int(movie_data['release_year'])
            if year < 1900 or year > 2030:
                raise ValueError("Release year must be between 1900 and 2030")
            validated_data['release_year'] = year
            
            # Validate categorical fields
            genre = str(movie_data['genre']).strip()
            if genre not in self.genre_options:
                raise ValueError(f"Invalid genre. Must be one of: {', '.join(self.genre_options)}")
            validated_data['genre'] = genre
            
            director = str(movie_data['director']).strip()
            if director not in self.director_options:
                validated_data['director'] = 'Other'
            else:
                validated_data['director'] = director
            
            studio = str(movie_data['studio']).strip()
            if studio not in self.studio_options:
                validated_data['studio'] = 'Other'
            else:
                validated_data['studio'] = studio
            
            return validated_data
            
        except (ValueError, TypeError) as e:
            logger.error(f"Data validation error: {str(e)}")
            raise ValueError(f"Invalid input data: {str(e)}")
    
    def get_feature_options(self) -> Dict[str, List[str]]:
        """
        Get available options for categorical features.
        
        Returns:
            Dictionary mapping feature names to their possible values
        """
        return {
            'genres': self.genre_options,
            'directors': self.director_options,
            'studios': self.studio_options
        }
    
    def extract_features_from_form(self, form_data) -> Dict[str, Any]:
        """
        Extract and convert form data to movie features.
        
        Args:
            form_data: Flask form data object
            
        Returns:
            Dictionary containing extracted features
        """
        try:
            features = {
                'budget_millions': float(form_data.get('budget', 0)),
                'runtime_minutes': float(form_data.get('runtime', 0)),
                'release_year': 2024,  # Default to current year
                'genre': form_data.get('genre', ''),
                'director': 'Other',  # Default director
                'studio': 'Universal Pictures'  # Default studio
            }
            
            return self.validate_movie_input(features)
            
        except Exception as e:
            logger.error(f"Error extracting features from form: {str(e)}")
            raise ValueError(f"Error processing form data: {str(e)}")
    
    def get_feature_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for numerical features to help with form validation.
        
        Returns:
            Dictionary containing min/max/suggested values for numerical features
        """
        return {
            'budget_millions': {
                'min': 0.1,
                'max': 500,
                'suggested': 50,
                'description': 'Production budget in millions of USD'
            },
            'runtime_minutes': {
                'min': 60,
                'max': 240,
                'suggested': 120,
                'description': 'Movie duration in minutes'
            },
            'release_year': {
                'min': 1990,
                'max': 2030,
                'suggested': 2024,
                'description': 'Year of release'
            }
        }
    
    def create_feature_summary(self, movie_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Create a human-readable summary of movie features.
        
        Args:
            movie_data: Dictionary containing movie features
            
        Returns:
            Dictionary with formatted feature descriptions
        """
        try:
            summary = {}
            
            summary['Budget'] = f"${movie_data['budget_millions']:.1f} million"
            summary['Runtime'] = f"{movie_data['runtime_minutes']:.0f} minutes"
            summary['Release Year'] = str(movie_data['release_year'])
            summary['Genre'] = movie_data['genre']
            summary['Director'] = movie_data['director']
            summary['Studio'] = movie_data['studio']
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating feature summary: {str(e)}")
            return {}
    
    def suggest_similar_movies(self, movie_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Suggest similar movies based on input features.
        This is a simplified implementation for demonstration.
        
        Args:
            movie_data: Dictionary containing movie features
            
        Returns:
            List of similar movie suggestions
        """
        # This would typically query a database of movies
        # For now, return some example suggestions based on genre
        
        genre_examples = {
            'Action': [
                {'title': 'Mad Max: Fury Road', 'rating': 8.1, 'year': 2015},
                {'title': 'John Wick', 'rating': 7.4, 'year': 2014}
            ],
            'Comedy': [
                {'title': 'The Grand Budapest Hotel', 'rating': 8.1, 'year': 2014},
                {'title': 'Parasite', 'rating': 8.6, 'year': 2019}
            ],
            'Drama': [
                {'title': 'The Shawshank Redemption', 'rating': 9.3, 'year': 1994},
                {'title': 'Moonlight', 'rating': 7.4, 'year': 2016}
            ],
            'Sci-Fi': [
                {'title': 'Blade Runner 2049', 'rating': 8.0, 'year': 2017},
                {'title': 'Arrival', 'rating': 7.9, 'year': 2016}
            ]
        }
        
        genre = movie_data.get('genre', 'Drama')
        return genre_examples.get(genre, genre_examples['Drama'])

# Global data processor instance
data_processor = MovieDataProcessor()

# app.py
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from model_training import PremierLeaguePredictor
import json

app = Flask(__name__)
CORS(app)

predictor = PremierLeaguePredictor()
# Load the pre-trained model instead of training
try:
    predictor.load_model('premier_league_model.pkl')
    predictor.load_data('data')  # Load historical data for feature calculation
    print(f"Model loaded successfully! {len(predictor.team_encoding)} teams available.")
except Exception as e:
    print(f"Failed to load model: {e}")
    print("Please train the model first by running: python model_training.py")

@app.route('/api/predict-match', methods=['POST', 'GET'])
def predict_match():
    """Predict match winner between two teams
    
    POST request body:
    {
        "home_team": "Arsenal",
        "away_team": "Liverpool"
    }
    
    Or GET request with query params:
    ?home_team=Arsenal&away_team=Liverpool
    """
    if predictor.model is None:
        return jsonify({
            'error': 'Model not loaded',
            'message': 'Please train the model first by running: python model_training.py'
        }), 500
    
    try:
        if request.method == 'POST':
            data = request.json
            if not data:
                return jsonify({'error': 'No JSON data provided'}), 400
            home_team = data.get('home_team')
            away_team = data.get('away_team')
        else:  # GET
            home_team = request.args.get('home_team')
            away_team = request.args.get('away_team')
        
        if not home_team or not away_team:
            return jsonify({
                'error': 'Missing team names',
                'message': 'Please provide both home_team and away_team parameters'
            }), 400
        
        # Get available teams for validation
        available_teams = list(predictor.team_encoding.keys())
        
        # Check if teams exist (case-insensitive matching)
        home_match = None
        away_match = None
        
        for team in available_teams:
            if team.lower() == home_team.lower():
                home_match = team
            if team.lower() == away_team.lower():
                away_match = team
        
        if not home_match:
            return jsonify({
                'error': f'Home team "{home_team}" not found',
                'available_teams': available_teams[:10],  # Show first 10 as hint
                'message': f'Please use exact team names. Found {len(available_teams)} teams.'
            }), 400
        
        if not away_match:
            return jsonify({
                'error': f'Away team "{away_team}" not found',
                'available_teams': available_teams[:10],
                'message': f'Please use exact team names. Found {len(available_teams)} teams.'
            }), 400
        
        # Make prediction
        prediction = predictor.predict_match(home_match, away_match)
        
        # Format response
        response = {
            'match': {
                'home_team': home_match,
                'away_team': away_match
            },
            'prediction': prediction['predicted_result'],
            'probabilities': {
                'home_win': round(prediction['predictions']['home_win'] * 100, 2),
                'draw': round(prediction['predictions']['draw'] * 100, 2),
                'away_win': round(prediction['predictions']['away_win'] * 100, 2)
            },
            'detailed_probabilities': prediction['predictions']
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

@app.route('/api/predict-week', methods=['GET'])
def predict_week():
    # TODO: Implement API to fetch upcoming fixtures
    # For now, return empty list - will be implemented later
    return jsonify({"message": "Weekly prediction will be implemented after model training", "predictions": []})

@app.route('/api/teams', methods=['GET'])
def get_teams():
    """Get list of all available teams"""
    if predictor.model is None:
        return jsonify({
            'error': 'Model not loaded',
            'message': 'Please train the model first'
        }), 500
    
    teams = sorted(list(predictor.team_encoding.keys()))
    return jsonify({
        'teams': teams,
        'count': len(teams)
    })

@app.route('/api/season-winner', methods=['GET'])
def predict_season_winner():
    """Predict season winner based on Monte Carlo simulation"""
    if predictor.model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    teams = predictor.team_encoding.keys()
    simulations = 1000
    wins = {team: 0 for team in teams}
    
    for _ in range(simulations):
        winner = simulate_season(teams, predictor)
        wins[winner] += 1
    
    probabilities = {team: count/simulations for team, count in wins.items()}
    return jsonify(sorted(probabilities.items(), key=lambda x: x[1], reverse=True))

def simulate_season(teams, predictor):
    # Simplified season simulation
    points = {team: 0 for team in teams}
    # Add your season simulation logic here
    return max(points, key=points.get)

if __name__ == '__main__':
    app.run(debug=True)
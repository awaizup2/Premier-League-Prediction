# Premier League Prediction API

## Why Training Was Fast

The model trained quickly for several reasons:

1. **Reduced Training Set**: Out of 5,790 matches, only 2,280 were used for training. This is because:
   - The feature calculation uses **historical data only** (data before each match)
   - Early matches in the dataset have no historical data, so they were filtered out
   - This prevents data leakage but reduces the training set size

2. **Small Feature Set**: Only 17 features were used, which is relatively small for a machine learning model

3. **Random Forest Efficiency**: Random Forest with 200 estimators is computationally efficient for datasets of this size (2,280 samples × 17 features)

4. **No Complex Preprocessing**: The features are relatively straightforward to calculate

## API Endpoints

### 1. Predict Match Winner
**POST** `/api/predict-match` or **GET** `/api/predict-match`

Predict the winner between two teams.

**POST Request Body:**
```json
{
  "home_team": "Arsenal",
  "away_team": "Liverpool"
}
```

**GET Request:**
```
GET /api/predict-match?home_team=Arsenal&away_team=Liverpool
```

**Response:**
```json
{
  "match": {
    "home_team": "Arsenal",
    "away_team": "Liverpool"
  },
  "prediction": "Home Win",
  "probabilities": {
    "home_win": 45.23,
    "draw": 28.15,
    "away_win": 26.62
  },
  "detailed_probabilities": {
    "home_win": 0.4523,
    "draw": 0.2815,
    "away_win": 0.2662
  }
}
```

### 2. Get Available Teams
**GET** `/api/teams`

Get a list of all teams available in the model.

**Response:**
```json
{
  "teams": ["Arsenal", "Chelsea", "Liverpool", ...],
  "count": 33
}
```

## Usage Examples

### Using cURL (POST):
```bash
curl -X POST http://localhost:5000/api/predict-match \
  -H "Content-Type: application/json" \
  -d '{"home_team": "Arsenal", "away_team": "Liverpool"}'
```

### Using cURL (GET):
```bash
curl "http://localhost:5000/api/predict-match?home_team=Arsenal&away_team=Liverpool"
```

### Using Python:
```python
import requests

# POST request
response = requests.post('http://localhost:5000/api/predict-match', 
                         json={'home_team': 'Arsenal', 'away_team': 'Liverpool'})
print(response.json())

# GET request
response = requests.get('http://localhost:5000/api/predict-match',
                       params={'home_team': 'Arsenal', 'away_team': 'Liverpool'})
print(response.json())
```

### Using JavaScript/Fetch:
```javascript
fetch('http://localhost:5000/api/predict-match', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ home_team: 'Arsenal', away_team: 'Liverpool' })
})
.then(res => res.json())
.then(data => console.log(data));
```

## Running the API

1. First, train the model:
   ```bash
   python model_training.py
   ```

2. Then start the API server:
   ```bash
   python app.py
   ```

The server will run on `http://localhost:5000`

## Notes

- Team names are **case-insensitive** but must match the exact team names in the dataset
- Use `/api/teams` to see all available team names
- The model uses historical match data to calculate features (form, average goals, etc.)


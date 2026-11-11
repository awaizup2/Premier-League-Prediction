# model_training.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import pickle
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Try to import XGBoost (optional)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Note: XGBoost not available. Install with: pip install xgboost")

# Create plots directory
if not os.path.exists('plots'):
    os.makedirs('plots')

class PremierLeaguePredictor:
    def __init__(self):
        self.model = None
        self.team_encoding = {}
        self.historical_data = None
        
    def load_data(self, data_folder='data'):
        """Load all historical Premier League data from CSV files in the data folder"""
        csv_files = glob.glob(os.path.join(data_folder, 'PL_*.csv'))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_folder} folder")
        
        # Load and combine all CSV files
        dataframes = []
        for file_path in sorted(csv_files):
            df = pd.read_csv(file_path)
            dataframes.append(df)
        
        # Combine all dataframes into one
        combined_df = pd.concat(dataframes, ignore_index=True)
        self.historical_data = combined_df
        return combined_df
    
    def calculate_features(self, df):
        """Calculate advanced features for prediction using match statistics
        Assumes df is already sorted by Date chronologically
        """
        features = []
        
        for idx, match in df.iterrows():
            # Filter data up to current match for accurate feature calculation
            historical_data = df.iloc[:idx] if idx > 0 else pd.DataFrame()
            
            feature_dict = {
                'home_team_avg_goals': self.get_team_avg_goals(match['HomeTeam'], historical_data, 'home'),
                'away_team_avg_goals': self.get_team_avg_goals(match['AwayTeam'], historical_data, 'away'),
                'home_team_avg_conceded': self.get_team_avg_conceded(match['HomeTeam'], historical_data, 'home'),
                'away_team_avg_conceded': self.get_team_avg_conceded(match['AwayTeam'], historical_data, 'away'),
                'home_team_form': self.get_recent_form(match['HomeTeam'], historical_data),
                'away_team_form': self.get_recent_form(match['AwayTeam'], historical_data),
                'head_to_head': self.get_head_to_head(match['HomeTeam'], match['AwayTeam'], historical_data),
                'home_team_avg_shots': self.get_team_avg_shots(match['HomeTeam'], historical_data, 'home'),
                'away_team_avg_shots': self.get_team_avg_shots(match['AwayTeam'], historical_data, 'away'),
            }
            
            # Add match statistics if available (for training, use actual values)
            if 'HS' in df.columns and pd.notna(match.get('HS')):
                feature_dict['home_shots'] = match['HS']
                feature_dict['away_shots'] = match['AS']
            else:
                feature_dict['home_shots'] = feature_dict['home_team_avg_shots']
                feature_dict['away_shots'] = feature_dict['away_team_avg_shots']
                
            if 'HST' in df.columns and pd.notna(match.get('HST')):
                feature_dict['home_shots_on_target'] = match['HST']
                feature_dict['away_shots_on_target'] = match['AST']
            else:
                feature_dict['home_shots_on_target'] = feature_dict['home_team_avg_shots'] * 0.4
                feature_dict['away_shots_on_target'] = feature_dict['away_team_avg_shots'] * 0.4
                
            if 'HC' in df.columns and pd.notna(match.get('HC')):
                feature_dict['home_corners'] = match['HC']
                feature_dict['away_corners'] = match['AC']
            else:
                feature_dict['home_corners'] = 5.0
                feature_dict['away_corners'] = 5.0
                
            features.append(feature_dict)
            
        return pd.DataFrame(features)
    
    def get_team_avg_goals(self, team, df, venue):
        """Calculate average goals for a team"""
        if len(df) == 0:
            return 1.5
        if venue == 'home':
            goals = df[df['HomeTeam'] == team]['FTHG'].mean()
        else:
            goals = df[df['AwayTeam'] == team]['FTAG'].mean()
        return goals if not np.isnan(goals) else 1.5
    
    def get_team_avg_conceded(self, team, df, venue):
        """Calculate average goals conceded for a team"""
        if len(df) == 0:
            return 1.5
        if venue == 'home':
            goals_conceded = df[df['HomeTeam'] == team]['FTAG'].mean()
        else:
            goals_conceded = df[df['AwayTeam'] == team]['FTHG'].mean()
        return goals_conceded if not np.isnan(goals_conceded) else 1.5
    
    def get_team_avg_shots(self, team, df, venue):
        """Calculate average shots for a team"""
        if len(df) == 0 or 'HS' not in df.columns:
            return 12.0
        if venue == 'home':
            shots = df[df['HomeTeam'] == team]['HS'].mean()
        else:
            shots = df[df['AwayTeam'] == team]['AS'].mean()
        return shots if not np.isnan(shots) else 12.0
    
    def get_recent_form(self, team, df, matches=5):
        """Calculate recent form (points per game over last N matches)"""
        if len(df) == 0:
            return 1.0
        
        # Get all matches for the team
        team_matches = df[((df['HomeTeam'] == team) | (df['AwayTeam'] == team))].tail(matches)
        
        if len(team_matches) == 0:
            return 1.0
        
        total_points = 0
        for _, match in team_matches.iterrows():
            if match['HomeTeam'] == team:
                if match['FTR'] == 'H':
                    total_points += 3
                elif match['FTR'] == 'D':
                    total_points += 1
            else:  # Away team
                if match['FTR'] == 'A':
                    total_points += 3
                elif match['FTR'] == 'D':
                    total_points += 1
                
        return total_points / len(team_matches) if len(team_matches) > 0 else 1.0
    
    def get_head_to_head(self, home_team, away_team, df):
        """Calculate head-to-head record"""
        if len(df) == 0 or 'HomeTeam' not in df.columns:
            return 0.5
        
        matches = df[((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
                    ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))]
        
        if len(matches) == 0:
            return 0.5
            
        home_wins = len(matches[((matches['HomeTeam'] == home_team) & (matches['FTR'] == 'H')) |
                               ((matches['AwayTeam'] == home_team) & (matches['FTR'] == 'A'))])
        
        return home_wins / len(matches)
    
    def prepare_features(self, df):
        """Prepare features for training"""
        # Create team encoding before calculating features
        all_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
        self.team_encoding = {team: idx for idx, team in enumerate(sorted(all_teams))}
        
        # Work on a copy for date parsing
        df_copy = df.copy()
        df_copy['Date'] = pd.to_datetime(df_copy['Date'], format='%d/%m/%y', errors='coerce')
        df_sorted = df_copy.sort_values('Date').reset_index(drop=True)
        
        feature_df = self.calculate_features(df_sorted)
        
        # Add team encodings using the sorted dataframe
        feature_df['home_team_encoded'] = df_sorted['HomeTeam'].map(self.team_encoding)
        feature_df['away_team_encoded'] = df_sorted['AwayTeam'].map(self.team_encoding)
        
        return feature_df
    
    def train(self, data_folder='data', compare_models=True):
        """Train the prediction model with multiple model comparison"""
        print("Loading data...")
        df = self.load_data(data_folder)
        print(f"Loaded {len(df)} matches from historical data")
        
        # Filter out matches with missing FTR (result)
        df = df[df['FTR'].isin(['H', 'D', 'A'])].copy()
        print(f"After filtering invalid results: {len(df)} matches")
        
        # Sort by date for proper feature calculation
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y', errors='coerce')
        df = df.sort_values('Date').reset_index(drop=True)
        df = df[df['Date'].notna()].reset_index(drop=True)  # Remove invalid dates
        
        print("Calculating features...")
        features = self.prepare_features(df)
        
        # Remove any NaN values
        features = features.fillna(features.mean())
        
        # Target variable (0=Home win, 1=Draw, 2=Away win) - aligned with sorted df
        target = df['FTR'].map({'H': 0, 'D': 1, 'A': 2})
        
        # Remove any rows with NaN targets
        valid_indices = ~target.isna()
        features = features[valid_indices].reset_index(drop=True)
        target = target[valid_indices].reset_index(drop=True)
        
        print(f"Training on {len(features)} samples with {len(features.columns)} features")
        print(f"Feature columns: {list(features.columns)}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42, stratify=target
        )
        
        print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
        
        if compare_models:
            # Compare multiple models
            best_model, model_results = self.compare_models(X_train, X_test, y_train, y_test, features.columns)
            self.model = best_model
        else:
            # Use Random Forest as default
            self.model = RandomForestClassifier(
                n_estimators=200, 
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            print("Training Random Forest model...")
            self.model.fit(X_train, y_train)
            model_results = None
        
        # Calculate and print metrics
        train_accuracy = self.model.score(X_train, y_train)
        test_accuracy = self.model.score(X_test, y_test)
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        # Calculate F1 scores
        train_f1 = f1_score(y_train, train_pred, average='weighted')
        test_f1 = f1_score(y_test, test_pred, average='weighted')
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': features.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        else:
            feature_importance = None
        
        print(f"\n{'='*60}")
        print(f"FINAL MODEL RESULTS:")
        print(f"{'='*60}")
        print(f"Training accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"Training F1-score: {train_f1:.4f}")
        print(f"Test F1-score: {test_f1:.4f}")
        
        if feature_importance is not None:
            print(f"\nTop 10 Most Important Features:")
            print(feature_importance.head(10).to_string(index=False))
        
        # Generate and save visualizations
        print("\nGenerating visualizations...")
        self.create_visualizations(X_train, X_test, y_train, y_test, 
                                train_pred, test_pred, feature_importance, 
                                model_results, test_accuracy)
        
        # Save model
        model_name = type(self.model).__name__
        with open('premier_league_model.pkl', 'wb') as f:
            pickle.dump({
                'model': self.model, 
                'encoding': self.team_encoding,
                'model_name': model_name,
                'test_accuracy': test_accuracy,
                'test_f1': test_f1
            }, f)
        
        print(f"\nModel ({model_name}) saved to premier_league_model.pkl")
        print(f"Team encoding saved for {len(self.team_encoding)} teams")
        print(f"Plots saved in 'plots/' directory")
    
    def compare_models(self, X_train, X_test, y_train, y_test, feature_names):
        """Compare multiple machine learning models"""
        print("\n" + "="*60)
        print("COMPARING MULTIPLE MODELS")
        print("="*60)
        
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=20, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, max_depth=10, learning_rate=0.1,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000, random_state=42, n_jobs=-1
            ),
            'SVM (RBF)': SVC(
                kernel='rbf', probability=True, random_state=42
            ),
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = XGBClassifier(
                n_estimators=100, max_depth=10, learning_rate=0.1,
                random_state=42, n_jobs=-1
            )
        
        results = []
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            try:
                model.fit(X_train, y_train)
                
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                train_acc = accuracy_score(y_train, train_pred)
                test_acc = accuracy_score(y_test, test_pred)
                train_f1 = f1_score(y_train, train_pred, average='weighted')
                test_f1 = f1_score(y_test, test_pred, average='weighted')
                
                results.append({
                    'Model': name,
                    'Train Accuracy': train_acc,
                    'Test Accuracy': test_acc,
                    'Train F1': train_f1,
                    'Test F1': test_f1,
                    'Model Object': model
                })
                
                print(f"  Train Accuracy: {train_acc:.4f} | Test Accuracy: {test_acc:.4f}")
                print(f"  Train F1: {train_f1:.4f} | Test F1: {test_f1:.4f}")
                
            except Exception as e:
                print(f"  Error training {name}: {e}")
                continue
        
        # Create comparison DataFrame
        results_df = pd.DataFrame(results)
        
        if len(results_df) == 0:
            print("No models trained successfully. Using Random Forest as default.")
            return models['Random Forest'], None
        
        # Find best model based on test accuracy
        best_idx = results_df['Test Accuracy'].idxmax()
        best_model = results[best_idx]['Model Object']
        best_name = results[best_idx]['Model']
        
        print(f"\n{'='*60}")
        print(f"BEST MODEL: {best_name}")
        print(f"Test Accuracy: {results[best_idx]['Test Accuracy']:.4f}")
        print(f"Test F1-Score: {results[best_idx]['Test F1']:.4f}")
        print(f"{'='*60}")
        
        return best_model, results_df
    
    def create_visualizations(self, X_train, X_test, y_train, y_test, 
                             train_pred, test_pred, feature_importance,
                             model_results, test_accuracy):
        """Create and save visualization plots"""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # 1. Model Comparison Bar Chart
        if model_results is not None:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # Accuracy comparison
            ax1 = axes[0]
            x = range(len(model_results))
            width = 0.35
            ax1.bar([i - width/2 for i in x], model_results['Train Accuracy'], 
                   width, label='Train Accuracy', alpha=0.8)
            ax1.bar([i + width/2 for i in x], model_results['Test Accuracy'], 
                   width, label='Test Accuracy', alpha=0.8)
            ax1.set_xlabel('Models')
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Model Comparison: Accuracy')
            ax1.set_xticks(x)
            ax1.set_xticklabels(model_results['Model'], rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # F1-Score comparison
            ax2 = axes[1]
            ax2.bar([i - width/2 for i in x], model_results['Train F1'], 
                   width, label='Train F1', alpha=0.8)
            ax2.bar([i + width/2 for i in x], model_results['Test F1'], 
                   width, label='Test F1', alpha=0.8)
            ax2.set_xlabel('Models')
            ax2.set_ylabel('F1-Score')
            ax2.set_title('Model Comparison: F1-Score')
            ax2.set_xticks(x)
            ax2.set_xticklabels(model_results['Model'], rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('plots/01_model_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Confusion Matrix
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        cm_train = confusion_matrix(y_train, train_pred)
        cm_test = confusion_matrix(y_test, test_pred)
        
        sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Home Win', 'Draw', 'Away Win'],
                   yticklabels=['Home Win', 'Draw', 'Away Win'],
                   ax=axes[0])
        axes[0].set_title('Training Set Confusion Matrix')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')
        
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Home Win', 'Draw', 'Away Win'],
                   yticklabels=['Home Win', 'Draw', 'Away Win'],
                   ax=axes[1])
        axes[1].set_title(f'Test Set Confusion Matrix (Accuracy: {test_accuracy:.2%})')
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('plots/02_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Feature Importance
        if feature_importance is not None:
            plt.figure(figsize=(12, 8))
            top_features = feature_importance.head(15)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importance')
            plt.title('Top 15 Feature Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('plots/03_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Prediction Distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        labels = ['Home Win', 'Draw', 'Away Win']
        
        # Count predictions for each class
        train_counts = [np.sum(train_pred == i) for i in range(3)]
        test_counts = [np.sum(test_pred == i) for i in range(3)]
        
        axes[0].bar(labels, train_counts, alpha=0.7)
        axes[0].set_title('Training Set: Prediction Distribution')
        axes[0].set_ylabel('Count')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        axes[1].bar(labels, test_counts, alpha=0.7)
        axes[1].set_title('Test Set: Prediction Distribution')
        axes[1].set_ylabel('Count')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('plots/04_prediction_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Accuracy by Class
        class_names = ['Home Win', 'Draw', 'Away Win']
        train_class_acc = []
        test_class_acc = []
        
        for i in range(3):
            train_mask = y_train == i
            test_mask = y_test == i
            if train_mask.sum() > 0:
                train_class_acc.append(accuracy_score(y_train[train_mask], train_pred[train_mask]))
            else:
                train_class_acc.append(0)
            if test_mask.sum() > 0:
                test_class_acc.append(accuracy_score(y_test[test_mask], test_pred[test_mask]))
            else:
                test_class_acc.append(0)
        
        plt.figure(figsize=(10, 6))
        x = np.arange(len(class_names))
        width = 0.35
        plt.bar(x - width/2, train_class_acc, width, label='Train', alpha=0.8)
        plt.bar(x + width/2, test_class_acc, width, label='Test', alpha=0.8)
        plt.xlabel('Class')
        plt.ylabel('Accuracy')
        plt.title('Accuracy by Class')
        plt.xticks(x, class_names)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('plots/05_accuracy_by_class.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("All visualizations saved to 'plots/' directory")
    
    def calculate_features_for_prediction(self, home_team, away_team, df=None):
        """Calculate features for a single match prediction"""
        if df is None:
            df = self.historical_data if self.historical_data is not None else self.load_data()
        
        # Parse dates if not already done
        if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df = df.copy()
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y', errors='coerce')
            df = df.sort_values('Date').reset_index(drop=True)
        
        feature_dict = {
            'home_team_avg_goals': self.get_team_avg_goals(home_team, df, 'home'),
            'away_team_avg_goals': self.get_team_avg_goals(away_team, df, 'away'),
            'home_team_avg_conceded': self.get_team_avg_conceded(home_team, df, 'home'),
            'away_team_avg_conceded': self.get_team_avg_conceded(away_team, df, 'away'),
            'home_team_form': self.get_recent_form(home_team, df),
            'away_team_form': self.get_recent_form(away_team, df),
            'head_to_head': self.get_head_to_head(home_team, away_team, df),
            'home_team_avg_shots': self.get_team_avg_shots(home_team, df, 'home'),
            'away_team_avg_shots': self.get_team_avg_shots(away_team, df, 'away'),
            'home_shots': self.get_team_avg_shots(home_team, df, 'home'),  # Use average for prediction
            'away_shots': self.get_team_avg_shots(away_team, df, 'away'),
            'home_shots_on_target': self.get_team_avg_shots(home_team, df, 'home') * 0.4,
            'away_shots_on_target': self.get_team_avg_shots(away_team, df, 'away') * 0.4,
            'home_corners': 5.0,  # Default value
            'away_corners': 5.0,  # Default value
            'home_team_encoded': self.team_encoding.get(home_team, 0),
            'away_team_encoded': self.team_encoding.get(away_team, 0)
        }
        
        # Return in the same order as training features
        feature_order = [
            'home_team_avg_goals', 'away_team_avg_goals', 'home_team_avg_conceded', 
            'away_team_avg_conceded', 'home_team_form', 'away_team_form', 
            'head_to_head', 'home_team_avg_shots', 'away_team_avg_shots',
            'home_shots', 'away_shots', 'home_shots_on_target', 'away_shots_on_target',
            'home_corners', 'away_corners', 'home_team_encoded', 'away_team_encoded'
        ]
        
        return [feature_dict[key] for key in feature_order]
    
    def load_model(self, model_path='premier_league_model.pkl'):
        """Load a saved model"""
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
            self.model = saved_data['model']
            self.team_encoding = saved_data['encoding']
    
    def predict_match(self, home_team, away_team, current_data=None):
        """Predict a single match"""
        if self.model is None:
            self.load_model()
        
        if self.historical_data is None:
            self.load_data()
            
        # Create features for prediction
        features = self.calculate_features_for_prediction(home_team, away_team, current_data)
        
        prediction = self.model.predict_proba([features])[0]
        result_map = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'predictions': {
                'home_win': float(prediction[0]),
                'draw': float(prediction[1]),
                'away_win': float(prediction[2])
            },
            'predicted_result': result_map[np.argmax(prediction)]
        }

# Usage
if __name__ == '__main__':
    predictor = PremierLeaguePredictor()
    predictor.train('data')
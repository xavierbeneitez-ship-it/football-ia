import requests
import numpy as np
from scipy.stats import poisson

class FootballEngine:
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {'x-apisports-key': api_key}
        self.base_url = "https://v3.football.api-sports.io/"
        # Coefficients d'auto-apprentissage (ajustés dynamiquement)
        self.weights = {"attack": 1.1, "defense": 0.9, "form": 1.2}

    def get_team_stats(self, league_id, season, team_id):
        params = {"league": league_id, "season": season, "team": team_id}
        res = requests.get(f"{self.base_url}teams/statistics", headers=self.headers, params=params).json()
        return res['response']

    def predict(self, home_stats, away_stats):
        # Calcul des potentiels d'attaque et défense
        home_attack = (home_stats['goals']['for']['average']['home']) * self.weights['attack']
        away_defense = (away_stats['goals']['against']['average']['away']) * self.weights['defense']
        
        home_lambda = home_attack * away_defense
        away_lambda = (away_stats['goals']['for']['average']['away']) * (home_stats['goals']['against']['average']['home'])

        # Matrice de probabilités (Score exact 0-0 à 5-5)
        matrix = np.outer([poisson.pmf(i, home_lambda) for i in range(6)], 
                          [poisson.pmf(i, away_lambda) for i in range(6)])

        home_win = np.sum(np.tril(matrix, -1))
        draw = np.sum(np.diag(matrix))
        away_win = np.sum(np.triu(matrix, 1))
        
        score_probable = np.unravel_index(np.argmax(matrix), matrix.shape)
        over_25 = 1 - (matrix[0,0] + matrix[0,1] + matrix[1,0] + matrix[0,1] + matrix[1,1])

        return {
            "win_home": home_win, "draw": draw, "win_away": away_win,
            "score": score_probable, "over_25": over_25
        }

    def auto_learn(self, predicted_score, actual_score):
        # Si le modèle a sous-estimé les buts, on augmente le poids de l'attaque
        error = np.mean(np.array(actual_score) - np.array(predicted_score))
        self.weights['attack'] += error * 0.01
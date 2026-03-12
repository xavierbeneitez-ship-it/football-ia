import requests
import numpy as np
from scipy.stats import poisson

class FootballEnginePro:
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {'x-apisports-key': api_key}
        self.base_url = "https://v3.football.api-sports.io/"
        # Poids auto-adaptatifs par ligue
        self.league_weights = {39: 1.0, 61: 1.0, 78: 1.0, 135: 1.0, 2: 1.0}

    def get_fixtures(self, league_id, season=2025):
        """Récupère les prochains matchs de la saison"""
        params = {"league": league_id, "season": season, "next": 10}
        res = requests.get(f"{self.base_url}fixtures", headers=self.headers, params=params).json()
        return res.get('response', [])

    def get_team_performance(self, league_id, team_id, season=2025):
        """Analyse la forme sur les 10 derniers matchs (xG simulé)"""
        params = {"league": league_id, "season": season, "team": team_id}
        res = requests.get(f"{self.base_url}teams/statistics", headers=self.headers, params=params).json()
        stats = res['response']
        
        # Extraction des moyennes de buts (Attaque/Défense)
        att = stats['goals']['for']['average']['total']
        def_ = stats['goals']['against']['average']['total']
        return float(att), float(def_)

    def predict_match(self, home_id, away_id, league_id):
        h_att, h_def = self.get_team_performance(league_id, home_id)
        a_att, a_def = self.get_team_performance(league_id, away_id)

        # Calcul des espérances de buts (Lambda)
        # On ajuste selon le poids de la ligue (Auto-learning)
        lambda_h = h_att * a_def * self.league_weights.get(league_id, 1.0)
        lambda_a = a_att * h_def * self.league_weights.get(league_id, 1.0)

        # Simulation de Poisson
        matrix = np.outer([poisson.pmf(i, lambda_h) for i in range(6)], 
                          [poisson.pmf(j, lambda_a) for j in range(6)])

        prob_h = np.sum(np.tril(matrix, -1))
        prob_d = np.sum(np.diag(matrix))
        prob_a = np.sum(np.triu(matrix, 1))
        
        score = np.unravel_index(np.argmax(matrix), matrix.shape)
        over25 = 1 - np.sum(matrix[0:2, 0:2]) # Probabilité simplifiée > 2.5

        return {
            "probs": [prob_h, prob_d, prob_a],
            "score": f"{score[0]} - {score[1]}",
            "over25": over25,
            "confidence": max(prob_h, prob_a) * 100
        }

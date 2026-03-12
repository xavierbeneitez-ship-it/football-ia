import requests
import numpy as np
from scipy.stats import poisson
from datetime import datetime, timedelta

class AdvancedFootballEngine:
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {'x-apisports-key': api_key}
        self.base_url = "https://v3.football.api-sports.io/"
        
    def rho_correction(self, x, y, lambda_x, lambda_y, rho):
        """Correction de Dixon-Coles pour les faibles scores (0-0, 1-0, 0-1, 1-1)"""
        if x == 0 and y == 0: return 1 - (lambda_x * lambda_y * rho)
        elif x == 0 and y == 1: return 1 + (lambda_x * rho)
        elif x == 1 and y == 0: return 1 + (lambda_y * rho)
        elif x == 1 and y == 1: return 1 - rho
        return 1

    def get_fixtures_by_date(self, league_id, date_obj):
        """Récupère les matchs pour une date précise"""
        date_str = date_obj.strftime('%Y-%m-%d')
        params = {"league": league_id, "season": 2025, "date": date_str}
        res = requests.get(f"{self.base_url}fixtures", headers=self.headers, params=params).json()
        return res.get('response', [])

    def calculate_advanced_prediction(self, home_stats, away_stats):
        # Paramètres d'attaque et défense avec avantage domicile (HFA)
        hfa = 1.25 # Home Field Advantage coefficient
        
        # Lambda calculé sur les moyennes pondérées
        lambda_h = home_stats['att'] * away_stats['def'] * hfa
        lambda_a = away_stats['att'] * home_stats['def']
        
        # Correction de corrélation (Dixon-Coles rho)
        rho = -0.11 # Valeur empirique moyenne pour les ligues européennes
        
        matrix = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                prob = poisson.pmf(i, lambda_h) * poisson.pmf(j, lambda_a)
                matrix[i, j] = prob * self.rho_correction(i, j, lambda_h, lambda_a, rho)
        
        return matrix

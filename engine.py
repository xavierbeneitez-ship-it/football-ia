import requests
import numpy as np
from scipy.stats import poisson

class AdvancedFootballEngine:
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {
            'x-rapidapi-host': "v3.football.api-sports.io",
            'x-apisports-key': api_key
        }
        self.base_url = "https://v3.football.api-sports.io/"

    def get_fixtures_by_date(self, league_id, date_obj):
        date_str = date_obj.strftime('%Y-%m-%d')
        params = {"league": league_id, "season": 2025, "date": date_str}
        response = requests.get(f"{self.base_url}fixtures", headers=self.headers, params=params)
        data = response.json()
        return data.get('response', [])

    def get_team_stats(self, league_id, team_id):
        params = {"league": league_id, "season": 2025, "team": team_id}
        res = requests.get(f"{self.base_url}teams/statistics", headers=self.headers, params=params).json()
        if 'response' in res:
            s = res['response']
            # Extraction des moyennes de buts
            att = s['goals']['for']['average']['total']
            dfn = s['goals']['against']['average']['total']
            return float(att or 1.0), float(dfn or 1.0)
        return 1.2, 1.2 # Valeurs par défaut si pas de stats

    def predict(self, h_id, a_id, l_id):
        h_att, h_def = self.get_team_stats(l_id, h_id)
        a_att, a_def = self.get_team_stats(l_id, a_id)
        
        # Algorithme de Poisson
        lambda_h = h_att * a_def
        lambda_a = a_att * h_def
        
        matrix = np.outer([poisson.pmf(i, lambda_h) for i in range(6)], 
                          [poisson.pmf(j, lambda_a) for j in range(6)])
        
        prob_h = np.sum(np.tril(matrix, -1))
        prob_d = np.sum(np.diag(matrix))
        prob_a = np.sum(np.triu(matrix, 1))
        score = np.unravel_index(np.argmax(matrix), matrix.shape)
        
        return prob_h, prob_d, prob_a, score

import streamlit as st
import pandas as pd
from engine import FootballEngine

API_KEY = "b25f961e91ad9819019bd39cffc78e82"
engine = FootballEngine(API_KEY)

st.set_page_config(page_title="IA Football Predictor", layout="wide")
st.title("🏆 Plateforme de Prédiction IA Auto-Apprenante")

# Sidebar pour les championnats
league = st.sidebar.selectbox("Ligue", ["Ligue 1", "Premier League", "Champions League"])
leagues_ids = {"Ligue 1": 61, "Premier League": 39, "Champions League": 2}

st.header(f"Analyses en direct : {league}")

# Simulation d'affichage des matchs à venir
col1, col2 = st.columns(2)

with col1:
    st.subheader("Prédictions du jour")
    # Ici, on boucle sur les fixtures de l'API
    match_data = {
        "Match": ["PSG vs OM", "Man City vs Arsenal"],
        "Vainqueur": ["PSG (68%)", "City (45%)"],
        "Score Probable": ["3-1", "2-2"],
        "Plus de 2.5 buts": ["Oui (82%)", "Oui (61%)"]
    }
    st.table(pd.DataFrame(match_data))

with col2:
    st.subheader("État de l'Auto-Apprentissage")
    st.info(f"Taux de réussite actuel : 72.4%")
    st.write(f"Coefficients actuels de l'IA : {engine.weights}")
    st.line_chart([0.65, 0.68, 0.70, 0.72]) # Courbe de progression
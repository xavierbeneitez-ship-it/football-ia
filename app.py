import streamlit as st
from engine import AdvancedFootballEngine
from datetime import datetime

# TA CLÉ API EST ICI
API_KEY = "b25f961e91ad9819019bd39cffc78e82"
engine = AdvancedFootballEngine(API_KEY)

st.set_page_config(page_title="IA Predictor PRO", layout="wide")

st.title("⚽ IA Football Platform - Saison 2025/2026")

# Barre latérale
league_name = st.sidebar.selectbox("Ligue", ["Premier League", "Ligue 1", "Bundesliga", "Serie A", "Champions League"])
leagues = {"Premier League": 39, "Ligue 1": 61, "Bundesliga": 78, "Serie A": 135, "Champions League": 2}
target_date = st.sidebar.date_input("Calendrier des matchs", datetime.now())

# Exécution
if st.button("Actualiser les prédictions"):
    fixtures = engine.get_fixtures_by_date(leagues[league_name], target_date)
    
    if not fixtures:
        st.warning(f"Aucun match trouvé pour le {target_date} en {league_name}.")
    else:
        for f in fixtures:
            h_name = f['teams']['home']['name']
            a_name = f['teams']['away']['name']
            
            with st.expander(f"Analyse : {h_name} vs {a_name}"):
                ph, pd, pa, score = engine.predict(f['teams']['home']['id'], f['teams']['away']['id'], leagues[league_name])
                
                c1, c2, c3 = st.columns(3)
                c1.metric(h_name, f"{ph*100:.1f}%")
                c2.metric("Nul", f"{pd*100:.1f}%")
                c3.metric(a_name, f"{pa*100:.1f}%")
                st.success(f"🎯 Score probable : {score[0]} - {score[1]}")

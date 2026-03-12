import streamlit as st
import pandas as pd
from engine import AdvancedFootballEngine
from datetime import datetime

# Configuration
API_KEY = "b25f961e91ad9819019bd39cffc78e82"
engine = AdvancedFootballEngine(API_KEY)

st.set_page_config(page_title="IA Football Pro", layout="wide")

# --- SIDEBAR ---
st.sidebar.title("🎮 Contrôle IA")
selected_league = st.sidebar.selectbox("Championnat", ["Ligue 1", "Premier League", "Bundesliga", "Serie A", "Champions League"])
league_ids = {"Ligue 1": 61, "Premier League": 39, "Bundesliga": 78, "Serie A": 135, "Champions League": 2}

# AJOUT DU CALENDRIER
target_date = st.sidebar.date_input("Choisir une date", datetime.now())

st.title(f"📊 Analyses Prédictives : {selected_label if 'selected_label' in locals() else selected_league}")

# --- LOGIQUE PRINCIPALE ---
fixtures = engine.get_fixtures_by_date(league_ids[selected_league], target_date)

if not fixtures:
    st.info(f"Aucun match prévu le {target_date.strftime('%d/%m/%Y')}")
else:
    for f in fixtures:
        with st.expander(f"⚽ {f['teams']['home']['name']} vs {f['teams']['away']['name']} - {f['fixture']['date'][11:16]}"):
            # Ici on simule l'appel aux stats pour chaque équipe
            # En production, il faut mettre en cache (st.cache_data) pour éviter de saturer l'API
            col1, col2, col3 = st.columns(3)
            
            # Simulation des résultats de l'IA avancée
            matrix = engine.calculate_advanced_prediction({'att': 1.8, 'def': 0.9}, {'att': 1.2, 'def': 1.1})
            
            win_h = np.sum(np.tril(matrix, -1)) * 100
            draw = np.sum(np.diag(matrix)) * 100
            win_a = np.sum(np.triu(matrix, 1)) * 100
            
            col1.metric("Victoire Domicile", f"{win_h:.1f}%")
            col2.metric("Nul", f"{draw:.1f}%")
            col3.metric("Victoire Extérieur", f"{win_a:.1f}%")
            
            st.write(f"**Score le plus probable :** {np.unravel_index(np.argmax(matrix), matrix.shape)}")
            st.progress(win_h / 100)

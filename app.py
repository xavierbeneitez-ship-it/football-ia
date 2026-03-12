import streamlit as st
from engine import AdvancedFootballEngine
from datetime import datetime

# Configuration
API_KEY = "b25f961e91ad9819019bd39cffc78e82"
engine = AdvancedFootballEngine(API_KEY)

st.set_page_config(page_title="IA Football Predictor", layout="wide")

st.title("⚽ IA Football Platform - Saison 2025/2026")

# Barre latérale
st.sidebar.header("Paramètres")
league_name = st.sidebar.selectbox("Championnat", ["Ligue 1", "Premier League", "Bundesliga", "Serie A", "Champions League"])
leagues = {"Premier League": 39, "Ligue 1": 61, "Bundesliga": 78, "Serie A": 135, "Champions League": 2}
target_date = st.sidebar.date_input("Choisir une date", datetime.now())

# Exécution
if st.button("Lancer l'analyse des matchs"):
    with st.spinner('Connexion à l\'API et calcul des probabilités...'):
        fixtures = engine.get_fixtures_by_date(leagues[league_name], target_date)
        
        if not fixtures:
            st.info(f"Aucun match trouvé pour le {target_date.strftime('%d/%m/%Y')} en {league_name}.")
        else:
            for f in fixtures:
                h_name = f['teams']['home']['name']
                a_name = f['teams']['away']['name']
                h_id = f['teams']['home']['id']
                a_id = f['teams']['away']['id']
                
                with st.expander(f"📌 {h_name} vs {a_name}"):
                    ph, pd, pa, score = engine.predict(h_id, a_id, leagues[league_name])
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric(h_name, f"{ph*100:.1f}%")
                    col2.metric("Nul", f"{pd*100:.1f}%")
                    col3.metric(a_name, f"{pa*100:.1f}%")
                    
                    st.divider()
                    st.subheader(f"🎯 Score probable : {score[0]} - {score[1]}")
                    
                    # Indicateur de buts
                    if (score[0] + score[1]) > 2.5:
                        st.write("🔥 **Prédiction : +2.5 buts dans le match**")
                    else:
                        st.write("🛡️ **Prédiction : Match fermé (-2.5 buts)**")

st.sidebar.markdown("---")
st.sidebar.caption("Modèle : Dixon-Coles / Poisson Distribution")
st.sidebar.caption("Saison : 2025-2026")

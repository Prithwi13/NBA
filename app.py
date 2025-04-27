# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load saved files
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
df_final_scaled = pd.read_csv('team_stats.csv')

# Features used for prediction
features_to_use = ['W_PCT', 'PTS', 'REB', 'AST', 'FG_PCT', 'TOV', 
                   'STL', 'BLK', 'PLUS_MINUS', 'AssistRatio', 'WinStreak']

# Title
st.title("🏀 Team vs Team Match Predictor")

# Team selection
teams = df_final_scaled['TEAM_NAME'].unique()
team1 = st.selectbox("Select Team 1", teams)
team2 = st.selectbox("Select Team 2", teams)

# Predict button
if st.button("Predict Winner"):
    # Fetch teams
    team1_row = df_final_scaled[df_final_scaled['TEAM_NAME'] == team1]
    team2_row = df_final_scaled[df_final_scaled['TEAM_NAME'] == team2]

    team1_input = team1_row[features_to_use].values[0]
    team2_input = team2_row[features_to_use].values[0]

    team1_scaled = scaler.transform([team1_input])[0]
    team2_scaled = scaler.transform([team2_input])[0]

    diff_features = pd.DataFrame([team1_scaled - team2_scaled], columns=features_to_use)
    
    # Prediction
    proba = model.predict_proba(diff_features)[0]
    pred = model.predict(diff_features)[0]

    winner = team1 if pred == 1 else team2
    win_percent = round(max(proba) * 100, 2)

    # Display result
    st.success(f"🏆 Predicted Winner: **{winner}** ({win_percent}%)")

    # 🕸️ Radar Chart
    angles = np.linspace(0, 2 * np.pi, len(features_to_use), endpoint=False).tolist()
    angles += angles[:1]

    t1 = list(team1_scaled) + [team1_scaled[0]]
    t2 = list(team2_scaled) + [team2_scaled[0]]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, t1, label=team1, linewidth=2)
    ax.fill(angles, t1, alpha=0.25)

    ax.plot(angles, t2, label=team2, linewidth=2)
    ax.fill(angles, t2, alpha=0.25)

    ax.set_title(f"Stats Comparison", size=16, weight='bold')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features_to_use, fontsize=8)
    ax.set_yticklabels([])
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    st.pyplot(fig)

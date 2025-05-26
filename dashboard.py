import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from utils.cleaner import creer_master, encoder_categorielles, engine
from models.train_pipeline import detect_anomalies_rules
import matplotlib.pyplot as plt

# ========== FONCTION PRINCIPALE ==========

@st.cache_data
def load_and_predict():
    # 1. Chargement des données brutes
    df_master = creer_master(engine)
    df_encoded = encoder_categorielles(df_master)

    # 2. Détection par règles métiers
    df = detect_anomalies_rules(df_encoded)
    df['anomaly_label'] = (df['total_flags'] > 0).astype(int)

    # 3. Prédiction ML
    drop_cols = ['IDArticle', 'IDCodeBarre', 'NumInterne', 'NumInterne_CB', 'total_flags', 'anomaly_label']
    X = df.select_dtypes(include=[np.number]).drop(columns=[col for col in drop_cols if col in df.columns])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

    pipeline = joblib.load("anomaly_xgb_pipeline.joblib")
    df['prediction'] = pipeline.predict(X)

    # 4. Réintégration des colonnes métiers importantes
    cols_to_add = [
        'Code', 'Famille', 'Fournisseur', 'Saison',
        'PrixFac', 'PrixAchat', 'Prix', 'CodeBarre', 'marge_rel', 'diff_prix'
    ]
    for col in cols_to_add:
        if col in df_master.columns:
            df[col] = df['IDArticle'].map(
                df_master.drop_duplicates(subset='IDArticle').set_index('IDArticle')[col]
            )
    return df

# ========== AFFICHAGE TABLEAU ==========
def show_table(df, title):
    st.subheader(f"📋 {title}")
    st.write(f"Nombre d'anomalies détectées : {df.shape[0]}")
    columns_to_hide = [col for col in df.columns if col.startswith("sai_") or col.startswith("fou_")]
    display_df = df.drop(columns=columns_to_hide)
    st.dataframe(display_df.head(100), use_container_width=True)

# ========== AFFICHAGE GRAPHIQUE ==========
def show_barplot(df, group_col, title):
    if df.empty or group_col not in df.columns:
        st.info("Pas de données disponibles pour afficher ce graphique.")
        return

    count_series = df[group_col].dropna().value_counts()
    if count_series.empty:
        st.info("Aucune valeur à afficher dans le graphique.")
        return
    count_series.index = [str(x)[:12] + '…' if len(str(x)) > 12 else str(x) for x in count_series.index]

    st.subheader(f"📊 Répartition des anomalies par {group_col}")
    fig, ax = plt.subplots()
    count_series.plot(kind='bar', ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Nombre d'anomalies")
    ax.set_xlabel(group_col)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    st.pyplot(fig)

# HISTOGRAMME : répartition du nombre total de flags

def show_histogram_flags(df):
    if 'total_flags' in df.columns and not df.empty:
        st.subheader("📉 Distribution du nombre total de flags")
        fig, ax = plt.subplots()
        df['total_flags'].plot.hist(bins=range(0, df['total_flags'].max()+2), ax=ax)
        ax.set_xlabel("Nombre de flags")
        ax.set_ylabel("Nombre d'articles")
        ax.set_title("Répartition des enregistrements selon le nombre d’anomalies détectées")
        st.pyplot(fig)

# CAMEMBERT : répartition des types d’anomalies activées

def show_pie_flags(df, flags):
    st.subheader("🥧 Répartition des types d’anomalies (flags activés)")
    
    # Ne garder que les flags avec au moins 1 anomalie
    flag_sums = df[flags].sum()
    flag_sums = flag_sums[flag_sums > 0].sort_values(ascending=False)

    # Renommer les flags avec des libellés clairs
    flag_labels = [flag_label_mapping.get(col, col) for col in flag_sums.index]
    flag_sums.index = flag_labels

    if flag_sums.sum() == 0:
        st.info("Aucune anomalie présente dans les flags sélectionnés.")
        return
    fig, ax = plt.subplots()
    flag_sums.plot.pie(autopct='%1.1f%%', ax=ax, ylabel="")
    ax.set_title("Répartition des anomalies détectées")
    st.pyplot(fig)


# ========== INTERFACE STREAMLIT ==========

st.set_page_config(page_title="Dashboard Anomalies", layout="wide")
st.title("🚨 Tableau de Bord - Détection d'Anomalies")

with st.spinner("Chargement et détection en cours..."):
    df = load_and_predict()

# ========== SIDEBAR ==========
filtered_df = df[df['prediction'] == 1].copy()
with st.sidebar:
    st.header("🔎 Filtres")
    fournisseurs = sorted(df['Fournisseur'].dropna().unique())
    saisons = sorted(df['Saison'].dropna().unique())
    familles = sorted(df['Famille'].dropna().unique())

    selected_fournisseur = st.multiselect("Fournisseur", fournisseurs)
    selected_saison = st.multiselect("Saison", saisons)
    selected_famille = st.multiselect("Famille", familles)
    

    # Filtrage par types d’anomalies (flag_)
    flag_columns = [col for col in df.columns if col.startswith("flag_")]
    flag_label_mapping = {
        'flag_prix_zero': "Prix d'achat nul",
        'flag_marge_exces': "Marge excessive",
        'flag_diff_neg': "Différence de prix négative",
        'flag_no_numart': "Numéro article manquant",
        'flag_no_numcb': "Numéro CodeBarre manquant",
        'flag_no_cb': "CodeBarre vide",
        'flag_code_invalid': "Code article invalide"
    }
    flag_labels = [flag_label_mapping.get(col, col) for col in flag_columns]
    label_to_flag = {v: k for k, v in flag_label_mapping.items()}
    selected_labels = st.multiselect("Types d'anomalies (règles métier)", options=flag_labels)
    selected_flags = [label_to_flag[label] for label in selected_labels if label in label_to_flag]
    seuil_flags = st.slider("Seuil total_flags (règles métier)", 0, int(df['total_flags'].max()), 1)
    
    st.markdown("---")
    st.subheader("🔍 Filtre personnalisé")
    filtered_columns = [col for col in df.columns if not (col.startswith("fou_") or col.startswith("sai_"))]
    filter_col = st.selectbox("Choisir une colonne", sorted(filtered_columns))

    if filter_col:
        unique_vals = sorted(df[filter_col].dropna().astype(str).unique())
        filter_val = st.text_input("Valeur exacte à rechercher", placeholder=f"Ex: {unique_vals[0] if unique_vals else ''}")
        if filter_val:
            filtered_df = filtered_df[filtered_df[filter_col].astype(str) == filter_val]

# ========== APPLICATION DES FILTRES ==========



if selected_fournisseur:
    filtered_df = filtered_df[filtered_df['Fournisseur'].isin(selected_fournisseur)]
if selected_saison:
    filtered_df = filtered_df[filtered_df['Saison'].isin(selected_saison)]
if selected_famille:
    filtered_df = filtered_df[filtered_df['Famille'].isin(selected_famille)]

filtered_df = filtered_df[filtered_df['total_flags'] >= seuil_flags]

if selected_flags:
    for flag in selected_flags:
        if flag in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[flag] == 1]

# ========== AFFICHAGE ==========

show_table(filtered_df, "Anomalies détectées")

if filtered_df.empty:
    st.warning("⚠️ Aucun enregistrement ne correspond aux filtres sélectionnés.")
else:
    # GRAPHE 1 & 2 côte à côte : Famille + Fournisseur
    col1, col2 = st.columns(2)

    with col1:
        if 'Famille' in filtered_df.columns:
            if not filtered_df['Famille'].dropna().empty:
                show_barplot(filtered_df, 'Famille', "Anomalies par Famille")
            else:
                st.info("✅ Aucun regroupement par 'Famille' possible.")

    with col2:
        if 'Fournisseur' in filtered_df.columns:
            if not filtered_df['Fournisseur'].dropna().empty:
                show_barplot(filtered_df, 'Fournisseur', "Anomalies par Fournisseur")
            else:
                st.info("✅ Aucun regroupement par 'Fournisseur' possible.")


    # GRAPHE 3 & 4 côte à côte : Saison + Histogramme des flags
    col3, col4 = st.columns(2)

    with col3:
        if 'Saison' in filtered_df.columns:
            if not filtered_df['Saison'].dropna().empty:
                show_barplot(filtered_df, 'Saison', "Anomalies par Saison")
            else:
                st.info("✅ Aucun regroupement par 'Saison' possible.")

    with col4:
        if 'total_flags' in filtered_df.columns and not filtered_df.empty:
            show_histogram_flags(filtered_df)
        else:
            st.info("✅ Impossible d'afficher l'histogramme (données manquantes).")


    # GRAPHE 5 : camembert des types d’anomalies
    show_pie_flags(filtered_df, [col for col in filtered_df.columns if col.startswith("flag_")])
# ========== TÉLÉCHARGEMENT ==========
st.download_button(
    label="📥 Télécharger le rapport filtré (CSV)",
    data=filtered_df.to_csv(index=False, encoding='utf-8-sig'),
    file_name=f"anomalies_filtrees_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    mime='text/csv'
)

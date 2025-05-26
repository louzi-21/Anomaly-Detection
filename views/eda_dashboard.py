import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import pandas as pd
from utils.cleaner import (
    engine,
    nettoyer_ar_sfamille,
    nettoyer_arfamille,
    nettoyer_saison,
    nettoyer_article,
    nettoyer_codebarre,
    nettoyer_fournisseur,
    nettoyer_tailles,
    creer_master,
    encoder_categorielles,
)
from ML import detect_anomalies_ml
from models.train_pipeline import detect_anomalies_rules
@st.cache_data
# Charge et prépare les données, crée master, encode, détecte anomalies (règles + ML)
def load_data():
    data = {}
    # Nettoyage des tables sources
    loaders = {
        'ar_sfamille': nettoyer_ar_sfamille,
        'arfamille': nettoyer_arfamille,
        'saison': nettoyer_saison,
        'article': nettoyer_article,
        'codebarre': nettoyer_codebarre,
        'fournisseur': nettoyer_fournisseur,
        'tailles': nettoyer_tailles,
    }
    for name, fn in loaders.items():
        try:
            df = fn(engine)
            if df is not None and not df.empty:
                data[name] = df
            else:
                st.warning(f"Table {name}: aucune donnée disponible.")
        except Exception as e:
            st.error(f"Erreur chargement {name} : {e}")
    # Fusion et encodage des données
    try:
        master_df = creer_master(engine)
        data['master_df'] = master_df
        encoded_df = encoder_categorielles(master_df)
        data['df_encoded'] = encoded_df
    except Exception as e:
        st.error(f"Erreur préparation master/encoded : {e}")
        return data
    # Détection d'anomalies par règles métiers
    try:
        df_rules = detect_anomalies_rules(encoded_df)
        data['df_rules'] = df_rules
    except Exception as e:
        st.error(f"Erreur détection règles métiers : {e}")
    # Détection d'anomalies par ML
    try:
        df_ml = detect_anomalies_ml(df_rules)
        data['df_ml'] = df_ml
    except Exception as e:
        st.error(f"Erreur détection ML : {e}")
    return data

# Génère résumé statistique et valeurs manquantes
def generate_summary(df):
    st.subheader("Résumé statistique")
    st.write(df.describe(include='all'))
    st.subheader("Valeurs manquantes")
    missing = pd.DataFrame(df.isnull().sum(), columns=['missing']).reset_index().rename(columns={'index':'column'})
    st.dataframe(missing)

# Interface Streamlit
def main():
    st.set_page_config(page_title="EDA Dashboard", layout="wide")
    st.title("Dashboard EDA interactif")
    data_dict = load_data()
    if not data_dict:
        st.warning("Aucune table chargée.")
        return
    tabs = st.tabs(list(data_dict.keys()))
    for tab, name in zip(tabs, data_dict.keys()):
        df = data_dict[name]
        with tab:
            st.header(f"Table : {name}")
            st.write(f"Dimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes")
            st.markdown("**Aperçu des données :**")
            st.dataframe(df.head(50))
            st.markdown("---")
            generate_summary(df)
            # Affichage des anomalies détectées
            if name == 'df_rules':
                st.markdown("---")
                st.subheader("Anomalies détectées par règles métiers")
                anomalies = df[df['total_flags'] > 0]
                st.write(f"Total flagged records: {anomalies.shape[0]}")
                st.dataframe(anomalies.head(50))
            if name == 'df_ml':
                st.markdown("---")
                st.subheader("Anomalies détectées par ML (IF ou LOF)")
                anomalies_ml = df[(df['anom_if'] == -1) | (df['anom_lof'] == -1)]
                st.write(f"Total ML anomalies: {anomalies_ml.shape[0]}")
                st.dataframe(anomalies_ml.head(50))

if __name__ == '__main__':
    main()

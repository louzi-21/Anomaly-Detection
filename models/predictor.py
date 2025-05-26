import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
import joblib
from utils.cleaner import encoder_categorielles
from models.train_pipeline import detect_anomalies_rules

# Charger le pipeline ML sauvegardé
pipeline = joblib.load("anomaly_xgb_pipeline.joblib")


def predict_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prend en entrée un DataFrame brut, applique les règles métier et le modèle ML,
    et retourne le DataFrame avec la colonne 'prediction'.
    """
    # Encodage et règles métiers
    df_encoded = encoder_categorielles(df)
    df_rules = detect_anomalies_rules(df_encoded)

    # Préparation des features pour la prédiction
    drop_cols = ['IDArticle', 'IDCodeBarre', 'NumInterne', 'NumInterne_CB', 'total_flags']
    X = df_rules.select_dtypes(include=[np.number]).drop(columns=[col for col in drop_cols if col in df_rules.columns])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

    # Prédiction
    predictions = pipeline.predict(X)
    df_rules['prediction'] = predictions
    return df

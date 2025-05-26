import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib
import re
from utils.cleaner import (
    encoder_categorielles,
    creer_master,
    engine
)

# Charger et encoder les données
df_master = creer_master(engine)
df_encoded = encoder_categorielles(df_master)

# Définir le motif et la fonction d'analyse du code
motif_code = re.compile(r'^([A-Z]{4,5})(\d{1,3})([A-Z]{0,2})$')
def analyser_code(code):
    return 0 if motif_code.match(str(code)) else 1

# Fonction de détection d'anomalies
def detect_anomalies_rules(df):
    df = df.copy()
    df['flag_prix_zero']   = (df['PrixAchat'] <= 0).astype(int)
    df['flag_marge_exces'] = (df['marge_rel'] > 6.0).astype(int)
    df['flag_diff_neg']    = (df['diff_prix'] < 0).astype(int)
    df['flag_no_numart']   = df['NumInterne'].isna().astype(int)
    df['flag_no_numcb']    = df['NumInterne_CB'].isna().astype(int)
    df['flag_no_cb']       = df['CodeBarre'].isna().astype(int)
    df['flag_code_invalid']  = df['Code'].apply(analyser_code)

    df['total_flags']      = df[[c for c in df if c.startswith('flag_')]].sum(axis=1)

    return df

# Appliquer les règles de détection
df = detect_anomalies_rules(df_encoded)

# Création du label (anomalies = 1, pas d'anomalies = 0)
y = (df['total_flags'] > 0).astype(int)

# Sélection des features (caractéristiques numériques)
drop_cols = ['IDArticle', 'IDCodeBarre', 'NumInterne', 'NumInterne_CB', 'total_flags']
X = df.select_dtypes(include=[np.number]).drop(columns=drop_cols)
X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

# Séparation des données en un jeu d'entraînement et un jeu de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline XGBoost
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('xgb', XGBClassifier(
        n_estimators=200,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    ))
])

# 1. Validation croisée sur le jeu d'entraînement
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['precision', 'recall', 'f1']

cv_results = cross_validate(
    pipeline, X_train, y_train, cv=skf,
    scoring=scoring, return_train_score=False
)

# Affichage des résultats de la validation croisée
print("=== Cross-Validation 5-fold sur le jeu d'entraînement ===")
for metric in scoring:
    scores = cv_results[f'test_{metric}']
    print(f"{metric.capitalize():>8}: {scores.mean():.3f} ± {scores.std():.3f}")

# 2. Entraînement final sur l'ensemble d'entraînement
pipeline.fit(X_train, y_train)

# 3. Évaluation sur le jeu de test indépendant
test_score = pipeline.score(X_test, y_test)
print(f"\n=== Évaluation sur le jeu de test indépendant ===")
print(f"Score sur le jeu de test : {test_score:.3f}")

# 4. Sérialisation du modèle
joblib.dump(pipeline, 'anomaly_xgb_pipeline.joblib')
print("\nPipeline sauvegardé dans anomaly_xgb_pipeline.joblib")

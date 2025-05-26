import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd  # bibliothèque de manipulation de données
import numpy as np   # bibliothèque de calcul numérique
import re            # expressions régulières
import pandas as pd
from sqlalchemy import create_engine  # pour la connexion SQL
from config import engine


def corriger_champs_texte(df):
    # Nettoyage des colonnes texte
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
        df[col].replace(r'^(?:|nan|none)$', np.nan, regex=True, inplace=True)
    return df

def convertir_en_numerique(df, colonnes):
    # Conversion des colonnes spécifiées en numérique
    for col in colonnes:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def convertir_en_datetime(df, colonnes):
    # Conversion des colonnes spécifiées en datetime
    for col in colonnes:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

# Validation du format des codes article
motif_code = re.compile(r'^([A-Z]{4,5})(\d{1,3})([A-Z]{0,2})$')
def analyser_code(code):
    return 0 if motif_code.match(str(code)) else 1

# Fonctions de nettoyage par table

def nettoyer_ar_sfamille(engine):
    df = pd.read_sql('SELECT * FROM ar_sfamille', con=engine)
    df = corriger_champs_texte(df)
    df = df[df['Etat'].isin([1, 2])]
    df.drop(['IDChaineMontage', 'IDCategorieOFpardefaut'], axis=1, inplace=True)
    df = convertir_en_numerique(df, ['IDArSousFamille', 'IDArFamille'])
    df.dropna(how='all', inplace=True)
    return df.drop_duplicates()

def nettoyer_arfamille(engine):
    df = pd.read_sql('SELECT * FROM arfamille', con=engine)
    df = corriger_champs_texte(df)
    df = df[df['Etat'].isin([1, 2])]
    df.drop(['IDChaineMontage','QtePPP','Type','CodeDouane','SaisonObligatoire'], axis=1, inplace=True)
    df = convertir_en_numerique(df, ['IDArFamille'])
    df.dropna(how='all', inplace=True)
    return df.drop_duplicates()

def nettoyer_saison(engine):
    df = pd.read_sql('SELECT * FROM saison', con=engine)
    df = corriger_champs_texte(df)
    df = df[df['Etat'].isin([1, 2])]
    df.drop(['DateDebut','DateFin'], axis=1, inplace=True)
    df = convertir_en_numerique(df, ['IDSaison','IDTypeSaison'])
    df.dropna(how='all', inplace=True)
    return df.drop_duplicates()

def nettoyer_article(engine):
    df = pd.read_sql('SELECT * FROM article', con=engine)
    df = corriger_champs_texte(df)
    df = df[df['Etat'].isin([1, 2])]
    df.drop([
        'IDGamme','IDClient','TempsClient','Image','IdProcess','prixMP','CodeDouane','Valeur','Cadence',
        'IdArticleBase','SemiFini','ValeurTissu','ValeurFourniture','ValeurMP','TypeTarif','IdMeilleurOF','IDGrille',
        'NomenclatureValide','BaseStylisme','IDPatronnage','IDTypeMatiereBase','IDVarianteModele','IDGenre','IDBroderie',
        'IDSerigraphie','IDGarniture','IDTypeAccessoire','IDTransfert','IDCouleurGarniture','IDCouleurBroderie',
        'IDCouleurSerigraphie','PrixEmballage','StockMin','StockAlerte','AppliqueFodec','ValeurMPEuro','ValeurMPAutre',
        'TypeArticleService','ValeurMPTunisie','ValeurMPEuromed','AQL','AQLMineur','IDNiveauControle','AQLCritique',
        'IDCategorie','DateValidationNomenclature','NomenclatureValidePar','IDCategoriereclamation','IDCartouche',
        'IDArticleParent','isParent','QteFils','NbrPiecesColis','NbrColisPalette','Dimensions','TempsAtelier',
        'TempsFinitions','ArticleLong','IDTypeMatelassage','IDMP','IsMP','IDDecorArticle','IsSemiFini','PoidsEmballage',
        'TempsUnitaire','IDcomplexite','TauxSondageQlte','IDNorme','IDAr_Theme','Emballage','Boutonnage','PieceCarton',
        'SupportArt','ReseauArt','ReferenceFssr','DDV','FraisTransport','AutresFrais','IDFibreComposition',
        'IDArticleEtqEntretien','DateMEP','IDUsine','IDSupportArticle','Ecologique','TauxDefectueux','Publier',
        'CompositionMatiere','IDAr_Looks','PrixOutlet','Matiere','TauxCommissionCA','CODE_OLD','IDPlanComptable','PrixEtude'
    ], axis=1, inplace=True)
    df['Code'] = df['Code'].astype(str)
    df['Anomalie'] = df['Code'].apply(analyser_code)
    extrait = df['Code'].str.extract(motif_code)
    extrait.columns = ['prefixe','nombre','suffixe']
    df = pd.concat([df, extrait], axis=1)
    df = convertir_en_numerique(df, ['IDArticle','TauxTVA','NumInterne','IDSaison'])
    df = convertir_en_datetime(df, ['SaisiLe','ModifieLe'])
    df.dropna(how='all', inplace=True)
    return df.drop_duplicates()

def nettoyer_codebarre(engine):
    df = pd.read_sql('SELECT * FROM codebarre', con=engine)
    df = corriger_champs_texte(df)
    df.drop(['Indice','IDSerieArticle','isSynchronized','isSynchronizedWeb'], axis=1, inplace=True)
    df = convertir_en_numerique(df, ['IDCodeBarre','Prix','NumInterne'])
    df.dropna(how='all', inplace=True)
    return df.rename(columns={'IdEntite':'IDArticle','Prix':'Prix_CB','NumInterne':'NumInterne_CB'}).drop_duplicates()

def nettoyer_fournisseur(engine):
    df = pd.read_sql('SELECT * FROM fournisseur', con=engine)
    df = corriger_champs_texte(df)
    df = df[df['Etat'].isin([1,2])]
    df.drop([
        'Fax','isFournisseur','MF','Observations','Note','Type','FournitMP','FournitPF','FournitMB',
        'NumInterne','Echeance','TauxRetenueSource','DateExonerationRS','ExonerationRS','IsPDR','Timbre',
        'IDConditionReglement','IDModeReglement','ToleranceMAxAccepte','IDBanque','AdresseBanque','VilleBanque',
        'NumCompte','CodeSwift','IBAN','NonAssujettiTVA','DelaisLivraison','Reference','IDFournisseurParent',
        'Difference','AppliqueFodec','Login_FRS','IDCGAFournisseur','IsMP','IsPF','isService','CodeTVA','IDPlanComptable'
    ], axis=1, inplace=True)
    df = convertir_en_numerique(df, ['IDFournisseur','Chiffre','Reglements','Solde'])
    df.dropna(how='all', inplace=True)
    return df.drop_duplicates()

def nettoyer_tailles(engine):
    df = pd.read_sql('SELECT * FROM tailles', con=engine)
    df = corriger_champs_texte(df)
    df.drop(['LibTailleAR','LibTailleAutre','LibTailleGER','LibTailleUSA',
             'LibTailleSP','LibTailleGRK','isMilieu'], axis=1, inplace=True)
    df = convertir_en_numerique(df, ['IdTaille','Ordre'])
    df.dropna(how='all', inplace=True)
    return df.drop_duplicates()

# Création du dataset centralisé sans duplication de colonnes
def creer_master(engine):
    df_fam  = nettoyer_arfamille(engine)[['IDArFamille','Famille']]
    df_sais = nettoyer_saison(engine)[['IDSaison','Saison']]
    df_art  = nettoyer_article(engine)[[
        'IDArticle','Code','Article','IDAr_Couleur','IDArFamille',
        'Prix','PrixFac','PrixAchat','IDSaison','NumInterne','IDFournisseur','IDPays','SaisiLe'
    ]]
    df_cb   = nettoyer_codebarre(engine)[['IDCodeBarre','CodeBarre','IDArticle','IdTaille','Prix_CB','NumInterne_CB']]
    df_fou  = nettoyer_fournisseur(engine)[['IDFournisseur','Fournisseur']]  # suppression d'IDPays
    df_tail = nettoyer_tailles(engine)[['IdTaille','LibTaille']]
    master = (
        df_art
        .merge(df_cb, how='left', on='IDArticle')
        .merge(df_fam, how='left', on='IDArFamille')
        .merge(df_sais, how='left', on='IDSaison')
        .merge(df_fou, how='left', on='IDFournisseur')
        .merge(df_tail, how='left', on='IdTaille')
    )
    # Indicateurs relatifs métier
    master['marge_rel'] = (master['PrixFac'] - master['PrixAchat']) / master['PrixAchat']  # marge relative
    master['diff_prix'] = master['PrixFac'] - master['PrixAchat']  # écart absolu de prix
    return master

# Encodage intelligent : OHE pour Saison et Fournisseur, fréquence pour Famille
def encoder_categorielles(df):
    df = pd.get_dummies(
        df,
        columns=['Saison','Fournisseur'],
        prefix=['sai','fou'],
        drop_first=True
    )
    freq = df['Famille'].value_counts(normalize=True)
    df['fam_freq'] = df['Famille'].map(freq)
    return df



def dataframe_to_csv(df: pd.DataFrame,
                     file_path: str,
                     sep: str = ',',
                     index: bool = False,
                     encoding: str = 'utf-8') -> None:
    
    df.to_csv(file_path, sep=sep, index=index, encoding=encoding)
    print(f"DataFrame enregistré sous : {file_path}")



if __name__ == '__main__':
    df_master = creer_master(engine)
    df_encoded = encoder_categorielles(df_master)
    print("Nettoyage et merge des tables terminés.")

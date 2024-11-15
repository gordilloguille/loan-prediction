"""
Este módulo realiza un análisis exploratorio de datos (EDA) sobre el conjunto de datos de préstamos.
"""

# %% [markdown]
# # EDA - Exploratory Data Analysis

# %%
import pickle
import pandas as pd

# %%
def eda():
    """Realiza un análisis exploratorio de datos (EDA) sobre el 
    conjunto de datos de préstamos."""
    dataset_train = pd.read_csv("../data/raw/loan_sanction_train.csv")

    dataset_train.drop('Loan_ID', axis=1, inplace=True)

    dataset_train["Loan_Status"].value_counts(normalize=True)

    continuas = [col for col in dataset_train.columns if (
        (dataset_train[col].dtype in ["int64","float64"]) and (
            len(dataset_train[col].unique())>30))]
    # Obtenemos variables numericas con escala discreta
    discretas = [col for col in dataset_train.columns if (
        (dataset_train[col].dtype in ["int64","float64"]) and (
            len(dataset_train[col].unique())<=30))]
    # Obtenemos variables numericas con escala categoricas
    categoricas = [col for col in dataset_train.columns if (
        dataset_train[col].dtype in ["object"] )]
    # Guardamos las variables segun su escala del train
    variables_escaling = {
        'categoricas': categoricas,
        'continuas': continuas,
        'discretas': discretas
    }

    with open("../artifacts/variables_escaling.pkl","wb") as fve:
        pickle.dump(variables_escaling,fve)

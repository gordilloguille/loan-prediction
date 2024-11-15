"""
Este módulo se encarga de crear características para el modelo de 
predicción utilizando un conjunto de datos procesados.
"""

import pickle
import pandas as pd

def create_features():
    """Crea características para el modelo a
    partir del conjunto de datos de entrenamiento."""
    dataset_train = pd.read_csv("../data/raw/loan_sanction_train.csv")

    with open("../artifacts/variables_escaling.pkl","rb") as file:
        variables_escaling = pickle.load(file)

    dataset_train.drop('Loan_ID', axis=1, inplace=True)
    # Se puede observar que las variables que tienen nulos
    # se pueden imputar ya que no son muchos
    # ### 3.3.1 Imputamos Varibles Categoricas
    dataset_train["Gender"].value_counts(normalize=True)

    mode_gender = dataset_train['Gender'].mode()[0]
    dataset_train['Gender'] = dataset_train['Gender'].fillna(
        mode_gender)

    dataset_train["Married"].value_counts(normalize=True)

    mode_married = dataset_train['Married'].mode()[0]
    dataset_train['Married'] = dataset_train['Married'].fillna(
        mode_married)

    dataset_train["Dependents"].value_counts(normalize=True)

    mode_dependents = dataset_train['Dependents'].mode()[0]
    dataset_train['Dependents'] = dataset_train['Dependents'].fillna(
        mode_dependents)

    dataset_train["Self_Employed"].value_counts(normalize=True)

    mode_self_employed = dataset_train['Self_Employed'].mode()[0]
    dataset_train['Self_Employed'] = dataset_train['Self_Employed'].fillna(
        mode_self_employed)

    media_loanamount = dataset_train['LoanAmount'].mean()
    dataset_train['LoanAmount'] = dataset_train['LoanAmount'].fillna(
        media_loanamount)

    media_loan_amount_term = dataset_train['Loan_Amount_Term'].mean()
    dataset_train['Loan_Amount_Term'] = dataset_train['Loan_Amount_Term'].fillna(
        media_loan_amount_term)

    media_credit_history= dataset_train['Credit_History'].mean()
    dataset_train['Credit_History'] = dataset_train['Credit_History'].fillna(
        media_credit_history)

    dataset_train[variables_escaling["categoricas"]].describe()

    codificador_dependents = dataset_train['Dependents'].value_counts()
    dataset_train['Dependents'] = dataset_train['Dependents'].map(
        codificador_dependents)

    codificador_property_area = dataset_train['Property_Area'].value_counts()
    dataset_train['Property_Area'] = dataset_train['Property_Area'].map(
        codificador_property_area)

    for colanme in variables_escaling["categoricas"]:
        if len(dataset_train[colanme].unique()) <3:
            dataset_train[colanme] = pd.get_dummies(
                dataset_train[colanme], drop_first=True).astype(int)

    dataset_train.to_csv('../data/processed/features_for_model.csv', index=False)

    feature_eng_configs = {
        'codificador_property_Area': codificador_property_area,
        'codificador_dependents': codificador_dependents,
        'media_credit_history': media_credit_history,
        'media_loan_amount_term': media_loan_amount_term,
        'media_loanamount': media_loanamount,
        'mode_self_employed': mode_self_employed,
        'mode_dependents': mode_dependents,
        'mode_married': mode_married,
        'mode_gender': mode_gender,
    }

    with open("../artifacts/feature_eng_configs.pkl","wb") as file:
        pickle.dump(feature_eng_configs,file)

"""
Este módulo se encarga de realizar predicciones utilizando un modelo de aprendizaje automático
y un conjunto de datos procesados.
"""

# %%
import pickle
import pandas as pd

# %%
def predict():
    """Realiza predicciones sobre el conjunto de datos de prueba utilizando un modelo entrenado."""
    data_test = pd.read_csv("../data/raw/loan_sanction_test.csv")
    # Procesos de preparación de datos para predecir
    with open("../artifacts/feature_eng_configs.pkl", "rb") as file:
        feature_eng_configs = pickle.load(file)

    with open("../artifacts/variables_escaling.pkl", "rb") as file:
        variables_escaling = pickle.load(file)

    categorias = variables_escaling["categoricas"]
    categorias.remove('Loan_Status')
    # Eliminamos variables no útiles
    data_test.drop('Loan_ID', axis=1, inplace=True)
    # Imputamos la variable Gender
    data_test['Gender'] = data_test['Gender'].fillna(
        feature_eng_configs['mode_gender'])
    # Imputamos Married
    data_test['Married'] = data_test['Married'].fillna(
        feature_eng_configs['mode_married'])
    # Imputamos Dependents
    data_test['Dependents'] = data_test['Dependents'].fillna(
        feature_eng_configs['mode_dependents'])
    # Imputamos Self_Employed
    data_test['Self_Employed'] = data_test['Self_Employed'].fillna(
       feature_eng_configs['mode_self_employed'])
    # Imputamos LoanAmount
    data_test['LoanAmount'] = data_test['LoanAmount'].fillna(
       feature_eng_configs['media_loanamount'])
    # Imputamos Loan_Amount_Term
    data_test['Loan_Amount_Term'] = data_test['Loan_Amount_Term'].fillna(
       feature_eng_configs['media_loan_amount_term'])
    # Imputamos Credit_History
    data_test['Credit_History'] = data_test['Credit_History'].fillna(
       feature_eng_configs['media_credit_history'])
    # Codificación de Variable Dependents
    data_test['Dependents'] = data_test['Dependents'].map(
       feature_eng_configs['codificador_dependents'])
    # Codificación de Variable Property_Area
    data_test['Property_Area'] = data_test['Property_Area'].map(
       feature_eng_configs['codificador_property_area'])
    # Codificación de variables categóricas con One Hot Encoding con valores menores a 3
    for col_name in categorias:
        if len(data_test[col_name].unique()) < 3:
            data_test[col_name] = pd.get_dummies(
                data_test[col_name], drop_first=True).astype(int)

    # Estandarización con objetos scaler de train
    with open("../artifacts/std_scaler.pkl", "rb") as file:
        std_scaler = pickle.load(file)
    # Cargamos modelo ya entrenado
    with open("../models/model_svm_v1.pkl", "rb") as file:
        model_svm = pickle.load(file)

    x_data_test_std = std_scaler.transform(data_test)
    data_test_predicts = model_svm.predict(x_data_test_std)

    print(data_test_predicts)

if __name__ == "__main__":
    predict()

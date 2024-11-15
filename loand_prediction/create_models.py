"""
Este módulo se encarga de crear modelos de predicción utilizando un conjunto de datos procesados.
"""

# %%
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def create_models():
    """_summary_
    Creacion del modelo final SVM
    """
    dataset = pd.read_csv("../data/processed/features_for_model.csv")

    x = dataset.drop(["Loan_Status"], axis=1)
    y = dataset["Loan_Status"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True,
                                                        random_state=2025)

    # Configuramos y calculamos el StandardScaler

    std_scaler = StandardScaler()
    std_scaler.fit(x_train)  # Calcular los valores para el scaler

    # Guardamos el scaler configurado (con datos de train) como artefacto del modelo.
    with open("../artifacts/std_scaler.pkl", "wb") as file:
        pickle.dump(std_scaler, file)

    # Luego de probar varios modelos, el SVM fue el que mejor accuracy obtuvo.
    x_train_std = std_scaler.transform(x_train)

    modelo_svm = SVC(C=1, gamma=0.01, kernel='rbf')
    modelo_svm.fit(x_train_std, y_train)

    x_test_std = std_scaler.transform(x_test)

    y_preds_rf = modelo_svm.predict(x_test_std)

    accuracy = accuracy_score(y_test, y_preds_rf)
    print("Accuracy del modelo SVM:", accuracy)

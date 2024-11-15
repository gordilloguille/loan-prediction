"""
Este módulo contiene funciones para cargar datos desde archivos CSV.
"""

import pandas as pd

def load_data():
    """Carga los datos desde archivos CSV y los imprime."""
    dataset_train = pd.read_csv("../data/raw/loan_sanction_train.csv")
    print(dataset_train.head())
    print(dataset_train.columns)

    dataset_test = pd.read_csv("../data/raw/loan_sanction_test.csv")
    print(dataset_test.head())
    print(dataset_test.columns)

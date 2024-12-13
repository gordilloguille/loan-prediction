""" 
Este módulo permite realizar los pipelines correspondientes
"""

import os
import configparser
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from feature_engine.imputation import MeanMedianImputer
from feature_engine.imputation import CategoricalImputer
from feature_engine.encoding import OneHotEncoder
from feature_engine.encoding import CountFrequencyEncoder

def create_features_and_base_pipeline():
    """
    Realiza las configuraciones correspondientes del Pipeline
    utiilizando el archivo pipeline.cfg para definir las variables    
    """
    # Obtener el directorio actual
    project_path = os.getcwd()
    # leemos del configparser.
    config = configparser.ConfigParser()
    config.read(os.path.join(project_path, "pipeline.cfg"))

    loan_prediction_model = Pipeline([
        # Imputacion de variables continuas
        ("continues_var_mean_imputacion",MeanMedianImputer(
            imputation_method="mean",
            variables=config.get('continues', 'vars_to_impute').split(','))),

        #Imputacion de variables categoricas
        ("categorical_var_freq_imputation",CategoricalImputer(
            imputation_method="frequent",
            variables=config.get('categorical', 'vars_to_impute').split(','))),

        #Codificaciones de las variables categoricas
        ("categorical_encoding_ohe",OneHotEncoder(
            variables=config.get('categorical', 'ohe_vars').split(','),
            drop_last=True)),
        ("categorical_encoding_freq_enc",CountFrequencyEncoder(
            encoding_method="count",
            variables=config.get('categorical', 'frequency_enc_vars').split(','))),

        # Estandarizacion de variables
        ("feature_scaling",StandardScaler())
    ])

    with open(os.path.join(project_path,"artifacts","pipeline.pkl"), 'wb') as file:
        pickle.dump(loan_prediction_model, file)

create_features_and_base_pipeline()

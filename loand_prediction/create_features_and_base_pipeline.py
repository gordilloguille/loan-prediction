""" 
Este módulo permite realizar los pipelines correspondientes
"""

import os
import configparser
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.encoding import OneHotEncoder, CountFrequencyEncoder
from feature_engine.selection import DropFeatures


def sklearn_pipeline():
    """
    Realiza las configuraciones correspondientes del Pipeline
    utiilizando el archivo pipeline.cfg para definir las variables    
    """
    project_path = os.getcwd()
    dataset = pd.read_csv(os.path.join(project_path,"data","raw","loan_sanction_train.csv"))

    # leemos del configparser.
    config = configparser.ConfigParser()
    config.read(os.path.join(project_path, "pipeline.cfg"))

    # configuracipon del Pipeline
    x_features = dataset.drop(labels=config.get('general', 'vars_to_drop').split(','),axis=1)
    y_target = dataset[config.get('general', 'target').split(',')].map({'Y': 1, 'N': 0})

    x_train, x_test, y_train, y_test = train_test_split(
        x_features,y_target, test_size=0.3,shuffle=True,random_state=2025)

    loan_prediction_model = Pipeline([
        # eliminamos variables que no usaremos.
        ('delete_features', DropFeatures(
            features_to_drop=config.get('general', 'vars_to_drop').split(','))),
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

    loan_prediction_model.fit(x_train)

    x_features_processed = loan_prediction_model.transform(x_train)
    df_features_process = pd.DataFrame(
        x_features_processed,columns=x_train.columns)
    df_features_process[
        config.get('general', 'target').split(',')] = y_train.reset_index()[
            config.get('general', 'target').split(',')]

    # Almacenado de datos para entrenar los modelos.
    df_features_process.to_csv(
        os.path.join(project_path,"data",'processed","features_for_model.csv'), index=False)

    x_test[config.get('general', 'target').split(',')] = y_test
    x_test.to_csv(os.path.join(project_path,"data","processed","test_dataset.csv"), index=False)

    with open(os.path.join(project_path,"artifacts","pipeline.pkl"), 'wb') as file:
        pickle.dump(loan_prediction_model, file)

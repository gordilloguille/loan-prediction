""" 
Este módulo permite realizar el pipeline con el Modelo Correspondiente
"""
import os
import configparser
import pickle
import pandas as pd
import mlflow

def sklearn_pipeline_predict():
    """Leemos el Pipeline ya con el modelo correspondiente"""
    project_path = os.getcwd()
    with open(os.path.join(project_path,"artifacts","pipeline_model.pkl"), 'rb') as  file:
        loan_prediction_model_pipeline = pickle.load(file)

    # leemos del configparser.
    config = configparser.ConfigParser()
    config.read(os.path.join(project_path, "pipeline.cfg"))

    # configuración de servidor
    mlflow.set_tracking_uri(config.get('mlflow', 'server').split(','))
    mlflow.set_experiment("Loan Predict Model - Prediccion de Modelos")

    with mlflow.start_run():
        test_dataset = pd.read_csv(os.path.join(project_path,"data","raw","loan_sanction_test.csv"))
        test_dataset.drop(["Loan_ID"], axis=1, inplace=True)
        predictions = loan_prediction_model_pipeline.predict(test_dataset)

        # Almacenar las predicciones en un archivo CSV con fecha y hora actual
        timestamp = pd.Timestamp.now().strftime('%Y-%m-%d-%H-%M-%S')
        predictions_df = pd.DataFrame(predictions, columns=['Predicción'])
        predictions_df.to_csv(f'../data/predictions/predictions_{timestamp}.csv', index=False)

        mlflow.log_artifact(f'../data/predictions/predictions_{timestamp}.csv')
        print(f"Predicciones guardadas en data/predictions/predictions_{timestamp}.csv")
        mlflow.end_run()

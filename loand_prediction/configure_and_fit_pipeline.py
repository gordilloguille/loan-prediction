""" 
Este módulo permite realizar el pipeline correspondiente para el train
"""
import os
import configparser
import pickle
import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def sklearn_pipeline_train():
    """Crea modelos de predicción utilizando un conjunto de datos procesados"""    
    project_path = os.getcwd()
    data_train = pd.read_csv(os.path.join(project_path,"data","processed","features_for_model.csv"))
    data_test = pd.read_csv(os.path.join(project_path,"data","processed","test_dataset.csv"))

    # leemos del configparser.
    config = configparser.ConfigParser()
    config.read(os.path.join(project_path, "pipeline.cfg"))

    x_features = data_train.drop([config.get('general', 'target').split(',')], axis=1)
    y_target = data_train[config.get('general', 'target').split(',')]

    x_features_test = data_test.drop([config.get('general', 'target').split(',')], axis=1)
    y_target_test = data_test[config.get('general', 'target').split(',')]

    # ### Leemos el Pipeline pre-configurado

    with open(os.path.join(project_path,"artifacts","pipeline.pkl"), 'rb') as  file:
        loan_prediction_model_pipeline = pickle.load(file)

    x_features_test_arr = loan_prediction_model_pipeline.transform(x_features_test)
    df_features_test = pd.DataFrame(x_features_test_arr, columns=x_features_test.columns)
    df_features_test.head()

    # Entrenamiento de Modelos

    # 1. Lista de modelos para evaluar
    models = [
        ('Logistic Regression',1, LogisticRegression(penalty='l2',C=1.0,
                                                    solver='lbfgs',max_iter=100)),
        ('Logistic Regression',2, LogisticRegression(penalty='l1',C=0.5,
                                                    solver='liblinear',max_iter=200)),
        ('Logistic Regression',3, LogisticRegression(penalty=None,max_iter=500,
                                                    tol=1e-5,solver='saga')),
        ('Random Forest', 1,RandomForestClassifier(n_estimators=50, max_depth=5)),
        ('Random Forest', 2,RandomForestClassifier(n_estimators=100, max_depth=10)),
        ('Random Forest', 3,RandomForestClassifier(n_estimators=300, max_depth=15)),
        ('XGBoost', 1,RandomForestClassifier(n_estimators=50, max_depth=5)),
        ('XGBoost', 2,RandomForestClassifier(n_estimators=100, max_depth=10)),
        ('XGBoost', 3,RandomForestClassifier(n_estimators=200, max_depth=15)),
        ('SVM',1, SVC(kernel='linear', C=1)),
        ('SVM',2, SVC(kernel='rbf', C=10)),
        ('SVM',3, SVC(kernel='poly', C=0.1, degree=3)),
        ('KNN',1, KNeighborsClassifier(n_neighbors=3, metric='euclidean')),
        ('KNN',2, KNeighborsClassifier(n_neighbors=5, metric='manhattan')),
        ('KNN',3, KNeighborsClassifier(n_neighbors=7, metric='minkowski', p=3))
    ]

    # Configuración de servidor
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Loan Predict Model - Training Modelos")

    # Entrenamiento de los modelos
    results = []

    with mlflow.start_run():
        for name, num, model in models:
            model.fit(x_features, y_target)
            y_pred = model.predict(df_features_test)
            acc = accuracy_score(y_target_test, y_pred)
            results.append((name,num, acc))

            # registramos hiper-parametros
            mlflow.log_params(model.get_params())

            # registramos métricas
            mlflow.log_metric("accuracy score", acc)

            # registramos modelo y entrenado.
            mlflow.sklearn.log_model(model, name)

            mlflow.end_run()

    # 4. Encontrar el mejor modelo
    best_model = max(results, key=lambda x: x[2])

    modelo = [model.get_params() for name, num, model in models
              if name == best_model[0] and num == best_model[1]]

    modelo = [model for name, num, model in models if
              name == best_model[0] and num == best_model[1]]

    if best_model[0]=="Logistic Regression":
        loan_prediction_model_pipeline.steps.append(
            ("modelo_regresion_logistica",modelo[0])
        )
    if best_model[0]=="Random Forest":
        loan_prediction_model_pipeline.steps.append(
            ("modelo_random_forest",modelo[0])
        )
    if best_model[0]=="XGBoost":
        loan_prediction_model_pipeline.steps.append(
            ("modelo_xgboost",modelo[0])
        )
    if best_model[0]=="SVM":
        loan_prediction_model_pipeline.steps.append(
            ("modelo_svm",modelo[0])
        )
    if best_model[0]=="KNN":
        loan_prediction_model_pipeline.steps.append(
            ("modelo_knn",modelo[0])
        )

    train_dataset = pd.read_csv(os.path.join(project_path, "data","raw","loan_sanction_train.csv"))
    train_dataset.drop(["Loan_ID"], axis=1, inplace=True)
    train_dataset_features = train_dataset.drop("Loan_Status", axis=1)
    train_dataset_target = train_dataset["Loan_Status"].map({'Y': 1, 'N': 0})

    loan_prediction_model_pipeline.fit(train_dataset_features,train_dataset_target)

    with open(os.path.join(project_path,"artifacts","pipeline_model.pkl"), 'wb') as file:
        pickle.dump(loan_prediction_model_pipeline, file)

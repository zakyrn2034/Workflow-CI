import mlflow
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll import scope
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, log_loss, roc_auc_score
from automate import cap_outlier,del_missing
import joblib

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("Star Classification Project")

path = "./MSML Project/"
file_path = path + "star_classification-headers.csv"
save_path = path + "pipeline.joblib"

data = pd.read_csv(path + "star_classification.csv")
X = data.drop(columns=["class"])
y = data["class"]

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
preprocessor = joblib.load(path + "pipeline.joblib")
X_train = preprocessor.transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)

input_example = X_train[0:5]
criterion_options = ['gini', 'entropy', 'log_loss']

space = {
    'criterion': hp.choice('criterion', criterion_options),
    'max_depth': hp.choice('max_depth', [None, scope.int(hp.quniform('max_depth_int', 10, 25, 1))]),
    'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1))
}

def objective(params):
    with mlflow.start_run(nested=True) as run:
        #define params
        params = {
            'ccp_alpha': params.get('ccp_alpha', 0.0),
            'class_weight': params.get('class_weight', None),
            'criterion': params.get('criterion', 'gini'),
            'max_depth': params.get('max_depth', None),
            'max_features': params.get('max_features', None),
            'max_leaf_nodes': params.get('max_leaf_nodes', None),
            'min_impurity_decrease': params.get('min_impurity_decrease', 0.0),
            'min_samples_leaf': params.get('min_samples_leaf', 1),
            'min_samples_split': params.get('min_samples_split', 2),
            'min_weight_fraction_leaf': params.get('min_weight_fraction_leaf', 0.0),
            'monotonic_cst': params.get('monotonic_cst', None),
            'random_state': params.get('random_state', None),
            'splitter': params.get('splitter', 'best')
        }

        #Define model
        model = DecisionTreeClassifier(
            criterion=params.get('criterion','gini'),
            max_depth=params.get('max_depth',10),
            min_samples_split=params.get('min_samples_split',5),
            random_state=42
            )
        
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)
        y_train_proba = model.predict_proba(X_train)

        res = classification_report(y_test,y_pred,output_dict=True)
        
        #get metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'test_avg_f1_score': res['macro avg']['f1-score'],
            'test_weighted_f1_score': res['weighted avg']['f1-score'],
            'DecisionTreeClassifier_score_X_test': model.score(X_test, y_test),
            'training_accuracy_score': accuracy_score(y_train, y_train_pred),
            'training_f1_score': f1_score(y_train, y_train_pred, average='micro'),
            'training_log_loss': log_loss(y_train, y_train_proba),
            'training_precision_score': precision_score(y_train, y_train_pred, average='micro'),
            'training_recall_score': recall_score(y_train, y_train_pred, average='micro'),
            'training_roc_auc': roc_auc_score(y_train, y_train_proba, multi_class='ovr', average='macro'),
            'training_score': model.score(X_train, y_train)
        }

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        return {'loss': -res['accuracy'], 'status': STATUS_OK}


with mlflow.start_run() as run:
    mlflow.set_tag("mlflow.user", "Kirby")

    training_dataset: mlflow.data.pandas_dataset.PandasDataset = mlflow.data.from_pandas(
        df=X_train, 
        source=str(file_path),
        name="stellar_classification_train"
    )
    
    testing_dataset: mlflow.data.pandas_dataset.PandasDataset = mlflow.data.from_pandas(
        df=X_test, 
        source=str(file_path),
        name="stellar_classification_test"
    )

    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=10,  # Jumlah evaluasi
                trials=trials)
    
    # Log best params
    best_params = {k: v[0] if isinstance(v, list) else v for k, v in trials.best_trial['misc']['vals'].items()}
    best_params = {
            'ccp_alpha': best_params.get('ccp_alpha', 0.0),
            'class_weight': best_params.get('class_weight', None),
            'criterion': best_params.get('criterion', 'gini'),
            'max_depth': best_params.get('max_depth', None),
            'max_features': best_params.get('max_features', None),
            'max_leaf_nodes': best_params.get('max_leaf_nodes', None),
            'min_impurity_decrease': best_params.get('min_impurity_decrease', 0.0),
            'min_samples_leaf': best_params.get('min_samples_leaf', 1),
            'min_samples_split': best_params.get('min_samples_split', 2),
            'min_weight_fraction_leaf': best_params.get('min_weight_fraction_leaf', 0.0),
            'monotonic_cst': best_params.get('monotonic_cst', None),
            'random_state': best_params.get('random_state', None),
            'splitter': best_params.get('splitter', 'best')
        }
    if 'criterion' in best_params and best_params['criterion'] is not None:
        best_params['criterion'] = criterion_options[int(best_params['criterion'])]
    for key in ['max_depth', 'min_samples_split']:
        if key in best_params and best_params[key] is not None:
            best_params[key] = int(best_params[key])
    mlflow.log_params(best_params)

    # Retrains best model (for metrics/artifact)

    best_model = DecisionTreeClassifier(
            criterion=best_params.get('criterion','gini'),
            max_depth=best_params.get('max_depth',10),
            min_samples_split=best_params.get('min_samples_split',5),
            random_state=42
            )
    
    best_model.fit(X_train,y_train)
    y_pred = best_model.predict(X_test)
    y_train_pred = best_model.predict(X_train)
    y_train_proba = best_model.predict_proba(X_train)

    res = classification_report(y_test,y_pred,output_dict=True)
    
    #get metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'test_avg_f1_score': res['macro avg']['f1-score'],
        'test_weighted_f1_score': res['weighted avg']['f1-score'],
        'DecisionTreeClassifier_score_X_test': best_model.score(X_test, y_test),
        'training_accuracy_score': accuracy_score(y_train, y_train_pred),
        'training_f1_score': f1_score(y_train, y_train_pred, average='micro'),
        'training_log_loss': log_loss(y_train, y_train_proba),
        'training_precision_score': precision_score(y_train, y_train_pred, average='micro'),
        'training_recall_score': recall_score(y_train, y_train_pred, average='micro'),
        'training_roc_auc': roc_auc_score(y_train, y_train_proba, multi_class='ovr', average='macro'),
        'training_score': best_model.score(X_train, y_train)
    }

    mlflow.log_metrics(metrics)
    mlflow.log_input(training_dataset, context="training")
    mlflow.log_input(testing_dataset, context="validation")

    # Dumps model
    joblib.dump(best_model,"run_best_model.joblib")
    mlflow.log_artifact("run_best_model.joblib")

    print(f'\n--- Finished Run ---\n')
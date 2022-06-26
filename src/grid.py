# Import mlflow
import json
import requests
import pandas as pd
from requests.structures import CaseInsensitiveDict
import mlflow
import mlflow.sklearn

mlflow.set_experiment('test_mlflowdb')
mlflow.set_tracking_uri("sqlite:///mlflow.db")
 # Import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC

 # Load data
data = load_breast_cancer()
x = data['data']
y = data['target']

# Build model using grid search and cross fold validation
metrics = ['f1', 'recall', 'precision']

models = {
    'ExtraTreesClassifier': ExtraTreesClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'SVC': SVC()
}

params = {
    'ExtraTreesClassifier': { 'n_estimators': [16, 32], 'max_depth': [10, 15] },
    'RandomForestClassifier': { 'n_estimators': [16, 32], 'max_depth': [10, 15]},
    'AdaBoostClassifier':  { 'n_estimators': [16, 32] },
    'GradientBoostingClassifier': { 'n_estimators': [16, 32], 'learning_rate': [0.8, 1.0] },
    'SVC': [{'kernel': ['linear'], 'C': [1, 10]}, {'kernel': ['rbf'], 
            'C': [1, 10], 'gamma': [0.001, 0.0001]}
            ]
}


for name in models :
  print('Start : ', name)
  model = models[name]
  clf = GridSearchCV(model, params[name], scoring=metrics, refit='f1', verbose=3)
  clf.fit(x, y)

  # Log artifacts to MLflow
  with mlflow.start_run():
    mlflow.log_param("Model Name", name)
    mlflow.sklearn.log_model(clf.best_estimator_, "bestmodel")
    mlflow.log_metric('best score', clf.best_score_)
    mlflow.log_artifact("src/grid.py")
    for k in clf.best_params_.keys():
        mlflow.log_param(k, clf.best_params_[k])
    
    run_id = mlflow.active_run().info.run_uuid
    print("Model saved in run %s" % run_id)


# #invoke the model : Method1
# logged_model = 'runs:/{}/bestmodel'.format(run_id)

# # Load model as a PyFuncModel.
# loaded_model = mlflow.pyfunc.load_model(logged_model)

# # Predict on a Pandas DataFrame.
# print(loaded_model.predict(pd.DataFrame(x)))


# # Method 2
# ## Run the following cmd before calling the python code below
# ## MLFLOW_TRACKING_URI=http://localhost:5000 mlflow models serve --no-conda -m "models:/modela/Production" -p 4242



# # Load data
# data = pd.DataFrame(x)

# url = "http://127.0.0.1:4242/invocations"

# headers = CaseInsensitiveDict()
# headers["Content-Type"] = "application/json; format=pandas-records"

# data = data.to_dict(orient ='records' )

# response = requests.post(url, headers=headers, data=json.dumps(data))

# print(response)
# print(response.text)

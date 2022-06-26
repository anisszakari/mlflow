# Import mlflow
import mlflow
import mlflow.sklearn

mlflow.set_experiment('test_mlflowdb')
mlflow.set_tracking_uri("sqlite:///mlflow.db")
 # Import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
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
    mlflow.sklearn.log_model(clf.best_estimator_, "best model")
    mlflow.log_metric('best score', clf.best_score_)
    mlflow.log_artifact("src/grid.py")
    for k in clf.best_params_.keys():
        mlflow.log_param(k, clf.best_params_[k])
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
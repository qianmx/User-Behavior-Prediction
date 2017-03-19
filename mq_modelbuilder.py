from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

def logistic_model(X_train, y_train):
  """
  use crossvalidation (CV) to report the best parameter 'C'
  parameter C: Inverse of regularization strength; must be a positive float.
  Check LogisticRegression() in sklearn for more information
  """
  print('Train logistic regression Model')
#   model = GridSearchCV(
#         estimator=LogisticRegression(penalty='l1'),
#         param_grid={'C': [0.05,0.08,0.1]},
#         scoring='log_loss',
#         cv=5
#   )
  model = LogisticRegression(C=0.06, solver='sag')
  model.fit(X_train, y_train)
  return model


def gradient_model(X_train, y_train):
  """
  use crossvalidation (CV) to report the best parameter 'C'
  parameter C: Inverse of regularization strength; must be a positive float.
  Check LogisticRegression() in sklearn for more information
  """
  print('Train Gradientdescent Model')
  model = GridSearchCV(
      estimator=GradientBoostingClassifier(),
      param_grid={'n_estimators': [100,200,300],'learning_rate':[0.2,0.4,0.6], 'max_depth':[20,30,50]},
      scoring='log_loss',
      cv=5
  )
  model.fit(X_train, y_train)
  return model


def adaboost_model(X_train, y_train):
    print('Train adaboost Model')
    model = GridSearchCV(
        estimator=AdaBoostClassifier(),
        param_grid={'n_estimators': [200, 300], 'learning_rate': [0.2, 0.4, 0.6]},
        scoring='log_loss',
        cv=5
    )
    model.fit(X_train, y_train)
    return model


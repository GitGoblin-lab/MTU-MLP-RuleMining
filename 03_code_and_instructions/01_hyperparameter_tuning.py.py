
import os

os.environ['JOBLIB_TEMP_FOLDER'] = 'D:/tmp'

import pylab
import xgboost as xgb
import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib as mpl
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics,preprocessing
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,RandomizedSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Categorical
from skopt.space import Real, Integer, Categorical




data = pd.read_csv('trainH.csv',header = None)    
X_train= data.iloc[:161,:26]                       
y_train= data.iloc[:161,26]
data2 = pd.read_csv('testH.csv',header = None)    
X_test = data2.iloc[:69,:26]                      
y_test = data2.iloc[:69,26]



# step1.
# param_test1 = {
#   'n_estimators':range(10,300,5),
#   'learning_rate':np.linspace(0.01, 0.5, 10)
# }

# gsearch1 = GridSearchCV(estimator= xgb.XGBRegressor(
#         max_depth=5,
#         min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
#         nthread=4, scale_pos_weight=1, seed=27, reg_alpha=0, reg_lambda=1),
#         param_grid=param_test1, cv=5
#     )




# # step2.
# param_test2 = {
#   'max_depth':range(3,30,2),
#   'min_child_weight':np.arange(0.5,6,0.2),
# }
# gsearch1 = GridSearchCV(estimator= xgb.XGBRegressor(
#         n_estimators=80,
#         learning_rate=0.1938, gamma=0, subsample=0.8, colsample_bytree=0.8,
#         nthread=4, scale_pos_weight=1, seed=27, reg_alpha=0, reg_lambda=1),
#         param_grid=param_test2, cv=5
#     )




# step3.
# param_test3 = {
#   'gamma':[i/100.0 for i in range(0, 5)],
# }
# gsearch1 = GridSearchCV(estimator= xgb.XGBRegressor(
#         n_estimators=56,
#         learning_rate=0.32, max_depth=8, min_child_weight=0.74, subsample=0.8, colsample_bytree=0.8,
#         nthread=4, scale_pos_weight=1, seed=27, reg_alpha=0, reg_lambda=1),
#         param_grid=param_test3, cv=5
#     )




# step4.
# param_test4 = {
#   'subsample': np.arange(0.3,1,0.05),
#   'colsample_bytree': np.arange(0.3,1,0.05),
# }
# gsearch1 = GridSearchCV(estimator= xgb.XGBRegressor(
#         n_estimators=80,
#         max_depth=6, min_child_weight=3.5242, learning_rate=0.1938, gamma=0, 
#         nthread=4, scale_pos_weight=1, seed=27, 
#         reg_alpha=0, reg_lambda=1),
#         param_grid=param_test4, cv=5
#     )



# step5.
# param_test5 = {
#   'reg_alpha':np.arange(0,5,0.25),
#   'reg_lambda':np.arange(0,5,0.25)
# }
# gsearch1 = GridSearchCV(estimator= xgb.XGBRegressor(
#         n_estimators=80, max_depth=6, min_child_weight=3.5242,
#         learning_rate=0.1938, gamma=0, subsample=0.8608, colsample_bytree=0.4,
#         nthread=4, scale_pos_weight=1, seed=27),
#         param_grid=param_test5, cv=5
#     )




# =================GS===========================
# model = XGBRegressor()
# param_grid = param_test5
# gsearch1 = GridSearchCV(model, param_grid, cv=5,n_jobs=-1)
# #=================GS==============================



# #==================BO============================
model = XGBRegressor()
param_grid = {
  'n_estimators':[80],
  'learning_rate':[0.1938],
  'max_depth':[6],
  'min_child_weight':[3.5242],
  'gamma':[0],
  'subsample':Real(0.5,1),
  'colsample_bytree':Real(0.5,0.8),
  'reg_alpha':[0], 
    'reg_lambda':[1]
  }
gsearch1 = BayesSearchCV(
    model,
    param_grid,
    cv=5,
    n_iter=60,
    n_jobs=-1,
    random_state=42
)
##==================BO=========================


gsearch1.fit(X_train, y_train)


# print("grid score: ", gsearch1.grid_scores_)
means = gsearch1.cv_results_['mean_test_score']
params = gsearch1.cv_results_['params']
for mean,param in zip(means,params):
    print("%f  with:   %r" % (mean,param))
print("Best parameters: ", gsearch1.best_params_)
print("Best score: ", gsearch1.best_score_)


# df = pd.DataFrame([params])


# df.to_excel('best_params.xlsx', index=False)


# ===================== =====================
# 1. 
results_dict = gsearch1.cv_results_

# 2. 
all_results = pd.DataFrame(results_dict['params'])

# 3. 
all_results['mean_test_score'] = results_dict['mean_test_score']

# 4. 
all_results.to_excel('grid_search_ordered_results.xlsx', index=False)

print("grid_search_ordered_results.xlsx")
# =========================================================================



# 4.
y_test_pre = gsearch1.predict(X_test)

# 5.
rmse = sqrt(mean_squared_error(np.array(list(y_test)), np.array(list(y_test_pre))))
print("rmse:", rmse)





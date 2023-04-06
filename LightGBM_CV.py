
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score, train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt

def openFile(filename):
    file = pd.read_csv(filename)
    # print(file)
    return file

def split(data):
    # print(data.shape)
    X = data.iloc[:,1:-1]
    y = data.iloc[:,-1]
    # print(X)
    # print(y)
    return X, y

def LightGBMCrossValidation(data):
    # Set up Optuna study and run optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objectiveCV, n_trials=100)
    
    # Print best hyperparameters and score
    print("Best parameters:", study.best_params)
    print("Best score:", study.best_value)
    
    # Set up LightGBM model with best hyperparameters
    best_params = study.best_params
    best_model = lgb.LGBMRegressor(**best_params)
    
    # Split data into features and target variables
    X, y = split(data)

    best_model.fit(X, y)

    # Compute the mean squared error of the final model
    y_pred = best_model.predict(X)
    mse = np.sqrt(mean_squared_error(y, y_pred))
    print("RMSE: ", mse)
    # Perform cross-validation and print scores
    # scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    # print("Cross-validation scores:", scores)
    # print("Mean score:", scores.mean())

def objectiveCV(trial):
    # Set hyperparameters to be tuned
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 2),
        'max_depth': trial.suggest_int('max_depth', -1, 30),
        'min_child_samples': trial.suggest_int('min_child_samples', 2, 10),
    }
    
    # Set up LightGBM model with hyperparameters
    model = lgb.LGBMRegressor(**params)
    
    # Split data into features and target variables
    X, y = split(data)
    
    # Perform cross-validation and return mean RMSE
    score = np.sqrt(-cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error'))
    return score.mean()

def lightGBMLOOOptuna(data):
    # Set up Optuna study and run optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objectiveLOO, n_trials=100)

    print("Best Parameters: ", study.best_params)
    print("Best Scorre:", study.best_value)

    # lightGBMLOO(data, study.best_params)

if __name__ == '__main__':
    print("running...")
    # Aquire data
    data = openFile("selected_features.csv") # 18 patients, 10 x, 1 y 
    # otherRF(data)
    # # RF(data) 
    # lightGBMLOO(data)
    lightGBMLOOOptuna(data)
    # LightGBMCrossValidation(data)

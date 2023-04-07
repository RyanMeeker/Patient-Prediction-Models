import numpy as np
from sklearn.model_selection import LeaveOneOut
import pandas as pd
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt


def split(data):
    # print(data.shape)
    X = data.iloc[:,1:-1]
    y = data.iloc[:,-1]
    # print(X)
    # print(y)
    return X, y

def lightGBMLOO(data, params): 
    X, y = split(data)
    actual = y
    loo = LeaveOneOut()
    rmse, feature_importances = [], []
    predicted = np.zeros_like(y)

    xgb_model = xgb.XGBRegressor(**params)

    print("Training Fold: ")
    for idx, (train_idx, test_idx) in enumerate(loo.split(X)):        
        print(idx, end=" ")
        # Get the training and testing data for the fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # eval_set = [(X_test, y_test)]
        # Fit the LightGBM model to the training data
        xgb_model.fit(X_train, y_train) #eval_set=eval_set

        # Make predictions on the test data
        y_pred = xgb_model.predict(X_test)

        rmse.append( np.sqrt( mean_squared_error( y_test, y_pred )))
        
        feature_importances.append(xgb_model.feature_importances_)
        predicted[idx] = y_pred

    # Compute the accuracy metrics of the model using the predicted labels and true labels
    mrmse = np.mean(rmse)

    for idx, x in enumerate(predicted):
        predicted[idx] = np.mean(x)

    #print results
    print()
    print( ("-" * 12), "XGBoost", ("-" * 12) )
    print("RMSE: ", mrmse)

    # Plots
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    mean_feature_importances = np.mean(feature_importances, axis=0)

    # Feature Importances
    axs[0].bar(X.columns, mean_feature_importances)
    axs[0].set_xticks(range(len(X.columns)))
    axs[0].set_xticklabels(X.columns, rotation=90, ha='right')
    axs[0].set_ylabel("Importance")
    axs[0].set_xlabel("Feature")
    axs[0].set_title("Feature Importances")

    # Residual Plot
    residuals = [a - p for a, p in zip(actual, predicted)]
    for idx, x in enumerate(residuals):
        residuals[idx] = x / len(residuals) #np.std(residuals)

    # = residuals / len(residuals)    #np.std(residuals)
    patient = np.arange(len(y))
    axs[1].scatter(patient, residuals)
    axs[1].set_xlabel("Patient")
    axs[1].set_ylabel("Actual-Pred / n")
    axs[1].set_title("Residual Plot")
    axs[1].axhline(y=0, color='blue', linestyle='-')

    # plt.tight_layout()
    plt.show()

    rounded_actual = [round(num, 1) for num in actual]
    rounded_predicted = [round(num, 1) for num in predicted]

    # # Actual vs Predicted
    fig = plt.figure(figsize=(15, 5))
    bar_width = 0.4
    x = np.arange(len(actual))
    plt.bar(x - 0.2, actual, width=bar_width, label='Actual', color='deepskyblue')
    plt.bar(x + 0.2, predicted, width=bar_width, label='Predicted', color='steelblue')
    plt.legend()
    plt.xlabel("Patient")
    plt.ylabel("Value")
    plt.title("Actual vs Predicted")
    
    plt.show()
    
    print("Actual: ", *rounded_actual, sep=', ')
    print("Predct: ", *rounded_predicted, sep=', ')




def objectiveLOO(trial):
    # Set hyperparameters to be tuned
    params = {
        'max_depth': trial.suggest_int('max_depth', 1, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0),
        'subsample': trial.suggest_float('subsample', 0.01, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0),
        'eval_metric': 'mlogloss',
        'use_label_encoder': False
    } 

    data = pd.read_csv("selected_features.csv")
    X, y = split(data)
    loo = LeaveOneOut()
    rmse = []
    xgb_model = xgb.XGBRegressor(**params)
                                      
    for idx, (train_idx, test_idx) in enumerate(loo.split(X)):        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        xgb_model.fit(X_train, y_train) #eval_set=eval_set
        y_pred = xgb_model.predict(X_test)
        rmse.append( np.sqrt( mean_squared_error( y_test, y_pred )))

    mrmse = np.mean(rmse)
    # print("MRMSE From OPTUNA: ", mrmse)
    return mrmse

def lightGBMLOOOptuna(data, n):
    # Set up Optuna study and run optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objectiveLOO, n_trials=n)

    print("Best Parameters: ", study.best_params)
    print("Best Scorre:", study.best_value)

    lightGBMLOO(data, study.best_params)

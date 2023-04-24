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

def ParamDyn():
    # Dynamic
    param = {'max_depth': 2, 'learning_rate': 0.07741428607518575, 'n_estimators': 79, 'min_child_weight': 6, 'gamma': 0.41458551141162997, 'subsample': 0.9907256921382913, 'colsample_bytree': 0.1681939392209627, 'reg_alpha': 0.09392755631098886, 'reg_lambda': 0.34314277693875533}
    return param

def ParamStatic():
    # Static
    param =  {'max_depth': 7, 'learning_rate': 0.04795554182253582, 'n_estimators': 294, 'min_child_weight': 2, 'gamma': 0.5822168323504584, 'subsample': 0.9965497814657729, 'colsample_bytree': 0.7985839849384547, 'reg_alpha': 0.8392046856779557, 'reg_lambda': 0.7671779054732297}
    return param

def XGBoost(data, params): 
    X, y = split(data)
    actual = y
    loo = LeaveOneOut()
    feature_importances = []
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

        feature_importances.append(xgb_model.feature_importances_)
        predicted[idx] = y_pred

    diff = y - predicted
    squared_diff = diff ** 2
    mean_squared_error = squared_diff.mean()
    rmse = mean_squared_error ** 0.5
    # Compute the accuracy metrics of the model using the predicted labels and true labels
    mrmse = np.mean(rmse)

    for idx, x in enumerate(predicted):
        predicted[idx] = np.mean(x)

    #print results
    print()
    print( ("-" * 12), "XGBoost", ("-" * 12) )
    print("RMSE: ", mrmse)

    mean_feature_importances = np.mean(feature_importances, axis=0)

    print("Feature Importance Values: ", *mean_feature_importances, sep=', ')

    rounded_actual = [round(num, 4) for num in actual]
    rounded_predicted = [round(num, 4) for num in predicted]
    
    print("Actual: ", *rounded_actual, sep=', ')
    print("Predct: ", *rounded_predicted, sep=', ')

    return mean_feature_importances, X, actual, predicted




def objectiveLOO(trial, filename):
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

    data = pd.read_csv(filename)
    X, y = split(data)
    loo = LeaveOneOut()
    predicted = np.zeros_like(y)

    xgb_model = xgb.XGBRegressor(**params)
                                      
    for idx, (train_idx, test_idx) in enumerate(loo.split(X)):        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        xgb_model.fit(X_train, y_train) #eval_set=eval_set
        predicted[idx] = xgb_model.predict(X_test)

    diff = y - predicted
    squared_diff = diff ** 2
    mean_squared_error = squared_diff.mean()
    rmse = mean_squared_error ** 0.5

    return rmse

def XGBoostOptuna(filename, n):
    # Set up Optuna study and run optimization
    study = optuna.create_study(direction='minimize')
    objective_fn = lambda trial: objectiveLOO(trial, filename)
    study.optimize(objective_fn, n_trials=n)

    print("Best Parameters: ", study.best_params)
    # print("Best Scorre:", study.best_value)
    return study.best_params

def ActualvsPredict(actual, static_predicted, dyn_predicted):
    # Actual vs Predicted
    fig = plt.figure(figsize=(20, 5))
    bar_width = 0.2
    x = np.arange(len(actual))

    plt.bar(x - bar_width, actual, width=bar_width, label='Actual', color='gold')
    plt.bar(x, static_predicted, width=bar_width, label='Static Predicted', color='steelblue')
    plt.bar(x + bar_width, dyn_predicted, width=bar_width, label='Dynamic Predicted', color='mediumorchid')

    plt.legend()
    plt.xlabel("Patient Number", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.title("Actual vs Predicted XGBoost", fontsize=20)
    plt.xticks(x, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', alpha=0.5)
    plt.show()

def FeatureImportance(static_mean_feature_importances, dyn_mean_feature_importances, static_X, dyn_X):
    fig, ax = plt.subplots(figsize=(15,7))

    # plot the static data
    static_features = pd.DataFrame(static_mean_feature_importances, index=static_X.columns, columns=['importance'])
    ax.barh(static_features.index, static_features['importance'], color='steelblue', label='Static')

    # plot the dynamic data
    dyn_features = pd.DataFrame(dyn_mean_feature_importances, index=dyn_X.columns, columns=['importance'])
    ax.barh(dyn_features.index, dyn_features['importance'], color='mediumorchid', label='Dynamic')

    # set axis labels and title
    ax.set_xlabel('Importance')
    ax.set_ylabel('Features')
    ax.set_title('XGBoost Static and Dynamic Feature Importance')

    # add legend
    ax.legend(loc='upper right')

    plt.show()

def residualPlot(static_actual, static_predicted, dyn_actual, dyn_predicted):

    static_residuals = [a - p for a, p in zip(static_actual, static_predicted)]
    for idx, x in enumerate(static_residuals):
        static_residuals[idx] = x / len(static_residuals) #np.std(residuals)

    dyn_residuals = [a - p for a, p in zip(dyn_actual, dyn_predicted)]
    for idx, x in enumerate(dyn_residuals):
        dyn_residuals[idx] = x / len(dyn_residuals) #np.std(residuals)

    fig, ax = plt.subplots(figsize=(20,5))
    # plot first set of residuals
    x1 = range(len(static_residuals))
    ax.scatter(x1, static_residuals, alpha=0.5, label='Static', color = 'steelblue')
    for i in x1:
        ax.plot([i, i], [0, static_residuals[i]], c='steelblue', alpha=0.5)
    # plot second set of residuals
    x2 = range(len(dyn_residuals))
    ax.scatter(x2, dyn_residuals, alpha=0.5, label='Dynamic', color = 'mediumorchid')
    for i in x2:
        ax.plot([i, i], [0, dyn_residuals[i]], c='mediumorchid', alpha=0.5)
    # set axis labels and title
    ax.set_xlabel('Patient')
    ax.set_ylabel('Residuals')
    ax.set_title('XGBoost Residuals')

    # add legend
    ax.legend()
    ax.axhline(y=0, color='black', alpha=0.5)
    plt.show()
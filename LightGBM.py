import numpy as np
from sklearn.model_selection import LeaveOneOut
import pandas as pd
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
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
    param = {'n_estimators': 93, 'num_leaves': 182, 'learning_rate': 1.1616608816994785, 'max_depth': 1, 'min_child_samples': 6, 'min_child_weight': 0.9040865766513123}
    return param

def ParamStatic():
    # Static
    param =  {'n_estimators': 196, 'num_leaves': 125, 'learning_rate': 0.0061479977675225825, 'max_depth': 14, 'min_child_samples': 16, 'min_child_weight': 0.7782067527243991}
    return param

def lightGBMLOO(data, params): 
    X, y = split(data)
    actual = y
    loo = LeaveOneOut()
    feature_importances = []
    predicted = np.zeros_like(y)

    lgb_model = lgb.LGBMRegressor(**params)

    print("Training Fold: ")
    for idx, (train_idx, test_idx) in enumerate(loo.split(X)):        
        print(idx, end=" ")
        # Get the training and testing data for the fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # eval_set = [(X_test, y_test)]
        # Fit the LightGBM model to the training data
        lgb_model.fit(X_train, y_train) #eval_set=eval_set

        # Make predictions on the test data
        y_pred = lgb_model.predict(X_test)

        # rmse.append( np.sqrt( mean_squared_error( y_test, y_pred )))
        
        feature_importances.append(lgb_model.feature_importances_)
        predicted[idx] = y_pred

    # Compute the accuracy metrics of the model using the predicted labels and true labels
    diff = actual - predicted
    squared_diff = diff ** 2
    mean_squared_error = squared_diff.mean()
    rmse = mean_squared_error ** 0.5

    for idx, x in enumerate(predicted):
        predicted[idx] = np.mean(x)

    #print results
    print()
    print( ("-" * 12), "LightGBM", ("-" * 12) )
    print("RMSE: ", rmse)

    mean_feature_importances = np.mean(feature_importances, axis=0)

    print("Feature Importance Values: ", *mean_feature_importances, sep=', ')
    
    rounded_actual = [round(num, 4) for num in actual]
    rounded_predicted = [round(num, 4) for num in predicted]
    
    print("Actual: ", *rounded_actual, sep=', ')
    print("Predct: ", *rounded_predicted, sep=', ')

    return mean_feature_importances, X, actual, predicted
    
def plots(data, params):

    mean_feature_importances, X, actual, predicted = lightGBMLOO(data, params)
    
    # Plots
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    

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
    patient = np.arange(len(actual))
    axs[1].scatter(patient, residuals)
    axs[1].set_xlabel("Patient")
    axs[1].set_ylabel("Actual-Pred / n")
    axs[1].set_title("Residual Plot")
    axs[1].axhline(y=0, color='blue', linestyle='-')

    # plt.tight_layout()
    plt.show()


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
    return predicted


def objectiveLOO(trial, filename):
    # Set hyperparameters to be tuned
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 2),
        'max_depth': trial.suggest_int('max_depth', -1, 30),
        'min_child_samples': trial.suggest_int('min_child_samples', 2, 20),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.0001, 1)
    } 

    data = pd.read_csv(filename)
    X, y = split(data)
    loo = LeaveOneOut()
    predicted = np.zeros_like(y)

    lgb_model = lgb.LGBMRegressor(**params)
                                      
    for idx, (train_idx, test_idx) in enumerate(loo.split(X)):        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        lgb_model.fit(X_train, y_train) #eval_set=eval_set
        predicted[idx] = lgb_model.predict(X_test)

    diff = y - predicted
    squared_diff = diff ** 2
    mean_squared_error = squared_diff.mean()
    rmse = mean_squared_error ** 0.5

    # print("MRMSE From OPTUNA: ", mrmse)
    return rmse

def lightGBMLOOOptuna(filename, n):
    # Set up Optuna study and run optimization
    study = optuna.create_study(direction='minimize')
    objective_fn = lambda trial: objectiveLOO(trial, filename)
    study.optimize(objective_fn, n_trials=n)

    print("Best Parameters: ", study.best_params)
    # print("Best Scorre:", study.best_value)
    return study.best_params
    # lightGBMLOO(data, study.best_params)


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
    plt.title("Actual vs Predicted LightGBM", fontsize=20)
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
    ax.set_title('LightGBM Static and Dynamic Feature Importance')

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
    ax.set_title('LighGBM Residuals')

    # add legend
    ax.legend()
    ax.axhline(y=0, color='black', alpha=0.5)
    plt.show()
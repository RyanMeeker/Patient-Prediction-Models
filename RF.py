import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
import optuna
import matplotlib.pyplot as plt
import pandas as pd

def split(data):
    # print(data.shape)
    X = data.iloc[:,1:-1]
    y = data.iloc[:,-1]
    return X, y

def RF(data):
    X, y = split(data)
    actual = y
    loo = LeaveOneOut()
    feature_importances = []
    predicted = np.zeros_like(y)

    rf = RandomForestRegressor(n_estimators=200, min_samples_split=4, random_state=0)

    print("Training Fold: ")
    for idx, (train_idx, test_idx) in enumerate(loo.split(X)):     
        print(idx, end=" ")
        # Get the training and testing data for the fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Fit the random forest regressor to the training data
        rf.fit(X_train, y_train)

        # Predict the labels for the test data
        y_pred = rf.predict(X_test)
        # rmse.append(root_mean_squared_error)
        # print("Feature importance: ", rf.feature_importances_)
        feature_importances.append(rf.feature_importances_)
        predicted[idx] = y_pred

    diff = actual - predicted
    squared_diff = diff ** 2
    mean_squared_error = squared_diff.mean()
    rmse = mean_squared_error ** 0.5
    # mrmse = np.mean(rmse)
    #print results
    print()
    print( ("-" * 16), "Random Forest", ("-" * 16) )
    print("Root Mean Squared Error: ", rmse)

    mean_feature_importances = np.mean(feature_importances, axis=0)
    
    print("Feature Importance Values: ", *mean_feature_importances, sep=', ')
    
    rounded_actual = [round(num, 4) for num in actual]
    rounded_predicted = [round(num, 4) for num in predicted]
    print("Actual: ", *rounded_actual, sep=', ')
    print("Predct: ", *rounded_predicted, sep=', ')

    # print("Print Feature Importances", feature_importances)
    # print("Print Mean Feature Importances", mean_feature_importances)

    return actual, predicted, mean_feature_importances, X


def plots(data):

    actual, predicted, mean_feature_importances, X = RF(data)

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
    patient = np.arange(len(y))
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
    plt.title("Actual vs Predicted Random Forest", fontsize=20)
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
    ax.set_title('Random Forest Static and Dynamic Feature Importance')

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
    ax.set_title('Random Forest Residuals')

    # add legend
    ax.legend()
    ax.axhline(y=0, color='black', alpha=0.5)
    plt.show()


# def objective(trial):
#     data = pd.read_csv('your_data.csv')
    
#     # Define the hyperparameters to optimize
#     n_estimators = trial.suggest_int('n_estimators', 10, 500)
#     max_depth = trial.suggest_int('max_depth', 2, 10)
#     min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    
#     # Split the data into independent and dependent variables
#     X, y = split(data)
    
#     # Initialize the random forest model with the optimized hyperparameters
#     rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, 
#                                 min_samples_split=min_samples_split, random_state=0)
    
#     # Initialize the Leave-One-Out cross-validation
#     loo = LeaveOneOut()
#     predicted = np.zeros_like(y)
#     actual = y
    
#     # Train and test the model with LOO cross-validation
#     for idx, (train_idx, test_idx) in enumerate(loo.split(X)):
#         X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
#         y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
#         rf.fit(X_train, y_train)
#         y_pred = rf.predict(X_test)
#         predicted[idx] = y_pred
    
#     # Calculate the root mean squared error
#     diff = actual - predicted
#     squared_diff = diff ** 2
#     mean_squared_error = squared_diff.mean()
#     rmse = mean_squared_error ** 0.5
    
#     return rmse

# Define the study
# study = optuna.create_study(direction='minimize')

# # Run the optimization process for a set number of trials
# n_trials = 100
# study.optimize(objective, n_trials=n_trials)

# # Print the best hyperparameters found
# best_params = study.best_params
# print('Best Hyperparameters:', best_params)
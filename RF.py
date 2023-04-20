import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
import optuna
import matplotlib.pyplot as plt

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
    print()
    print("All rmse: ", rmse)
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


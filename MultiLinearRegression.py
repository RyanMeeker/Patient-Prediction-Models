import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt


def split(data):
    X = data.iloc[:,1:-1]
    y = data.iloc[:,-1]
    return X, y

def MLR(data): 
    X, y = split(data)
    actual = y
    loo = LeaveOneOut()
    # rmse = []
    predicted = np.zeros_like(y)
    # array_FI = []
    reg = LinearRegression()
    cof_list = []                          
    print("Training Fold: ")
    for idx, (train_idx, test_idx) in enumerate(loo.split(X)):       
        print(idx, end=" ") 
        # Get the training and testing data for the fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        reg.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = reg.predict(X_test)

        # rmse.append( np.sqrt( mean_squared_error( y_test, y_pred )))
        coefficients = np.abs(reg.coef_)
        cof_list.append(coefficients / np.sum(coefficients))
        # feature_importances = pd.DataFrame({'feature': X.columns, 'importance': coefficients})
        # feature_importances = feature_importances.reindex(feature_importances['importance'].abs().sort_values(ascending=False).index)
        # array_FI.append(feature_importances)
        predicted[idx] = y_pred

    # Compute the accuracy metrics of the model using the predicted labels and true labels
    diff = actual - predicted
    squared_diff = diff ** 2
    mean_squared_error = squared_diff.mean()
    rmse = mean_squared_error ** 0.5

    avg_cof = np.mean(cof_list, axis=0)

    #print results
    print()
    print( ("-" * 9), "MultiLinearRegression", ("-" * 9) )
    print("RMSE: ", rmse)
    print("Feature Importance Values: ", *avg_cof, sep=', ')

    rounded_actual = [round(num, 4) for num in actual]
    rounded_predicted = [round(num, 4) for num in predicted]
    print("Actual: ", *rounded_actual, sep=', ')
    print("Predct: ", *rounded_predicted, sep=', ')


    return actual, predicted, avg_cof, X
    

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
    plt.title("Actual vs Predicted MultiLinearRegression", fontsize=20)
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
    ax.set_title('MultiLinearRegression Static and Dynamic Feature Importance')

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
    ax.set_title('MultiLinearRegression Residuals')

    # add legend
    ax.legend()
    ax.axhline(y=0, color='black', alpha=0.5)
    plt.show()

# def plot(data):
    
#     avg_cof, actual, predicted, X = MLR(data)

#     # Plot
#     fig, axs = plt.subplots(1, 1, figsize=(8, 5))
#     plt.bar(X.columns, avg_cof)
#     plt.xticks(rotation = 90)
#     plt.ylabel('Importance')
#     plt.xlabel('Feature')
#     plt.title('Feature Importance Plot')
#     plt.show()
#     # Feature Importances
#     # axs[0].bar(X.columns, mean_feature_importances)
#     # axs[0].set_xticks(range(len(X.columns)))
#     # axs[0].set_xticklabels(X.columns, rotation=90, ha='right')
#     # axs[0].set_ylabel("Importance")
#     # axs[0].set_xlabel("Feature")
#     # axs[0].set_title("Feature Importances(STILL WORKING ON THIS)")

#     # Residual Plot
#     # residuals = [a - p for a, p in zip(actual, predicted)]
#     # for idx, x in enumerate(residuals):
#     #     residuals[idx] = x / len(residuals) #np.std(residuals)

#     # # = residuals / len(residuals)    #np.std(residuals)
#     # patient = np.arange(len(y))
#     # plt.scatter(patient, residuals)
#     # plt.xlabel("Patient")
#     # plt.ylabel("Actual-Pred / n")
#     # plt.title("Residual Plot")
#     # plt.axhline(y=0, color='blue', linestyle='-')

#     # plt.tight_layout()
#     plt.show()
#     # # Actual vs Predicted
#     fig = plt.figure(figsize=(15, 5))
#     bar_width = 0.4
#     x = np.arange(len(actual))
#     plt.bar(x - 0.2, actual, width=bar_width, label='Actual', color='deepskyblue')
#     plt.bar(x + 0.2, predicted, width=bar_width, label='Predicted', color='steelblue')
#     plt.legend()
#     plt.xlabel("Patient")
#     plt.ylabel("Value")
#     plt.title("Actual vs Predicted")
    
#     plt.show()



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
    rmse = []
    predicted = np.zeros_like(y)
    array_FI = []
    reg = LinearRegression()
                                      
    
    for idx, (train_idx, test_idx) in enumerate(loo.split(X)):        
        # Get the training and testing data for the fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        reg.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = reg.predict(X_test)

        rmse.append( np.sqrt( mean_squared_error( y_test, y_pred )))
        coefficients = reg.coef_
        feature_importances = pd.DataFrame({'feature': X.columns, 'importance': coefficients})
        feature_importances = feature_importances.reindex(feature_importances['importance'].abs().sort_values(ascending=False).index)
        array_FI.append(feature_importances)
        predicted[idx] = y_pred

    # Compute the accuracy metrics of the model using the predicted labels and true labels
    mrmse = np.mean(rmse)
    
    #print results
    print( ("-" * 12), "LightGBM", ("-" * 12) )
    print("RMSE: ", mrmse)
    
    mean_feature_importances = np.mean(feature_importances, axis=0)

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
    axs[1].scatter(actual, residuals)
    axs[1].set_xlabel("Actual Values")
    axs[1].set_ylabel("Standardized Residuals")
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





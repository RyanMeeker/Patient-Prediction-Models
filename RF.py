import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
import numpy as np
import math

def openFile(filename):
    file = pd.read_csv(filename, header = 1)
    # print(file)
    return file

def split(data):
    # print(data.shape)
    X = data.iloc[:,1:-1]
    y = data.iloc[:,-1]
    # print(X)
    # print(y)
    return X, y

def randomForest(data):
    X, y = split(data)
    # print(X)
    # print(y)
    loo = LeaveOneOut()
    y_pred = []
    # model = RandomForestRegressor()
    # scores = cross_val_score(model, X, y, scoring = 'neg_mean_squared_error', cv=loo)
    # print(scores)
    # print(np.sqrt(np.mean(np.absolute(scores))))
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        y_pred.append(model.predict(X_test)[0])

    # for i, (train_index, test_index) in enumerate(loo.split(X)):
    #     rf = RandomForestRegressor()

    #     print("TRAIN:", train_index, "TEST:", test_index)
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index.tolist()]
    #     print(X_train, X_test, y_train, y_test)

    #     rf.fit(X_train, y_train)
    #     pred.append(rf.predict(X_test))
    #     print(pred)
    #     # print(f"Fold {i}:")
    #     # print(f"  Train: index={train_index}")
    #     # print(f"  Test:  index={test_index}")
    #     # print(type(train_index))

    return None

def RF(data):
    # Split data into X and y
    X, y = split(data)
    # Initialize the random forest regressor. Need to check if this goes in the for loop or not
    rf = RandomForestRegressor(n_estimators=1000)

    # Initialize the leave-one-out cross-validator
    loo = LeaveOneOut()

    # Initialize arrays to store the predicted labels and true labels for each fold
    y_pred, y_true = np.zeros_like(y)

    # Iterate over each loo
    for train_idx, test_idx in loo.split(X):
        # Get the training and testing data for the fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Fit the random forest regressor to the training data
        rf.fit(X_train, y_train)
        
        # Predict the labels for the test data
        y_pred[test_idx] = rf.predict(X_test)
        
        # Store the true labels for the test data
        y_true[test_idx] = y_test

    # Compute the accuracy metrics of the model using the predicted labels and true labels
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)

    #print results
    print("Y: ", y)
    print("Y_true: ", y_true)
    print("Y_predict: ", y_pred)

    print("MSE:", mse)
    print("RMSE: ", math.sqrt(mse))
    print("R-squared Score:", r2)
    print("Explained Variance Score:", evs)

if __name__ == '__main__':
    print("running...")
    # Aquire data
    data = openFile("selected_features.csv") # 18 patients, 10 x, 1 y 
    # Run RF. Split our data into train/test using LOO
    # randomForest(data)
    RF(data)
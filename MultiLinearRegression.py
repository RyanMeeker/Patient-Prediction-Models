import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import LeaveOneOut


def openFile(filename):
    file = pd.read_csv(filename)
    return file

def split(data):
    X = data.iloc[:,1:-1]
    y = data.iloc[:,-1]
    return X, y

def MLR(data): 
    X, y = split(data)
    loo = LeaveOneOut()
    rmse = []

    reg = LinearRegression()
                                      
    
    for idx, (train_idx, test_idx) in enumerate(loo.split(X)):        
        print( ("-" * 12), "Training Fold", idx, ("-" * 12) )
        # Get the training and testing data for the fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        reg.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = reg.predict(X_test)

        rmse.append( np.sqrt( mean_squared_error( y_test, y_pred )))

        print("Fold", idx, "finished with rmse: ", np.sqrt( mean_squared_error( y_test, y_pred )))

    # Compute the accuracy metrics of the model using the predicted labels and true labels
    mrmse = np.mean(rmse)
    
    #print results
    print( ("-" * 12), "LightGBM", ("-" * 12) )
    print("RMSE: ", mrmse)



if __name__ == '__main__':
    print("running...")
    # Aquire data
    # data = openFile("selected_features_stat.csv") # 18 patients, 10 x, 1 y 
    # data = openFile("selected_features.csv") # 18 patients, 10 x, 1 y 
    data = openFile("selected_features_dyn.csv") # 18 patients, 10 x, 1 y 

    MLR(data)

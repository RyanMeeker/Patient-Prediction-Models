import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.metrics import mean_squared_error
import lightgbm as lgb


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

def otherRF(data):
    X, y = split(data)

    #define cross-validation method to use
    loo = LeaveOneOut()

    #build multiple linear regression model
    model = RandomForestRegressor(n_estimators=500, random_state=0)

    #use LOOCV to evaluate model
    scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error',
                            cv=loo, n_jobs=-1)

    #view RMSE
    print( ("-" * 12), "Random Forest (Method 1)", ("-" * 12) )
    # print(scores)
    print("RMSE: ", np.sqrt(np.mean(np.absolute(scores))))

def RF(data):
    X, y = split(data)
    loo = LeaveOneOut()

    rmse = []
    # Initialize the random forest regressor
    rf = RandomForestRegressor(n_estimators=200, min_samples_split=4, random_state=0)

    for idx, (train_idx, test_idx) in enumerate(loo.split(X)):        
        print(("-" * 12),"Training fold ",idx, (12 * "-"))
        
        # Get the training and testing data for the fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Fit the random forest regressor to the training data
        rf.fit(X_train, y_train)
        
        # Predict the labels for the test data
        y_pred = rf.predict(X_test)

        rmse.append( np.sqrt( mean_squared_error( y_test, y_pred )))
        print("Fold", idx, " finished with rmse: ", np.sqrt( mean_squared_error( y_test, y_pred )))

    mrmse = np.mean(rmse)

    #print results
    print( ("-" * 12), "Random Forest", ("-" * 12) )
    # print("Y_true: ", y_true)
    # print("Y_predict: ", y_pred)
    # print("RMSE:", rmse)
    print("RMSE: ", mrmse)

def lightGBM(data): 
    X, y = split(data)
    loo = LeaveOneOut()
    rmse = []
    lgb_model = lgb.LGBMRegressor(num_leaves = 30, learning_rate= 0.3, n_estimators = 100, random_state=0, min_child_samples=4)
                                      
    
    for idx, (train_idx, test_idx) in enumerate(loo.split(X)):        
        print( ("-" * 12), "Training Fold", idx, ("-" * 12) )
        # Get the training and testing data for the fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # eval_set = [(X_test, y_test)]

        # Fit the LightGBM model to the training data
        lgb_model.fit(X_train, y_train) #eval_set=eval_set

        # Make predictions on the test data
        y_pred = lgb_model.predict(X_test, raw_score = True, num_iteration = -1)
        # print(y_pred[0])
        rmse.append( np.sqrt( mean_squared_error( y_test, y_pred )))

        print("Fold", idx, "finished with rmse: ", np.sqrt( mean_squared_error( y_test, y_pred )))

    # Compute the accuracy metrics of the model using the predicted labels and true labels
    mrmse = np.mean(rmse)

    #print results
    print( ("-" * 12), "LightGBM", ("-" * 12) )
    # print("Y_true: ", y_true)
    # print("Y_predict: ", y_pred)
    print("RMSE: ", mrmse)

def lightGBM_cross_validation(data):
    X, y = split(data)
    
    # Define hyperparameter search space
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
    }
    
    # Set up LightGBM dataset
    lgb_data = lgb.Dataset(X, y)
    
    # Run hyperparameter search using LightGBM's built-in cross-validation tool
    cv_results = lgb.cv(params=params,
                        train_set=lgb_data,
                        num_boost_round=200,
                        nfold=5,
                        stratified=False)
    
    # Get the best hyperparameters
    best_params = cv_results['params']
    
    # Train final LightGBM model using the best hyperparameters
    lgb_model = lgb.train(params=best_params,
                          train_set=lgb_data,
                          num_boost_round=cv_results['best_iteration'])
    
    # Make predictions on the test data
    y_pred = lgb_model.predict(X)
    
    # Compute the accuracy metrics of the model using the predicted labels and true labels
    mrmse = np.sqrt(mean_squared_error(y, y_pred))
    
    # Print results
    print( ("-" * 12), "Light_GBM_CrossValidation", ("-" * 12) )
    print("MRMSE: ", mrmse)


if __name__ == '__main__':
    print("running...")
    # Aquire data
    data = openFile("selected_features.csv") # 18 patients, 10 x, 1 y 
    # otherRF(data)
    RF(data) 
    lightGBM(data)

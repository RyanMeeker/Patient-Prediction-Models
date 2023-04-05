import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import matplotlib.pyplot as plt

def openFile(filename):
    file = pd.read_csv(filename)
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
    lgb_model = lgb.LGBMRegressor(boosting_type='gbdt', num_leaves = 30, learning_rate= 0.025, n_estimators = 125, random_state=0, min_child_samples=5)
    toPlot = []     
    test = []     

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
        toPlot.append(y_pred)
        test.append(y_test)
        rmse.append( np.sqrt( mean_squared_error( y_test, y_pred )))
        # for j in range(len(y_test)):
        #     toPlot[j] = ( toPlot[j] + (y_test - y_pred) )
        # # print(toPlot)

        print("Fold", idx, "finished with rmse: ", np.sqrt( mean_squared_error( y_test, y_pred )))
    
    # print("y= ", y)
    y_plot = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
    print(toPlot)
    for i in range(len(toPlot)):
        toPlot[i] = (test[i] - toPlot[i])

    # print("x = ", toPlot)
    plt.scatter(y_plot, toPlot)
    plt.show()

    # Compute the accuracy metrics of the model using the predicted labels and true labels
    mrmse = np.mean(rmse)
    # fig = plt.figure(figsize=(14,8))
    
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

def plot(patient, y_pred):


    # Create scatterplots for Feature1 to Feature3
    # for i, col in enumerate(['SUVvar_dyn_rel_wb_PL_min', 'SUVmax_dyn_abs_spine_allL_max', 'SUVmax_dyn_abs_wb_allL_var']):
    #     ax = axes[i][0]
    #     ax.scatter(data[col], data['pfs'])
    #     ax.set_xlabel(col)
    #     ax.set_ylabel('pfs')

    plt.show()

if __name__ == '__main__':
    print("running...")
    # Aquire data
    data = openFile("selected_features.csv") # 18 patients, 10 x, 1 y 
    # otherRF(data)
    # RF(data) 
    lightGBM(data)
    # plot(data)

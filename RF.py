 import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score, train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import optuna


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
    print( ("-" * 16), "Random Forest", ("-" * 16) )
    # print("Y_true: ", y_true)
    # print("Y_predict: ", y_pred)
    # print("RMSE:", rmse)
    print("RMSE: ", mrmse)

def lightGBMLOO(data, params): 
    X, y = split(data)
    loo = LeaveOneOut()
    rmse = []
    lgb_model = lgb.LGBMRegressor(**params)
                                      
    
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


def objectiveCV(trial):
    # Set hyperparameters to be tuned
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 2),
        'max_depth': trial.suggest_int('max_depth', -1, 30),
        'min_child_samples': trial.suggest_int('min_child_samples', 2, 10),
    }
    
    # Set up LightGBM model with hyperparameters
    model = lgb.LGBMRegressor(**params)
    
    # Split data into features and target variables
    X, y = split(data)
    
    # Perform cross-validation and return mean RMSE
    score = np.sqrt(-cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error'))
    return score.mean()

def LightGBMCrossValidation(data):
    # Set up Optuna study and run optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objectiveCV, n_trials=100)
    
    # Print best hyperparameters and score
    print("Best parameters:", study.best_params)
    print("Best score:", study.best_value)
    
    # Set up LightGBM model with best hyperparameters
    best_params = study.best_params
    best_model = lgb.LGBMRegressor(**best_params)
    
    # Split data into features and target variables
    X, y = split(data)

    best_model.fit(X, y)

    # Compute the mean squared error of the final model
    y_pred = best_model.predict(X)
    mse = np.sqrt(mean_squared_error(y, y_pred))
    print("RMSE: ", mse)
    # Perform cross-validation and print scores
    # scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    # print("Cross-validation scores:", scores)
    # print("Mean score:", scores.mean())

def objectiveLOO(trial):
    # Set hyperparameters to be tuned
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 2),
        'max_depth': trial.suggest_int('max_depth', -1, 30),
        'min_child_samples': trial.suggest_int('min_child_samples', 2, 10),
    } 

    data = pd.read_csv("selected_features.csv")
    X, y = split(data)
    loo = LeaveOneOut()
    rmse = []
    lgb_model = lgb.LGBMRegressor(**params)
                                      
    for idx, (train_idx, test_idx) in enumerate(loo.split(X)):        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        lgb_model.fit(X_train, y_train) #eval_set=eval_set
        y_pred = lgb_model.predict(X_test, raw_score = True, num_iteration = -1)
        rmse.append( np.sqrt( mean_squared_error( y_test, y_pred )))
    mrmse = np.mean(rmse)
    return mrmse

def lightGBMLOOOptuna(data):
    # Set up Optuna study and run optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objectiveLOO, n_trials=100)
    
    # Print best hyperparameters and score
    print("Best parameters:", study.best_params)
    print("Best score:", study.best_value)
    
    # Set up LightGBM model with best hyperparameters
    best_params = study.best_params
    lightGBMLOO(data, best_params)


if __name__ == '__main__':
    print("running...")
    # Aquire data
    data = openFile("selected_features.csv") # 18 patients, 10 x, 1 y 
    # otherRF(data)
    # RF(data) 
    # lightGBMLOO(data)
    # lightGBMLOOOptuna(data)
    LightGBMCrossValidation(data)

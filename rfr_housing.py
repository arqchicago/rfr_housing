import pandas as pd
import numpy as np
import sklearn.model_selection as skms
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.compose import ColumnTransformer


def rmse(y_true, y_pred, weight=0):
    # returns root mean squared error
    # if weight=0:  rmse = sqrt(sum(y_true-y_pred)^2)
    # if weighted:  rmse = sqrt(sum(weight*(y_true-y_pred)^2)/sum(weight))

    if weight==0:
        rmse = (np.mean((y_true.values-y_pred)**2))**0.5
    else:
        weight = weight.loc[y_true.index.values]
        rmse = (np.sum(weight*(y_true.values-y_pred)**2)/np.sum(weight))**0.5
    return rmse


def rmae(y_true, y_pred, weight=0):
    # returns root mean absolute error
    # if weight=0:  rmae = sqrt(sum(abs(y_true-y_pred)))
    # if weighted:  rmse = sqrt(sum(weight*(abs(y_true-y_pred)))/sum(weight))
    
    if weight==0:
        rmae = (np.mean(np.absolute(y_true.values-y_pred)))**0.5
    else:
        weight = weight.loc[y_true.index.values]
        rmae = (np.sum(weight*np.absolute(y_true.values-y_pred))/np.sum(weight))
    return rmae



if __name__ == "__main__":

    #----  features and target variable
    features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'grade', 'age', 'appliance_age', \
                'crime', 'backyard', 'view', 'condition', 'renovated'] 
    quantitative_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'age', 'appliance_age']
    target_var = 'price'
    sqft = 'sqft_living'


    #----  uploading data
    heart_df = pd.read_csv('data\\housing.csv')
    rows, cols = heart_df.shape
    print(f'> rows = {rows},  cols = {cols}')


    #----  train/test split
    X, y = heart_df[features], heart_df[target_var]
    X_train, X_test, y_train, y_test = skms.train_test_split(X, y, test_size=0.20)

    X_train_rows, y_train_rows = X_train.shape[0], y_train.shape[0]
    X_test_rows, y_test_rows = X_test.shape[0], y_test.shape[0]
    train_rows, test_rows = -1, -1

    if X_train_rows == y_train_rows:
        train_rows = X_train_rows

    if X_test_rows == y_test_rows:
        test_rows = X_test_rows
        
      
    X_train_sqft = heart_df[sqft].loc[X_train.index.values]
    X_train_target_var = heart_df[target_var].loc[X_train.index.values]

    print(f'> features = {len(features)}')
    print(f'> training set = {train_rows} ({round(train_rows*1.0/rows,3)})')
    print(f'> testing set = {test_rows} ({round(test_rows*1.0/rows,3)}) \n')


    #----  weighted scorer functions 
    wgt_rmse_scorer = make_scorer(rmse, greater_is_better=False)
    wgt_rmae_scorer = make_scorer(rmae, greater_is_better=False)

    scorers = {'rmse': wgt_rmse_scorer, 'rmae': wgt_rmae_scorer}


    #----  random forest training with hyperparameter tuning

    best_model = {}
    rfr_grid = {"rfr__n_estimators": [100, 500, 1000],
                "rfr__max_depth": [10, 20, 30],
                "rfr__max_features": [0.25, 0.50, 0.75],
                "rfr__min_samples_split": [5, 10, 20],
                "rfr__min_samples_leaf": [3, 5, 10],
                "rfr__bootstrap": [True, False]}

    models = {'rfr': [RandomForestRegressor(), rfr_grid]}
    transformations = { 'StandardScaler': StandardScaler(), 
                        'RobustScaler': RobustScaler()}

    for key, value in transformations.items():

        transformation = Pipeline(steps=[(key, value)])
        col_transformations = ColumnTransformer(transformers=[('quant', transformation, quantitative_features)])
        pipe = Pipeline([(key, col_transformations), ('rfr', RandomForestRegressor())])

        optimized_rfr = skms.RandomizedSearchCV(pipe, 
                                                param_distributions=rfr_grid, 
                                                n_iter = 50, 
                                                cv = 5, 
                                                verbose = 10, 
                                                scoring = scorers,
                                                refit = 'rmse',
                                                random_state = 42, 
                                                n_jobs = -1)

        optimized_rfr.fit(X_train, y_train)
        print('\n')

        #----  predicting on the training and testing set
        y_train_pred = optimized_rfr.predict(X_train)
        rmse_train = round(rmse(y_train, y_train_pred), 0)
        rmae_train = round(rmae(y_train, y_train_pred), 0)

        y_test_pred = optimized_rfr.predict(X_test)
        rmse_test = round(rmse(y_test, y_test_pred), 0)
        rmae_test = round(rmae(y_test, y_test_pred), 0)


        print(f'> evaluation metrics \n')
        print(f'> random forest, {key}')
        print('%-10s %20s %10s' % ('metric','training','testing'))
        print('%-10s %20s %10s' % ('rmse', rmse_train, rmse_test))
        print('%-10s %20s %10s' % ('rmae', rmae_train, rmae_test))
        print('\n')

        best_model[key] = {}
        best_model[key]['train'] = rmse_train
        best_model[key]['test'] = rmse_test
        
        #----  obtaining results of the grid run
        cv_results = optimized_rfr.cv_results_
        cv_results_df = pd.DataFrame(cv_results)

        print('> hyperparameter tuning results')
        print(cv_results_df)

        #----  getting best parameters
        best_params = optimized_rfr.best_params_
        best_score = optimized_rfr.best_score_

        print(f'> best hyperparameters = {best_params}')

        #----  saving model results
        cv_results_df.to_csv('output\\rfr_'+key+'__cv_results.csv', index=False)
        best_params_str = ', '.join('{}={}'.format(key, val) for key, val in best_params.items())

        with open('output//rfr_'+key+'__results.txt', 'w') as file:
            file.write('best parameters = '+best_params_str+'\n')
            file.write('rmse:  '+'(train='+str(rmse_train)+')  (test='+str(rmse_test)+')'+'\n')
            file.write('rmae:  '+'(train='+str(rmae_train)+')  (test='+str(rmae_test)+')'+'\n')
            

    print(f'> evaluation metrics \n')
    print('%-40s %-10s %20s %10s' % ('transformation', 'metric', 'training', 'testing'))

    for key, value in best_model.items():
        
        transformation_type = 'random forest ' + key
        print('%-40s %-10s %20s %10s' % (transformation_type, 'rmse', value['train'], value['test']))
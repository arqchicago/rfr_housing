import pandas as pd
import numpy as np
import sklearn.model_selection as skms
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def rmse(y_true, y_pred, data_df, weight_var_name=''):
    # returns root mean squared error
    # if weight=0:  rmse = sqrt(sum((y_true-y_pred)^2))
    # if weighted:  rmse = sqrt(sum(weight*(y_true-y_pred)^2)/sum(weight))

    if weight_var_name == '':
        df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
        df['weight'] = 1
        
    else:
        weight = data_df[weight_var_name].loc[y_true.index.values]
        df = pd.DataFrame({'weight': weight, 'y_true': y_true, 'y_pred': y_pred})

    rmse_ = np.sqrt(np.sum(df['weight']*(df['y_true']-df['y_pred'])**2)/np.sum(df['weight']))
    
    return rmse_


def rmae(y_true, y_pred, data_df, weight_var_name=''):
    # returns root mean absolute error
    # if weight=0:  rmae = sqrt(sum(abs(y_true-y_pred)))
    # if weighted:  rmse = sqrt(sum(weight*(abs(y_true-y_pred)))/sum(weight))      
        
    if weight_var_name == '':
        df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
        df['weight'] = 1
        
    else:
        weight = data_df[weight_var_name].loc[y_true.index.values]
        df = pd.DataFrame({'weight': weight, 'y_true': y_true, 'y_pred': y_pred})
    
    #if weight==0:
    #    rmae = (np.sum(weight*np.absolute(y_true.values-y_pred)))**0.5
    #else:
    #    rmae_ = (np.sum(weight*np.absolute(y_true.values-y_pred))/np.sum(weight))
    rmae_ = np.sqrt(np.sum(df['weight']*np.absolute(df['y_true']-df['y_pred']))/np.sum(df['weight']))
    
    return rmae_


def get_rmse_pctl(y_true, y_pred, weight, var, var_dict):
    # weighted rmse by percentiles of a variable of interest
    # -- returns dict {pct: rmse}
    
    weight = weight.loc[y_true.index.values]
    var = var.loc[y_true.index.values]

    df = pd.DataFrame({'var': var, 'weight': weight, 'y_true': y_true, 'y_pred': y_pred})
    dict_ = {}
    for pct, threshold in var_dict.items():
        if pct<100:
            df_temp = df[df['var']<threshold]
            pct_rmse = np.sqrt(((df_temp['weight']*(df_temp['y_true']-df_temp['y_pred'])**2).sum())/df_temp['weight'].sum())
            dict_[pct] = pct_rmse
    return dict_




if __name__ == '__main__':

    seed = 5941
    #----  features and target variable
    quant_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'grade', 'age', 'appliance_age', 'crime', 'renovated'] 
    cat_features = ['backyard', 'view', 'condition']    
    features = quant_features

    target_var = 'price'
    weight = 'weight'
    sqft = 'sqft_living'

    #----  uploading data
    heart_df = pd.read_csv('data\\housing.csv')
    rows, cols = heart_df.shape
    print(f'> rows = {rows},  cols = {cols}')

    #---- get weighted percentiles 
    #sqft_pctl_dict = wgt_percentile(heart_df, weight, sqft)
    #target_pctl_dict = wgt_percentile(heart_df, weight, target_var)


    #----  train/test split
    X, y = heart_df[features], heart_df[target_var]
    X_train, X_test, y_train, y_test = skms.train_test_split(X, y, test_size=0.20, random_state = seed)

    X_train_rows, y_train_rows = X_train.shape[0], y_train.shape[0]
    X_test_rows, y_test_rows = X_test.shape[0], y_test.shape[0]
    train_rows, test_rows = -1, -1

    if X_train_rows == y_train_rows:
        train_rows = X_train_rows

    if X_test_rows == y_test_rows:
        test_rows = X_test_rows
        
      
    X_train_weights = heart_df[weight].loc[X_train.index.values]
    X_test_weights = heart_df[weight].loc[X_test.index.values]
    #X_train_sqft = heart_df[sqft].loc[X_train.index.values]
    #X_train_target_var = heart_df[target_var].loc[X_train.index.values]

    params_score = {"data_df": heart_df}

    print(f'> features = {len(features)}')
    print(f'> training set = {train_rows} ({round(train_rows*1.0/rows,3)})')
    print(f'> testing set = {test_rows} ({round(test_rows*1.0/rows,3)}) \n')


    #----  weighted scorer functions 
    wgt_rmse_scorer = make_scorer(rmse, greater_is_better=False, **params_score, weight_var_name=weight)
    wgt_rmae_scorer = make_scorer(rmae, greater_is_better=False, **params_score, weight_var_name=weight)

    scorers = {'rmse': wgt_rmse_scorer, 'rmae': wgt_rmae_scorer}

    #----  random forest training with hyperparameter tuning
    pipe = Pipeline([("scaler", StandardScaler()), ("rfr", RandomForestRegressor())])

    random_grid = { "rfr__n_estimators": [100, 500, 1000],
                    "rfr__max_depth": [10, 20, 30],
                    "rfr__max_features": [0.25, 0.50, 0.75],
                    "rfr__min_samples_split": [15, 25],
                    "rfr__min_samples_leaf": [5, 10, 15],
                    "rfr__bootstrap": [True, False]
                   }

    optimized_rfr = skms.RandomizedSearchCV(pipe, 
                                            param_distributions=random_grid, 
                                            n_iter = 25, 
                                            cv = 5, 
                                            verbose = 10, 
                                            scoring = scorers,
                                            refit = 'rmse',
                                            random_state = seed, 
                                            n_jobs = 4)

    optimized_rfr.fit(X_train, y_train, **{'rfr__sample_weight': X_train_weights.values.ravel()})
    print('\n')

    #----  predicting on the training and testing set
    y_train_pred = optimized_rfr.predict(X_train)
    rmse_train = round(rmse(y_train, y_train_pred, heart_df, weight), 0)
    rmae_train = round(rmae(y_train, y_train_pred, heart_df, weight), 0)
    r2_train = round(r2_score(y_train, y_train_pred, X_train_weights), 4)

    y_test_pred = optimized_rfr.predict(X_test)
    rmse_test = round(rmse(y_test, y_test_pred, heart_df, weight), 0)
    rmae_test = round(rmae(y_test, y_test_pred, heart_df, weight), 0)
    r2_test = round(r2_score(y_test, y_test_pred, X_test_weights), 4)


    print('> evaluation metrics \n')
    print('%-10s %20s %10s' % ('metric','training','testing'))
    print('%-10s %20s %10s' % ('rmse', rmse_train, rmse_test))
    print('%-10s %20s %10s' % ('rmae', rmae_train, rmae_test))
    print('%-10s %20s %10s' % ('r2', r2_train, r2_test))
    print('\n')


    #----  obtaining results of the grid run
    cv_results = optimized_rfr.cv_results_
    cv_results_df = pd.DataFrame(cv_results)

    print('> hyperparameter tuning results')
    print(cv_results_df)

    #----  getting best parameters
    best_params = optimized_rfr.best_params_
    best_score = optimized_rfr.best_score_

    print(f'> best hyperparameters = {best_params}')
    print(f'> best cv score = {best_score} \n')

    #----  saving model results
    cv_results_df.to_csv('output\\cv_results.csv', index=False)

    best_params_str = ', '.join('{}={}'.format(key, val) for key, val in best_params.items())

    with open('output//rfr_results.txt', 'w') as file:
        file.write('best parameters = '+best_params_str+'\n')
        file.write('rmse:  '+'(train='+str(rmse_train)+')  (test='+str(rmse_test)+')'+'\n')
        file.write('rmae:  '+'(train='+str(rmae_train)+')  (test='+str(rmae_test)+')'+'\n')

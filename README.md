# Random Forest Regression on House Price dataset
This project demonstrates hyperparameter optimization of Random Forest model using grid search and 5-fold cross validation. Scikit-Learn pipeline is used
to sequentially apply important feature transformation. The dataset is a variation of the House Sales dataset in King County, Washington and is obtained from 
Kaggle website (Public Domain license). The target variable represents sale price of houses. The features measured in the dataset are age of the house, square 
feet area of the living area, square feet area of the lot, number of bedrooms, number of bathrooms, age of appliances, crime rate in the area, number of years 
since last major renovation and grade condition of the home.  

## Blog 
My blog on this project can be accessed at https://rfrhousing1.blogspot.com/2021/05/random-forest-model-with-feature.html

## Cross Validation
5-fold cross validation is used to avoid overfitting and to collect model evaluation metrics. In 5-fold cross validation, the training set is split
into 5 groups. In each iteration, one group is used as a hold-out set and the model is trained on the remaining groups. Evaluation metrics are collected
and the process is repeated. Overall performance of the model is evaluated based on the metrics. 

## Hyperparameter Optimization
This Random Forest model includes hyperparameter optimization. In this optimization procedure, a grid search on a set of hyperparameters is performed in 
order to find model settings that achieve the best performance on a given dataset. It is important to note that 5-fold cross validation is used so that
the performance is evaluated on an independent hold-out test set. 

## Scikit Learn Pipeline
The Scikit Learn Pipeline is used to sequentially apply transformations on the features. This pipeline is capable of putting together several steps of
transformations that can be cross-validated together.

## Model Results and Feature Importance
Once the best model settings are picked, evaluation metrics are obtained for the training and testing set. This includes root mean squared error, root
mean absolutely error and coefficient of determination. Additionally, numeric values of feature importances are also collected to highlight contribution
of each feature to the model.
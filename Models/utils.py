import os
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import max_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV


def hyperparameterTuning_RandomForest(x, y):
    model_rand = RandomForestRegressor()
    n_estimators = [5, 20, 50, 100]  # number of trees in the random forest
    # number of features in consideration at every split
    max_features = ['auto', 'sqrt']
    # maximum number of levels allowed in each decision tree
    max_depth = [int(x) for x in np.linspace(10, 120, num=12)]
    min_samples_split = [2, 6, 10]  # minimum sample number to split a node
    # minimum sample number that can be stored in a leaf node
    min_samples_leaf = [1, 3, 4]
    bootstrap = [True, False]  # method used to sample data points
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    rf_random = RandomizedSearchCV(estimator=model_rand, param_distributions=random_grid,
                                   n_iter=100, cv=5, verbose=2, random_state=35, n_jobs=-1)
    rf_random.fit(x, y)
    return rf_random.best_params_

'''
importance: (ndarray)
names: (pandas.core.indexes.numeric.Int64Index) - has to be converted into list
'''
def plot_feature_importance(feature_importance, names, model_type):


    # # Create arrays from feature importance and feature names
    # feature_importance = np.array(importance)
    # feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {'feature_names': names,
            'feature_importance': list(feature_importance)}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values('feature_importance', inplace = True)
    # fi_df.sort_values(by=['feature_importance'], ascending=True, inplace=True)

    # Define size of bar plot
    plt.figure(figsize=(20, 20))
    # Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_names'], y=fi_df['feature_importance'])
    # Add chart labels
    plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE NAMES')
    plt.xticks(rotation=90)
    plt.ylabel('FEATURE IMPORTANCE')


def featureScore(importance):
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    for i, v in enumerate(feature_importance):
        print('Feature: %0d, Score: %.5f' % (i, v))



def folderPath(folderName):
    print("1")
    Figurefolder = os.path.join(os.getcwd(), folderName)
    return Figurefolder


def metrics(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    R2 = r2_score(actual, predicted)
    MAE = mean_absolute_error(actual, predicted)
    MSE = mean_squared_error(actual, predicted)
    RMSLE = np.log(np.sqrt(mean_squared_error(actual, predicted)))
    max_error_ = max_error(actual, predicted)
    return rmse,R2,MAE,MSE,RMSLE,max_error_


def hyperparameterTuning_XGBoost(x, y):
    model = xgb.XGBRegressor()
    parameters = {'nthread': [4],
                  'objective': ['reg:squarederror'],
                  # so called 'eta' value
                  'learning_rate': [.03, 0.05, .07],
                  'max_depth': [5, 6, 7],
                  'min_child_weight': [4],
                  'subsample': [0.7],
                  'colsample_bytree': [0.7],
                  'n_estimators': [500]}
    model_grid = GridSearchCV(model, parameters, cv=2, n_jobs=5, verbose=True)
    model_grid.fit(x, y)
    model_grid.best_score_
    model_grid.best_params_
    bestScore = model_grid.best_score_
    bestParams = model_grid.best_params_
    return bestScore, bestParams


def hyperparameterTuning_MLP(X, Y):
    estimator = MLPRegressor()
    param_grid = {'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100, 1)],
                    'activation': ['relu', 'tanh', 'logistic'],
                    'alpha': [0.0001, 0.05],
                    'learning_rate': ['constant', 'adaptive'],
                    'solver': ['adam']}
    gsc = GridSearchCV(estimator, param_grid, cv=5,
                        scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
    grid_result = gsc.fit(X, Y)
    bestScore = grid_result.best_score_
    bestParams = grid_result.best_params_
    return bestScore, bestParams


def normalizing(X):
    normalizer = preprocessing.MinMaxScaler()
    normalized = pd.DataFrame(normalizer.fit_transform(X),  columns = X.columns)
    return normalized



def removeColumnContainString(df):
    cols_to_remove = []
    for col in df.columns:
        try:
            _ = df[col].astype(float)
        except ValueError:
            print('Couldn\'t covert %s to float' % col)
            cols_to_remove.append(col)
            pass
    # keep only the columns in df that do not contain string
    df = df[[col for col in df.columns if col not in cols_to_remove]]
    return df



def Feature_Extraction(dataframe,n,featureExtraction):
    vectorizer = featureExtraction(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None,max_features=100)
    bow_words = vectorizer.fit_transform(dataframe)
    bow_clean = bow_words.toarray()
    vocab = vectorizer.get_feature_names()
    # print(vocab)
    new_vocab = []
    for word in vocab:
        if len(word)>n:
            new_vocab.append(word)
    print (' '.join(new_vocab))
    words_clean = pd.DataFrame(data=bow_clean, columns=vocab)
    # words_clean.head()
    j1 = words_clean.columns.get_level_values(0).isin(new_vocab)
    words_clean = words_clean.loc[:,j1]
    # words_clean.head()
    return words_clean



def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)



import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline

from lightgbm.sklearn import LGBMRegressor


class CrossValidator:
    
    def __init__(self, n_splits: int, test_size: int):
        
        self.n_splits = n_splits
        self.test_size = test_size
    
    def grid_search(self, X, y, train_df, sample_size: int=None) -> pd.DataFrame:
        
        if sample_size:
            X = X.sample(sample_size)
        
        # Source: https://stackoverflow.com/questions/38555650/try-multiple-estimator-in-one-grid-search    
        class DummyEstimator(BaseEstimator):
            def fit(self): pass
            def score(self): pass

        # Placeholder Estimator
        pipe = Pipeline(
            [('estimator', DummyEstimator())]
        ) 

        # Candidate learning algorithms and their hyperparameters
        param_grid = [
            {
                'estimator': [LinearRegression(normalize=True)],
                'estimator__fit_intercept': [True, False]
            },
            {
                'estimator': [LGBMRegressor()],
                'estimator__n_estimators': [100, 300]
            }
        ]
        
        gscv = GridSearchCV(
            estimator=pipe, 
            param_grid=param_grid, 
            cv=self.create_splits(train_df),
            scoring=make_scorer(self.rmsse, greater_is_better=False),
            n_jobs=-1)

        gscv.fit(X=X, y=y)

        return (pd.DataFrame(gscv.cv_results_)
                .sort_values('rank_test_score')
               [['params', 'mean_test_score', 'rank_test_score']])
    
    def create_splits(self, df: pd.DataFrame):

        dates = df['d'].drop_duplicates()

        counter=0
        for i in range(len(dates)-self.test_size, 0, -self.test_size):
            if counter == self.n_splits:
                break
            counter +=1

            train_dates = dates.iloc[range(0, i)].values
            test_dates = dates.iloc[range(i, i+self.test_size)].values

            yield (df[df['d'].apply(lambda x: x in train_dates)].index, 
                   df[df['d'].apply(lambda x: x in test_dates)].index)
            
    # Source: https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9
    def rmsse(self, actual: np.ndarray, predicted: np.ndarray, seasonality: int = 1):
        """ Root Mean Squared Scaled Error """
        actual = np.array(actual)
        predicted = np.array(predicted)
        q = self._mse(actual, predicted) / self._mse(actual[seasonality:], self._naive_forecasting(actual, seasonality))
        return np.sqrt(q)

    def _error(self, actual: np.ndarray, predicted: np.ndarray):
        """ Simple error """
        return actual - predicted

    def _mse(self, actual: np.ndarray, predicted: np.ndarray):
        """ Mean Squared Error """
        return np.mean(np.square(self._error(actual, predicted)))

    def _naive_forecasting(self, actual: np.ndarray, seasonality: int = 1):
        """ Naive forecasting method which just repeats previous samples """
        return actual[:-seasonality] 
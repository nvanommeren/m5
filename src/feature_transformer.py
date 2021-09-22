import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

class FeatureTransformer(TransformerMixin):

    def __init__(self, shift_days: int):
        """Creates featurs for the model. This class can easily be
        extended to include more features.

        Args:
            shift_days (int): The number of days to shift the timeseries
            feautures to avoid leakage. For example if we make a 3 step
            ahead forecast, we cannot use the rolling mean based on 2
            days ago.
        """
        self.shift_days = shift_days

        self.categorical_features = [
            'dept_id', 'cat_id', 'store_id', 'state_id',
            'event_type_1', 'month', 'wday',
            'is_weekend', 'snap'
        ]


    def fit(self, X: pd.DataFrame):
        """Converts to categorical columns to OneHot encoded columns.

        Args:
            X (pd.DataFrame): Dataframe with the features.

        Returns:
            [ColumnTransformer]: Fitted feature transformer.
        """
        self.numeric_features = [
            x for x in X.columns
            if 'lag_' in x
            or 'rolling_' in x
            or 'price' in x
        ]

        self.feature_transformer = ColumnTransformer(
             [('numeric', 'passthrough', self.numeric_features),
              ('categorical', OneHotEncoder(sparse=False, drop='first'),
               self.categorical_features)]
        )

        self.feature_transformer.fit(X)

        return self.feature_transformer

    def transform(self, X: pd.DataFrame):
        """Encodes the categorical columns in the dataframe.

        Args:
            X ([pd.DataFrame]): Sales data.

        Returns:
            [np.array]: Matrix with the encoded features.
        """
        return self.feature_transformer.transform(X)

    def calculate_timebase_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculates time based (lagged and rolling mean) features
        of the sales data.

        Args:
            X (pd.DataFrame): Dataframe with the sales data.

        Returns:
            [pd.DataFrame]: Dataframe with the sales data including lag and rolling
            features.
        """
        X = self._add_lagged_features(X, [1, 3, 7, 14, 21, 365])

        X = self._add_rolling(X, 'mean', [5, 50])
        X = self._add_rolling(X, 'min', [5, 50])
        X = self._add_rolling(X, 'max', [5, 50])

        return X

    def _add_lagged_features(self, X: pd.DataFrame, lags: list) -> pd.DataFrame:
        """Adds lagged features of the sales data, shifted by the number
        of shift_days to avoid leakage.

        Args:
            X (pd.DataFrame): Dataframe with the sales data.
            lags (list): List of integers with the lags to include.

        Returns:
            pd.DataFrame: Dataframe including the lagged sales.
        """
        for l in lags:
            X[f'sales_lag_{l + self.shift_days}'] = (X[['id', 'sales', 'd']]
                    .groupby('id')['sales']
                    .transform(lambda x: x.shift(l  + self.shift_days))
                    .fillna(0))
        return X

    def _add_rolling(self, X: pd.DataFrame, aggregate: str, days: list) -> pd.DataFrame:
        """Adds rolling features of the sales data, shifted by the number of
        shift_days to avoid leakage.

        Args:
            X (pd.DataFrame): Dataframe with the sales data.
            aggregate (str): Aggregate to take for creating rolling features, this
            can be 'mean', 'max', 'min', ..
            days (list): List of integers with the days to include for calculating
            the aggregate.

        Returns:
            pd.DataFrame: Dataframe including the rolling aggregate sales.
        """
        for d in days:
            X[f'rolling_{aggregate}_{d}'] = (X[['id', 'sales', 'd']]
                        .groupby(['id'])['sales']
                        .transform(lambda x: x
                                   .shift(self.shift_days + 1)
                                   .rolling(d).agg(aggregate))
                        .fillna(0))
        return X

    def save_data(self, X, y, filename: str='train'):

        with open(f'{filename}.npy', 'wb') as f:
            np.save(f, X)
            np.save(f, np.array(y))

    def load_data(self, filename: str='train'):

        with open(f'{filename}.npy', 'rb') as f:
            X = np.load(f)
            y = np.load(f)

        return X, y
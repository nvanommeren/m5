import pandas as pd
import numpy as np

# def create_timeseries(sales_data: pd.DataFrame, calendar_df: pd.DataFrame) -> pd.DataFrame:
    
#     calendar_df['date'] = pd.to_datetime(calendar_df['date'])
    
#     return (sales_data
#                  .groupby(['id']).sum().T
#                  .reset_index()
#                  .rename(columns={'index':'d'})
#                  .merge(calendar_df[['date', 'd']], on='d', how='left')
#                  .drop('d', axis=1)
#                  .set_index('date'))


# def create_splits(X: pd.DataFrame, n_splits: int, test_size: int):
#     counter=0
#     for i in range(len(X)-test_size, 0, -test_size):
#         if counter == n_splits:
#             break
#         counter +=1

#         train_index = range(0, i)
#         test_index = range(i, i+test_size)
        
#         yield train_index, test_index

def create_timeseries(df: pd.DataFrame, date_cols: list) -> pd.DataFrame:

    return (df
             .melt(
                id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], 
                value_vars=date_cols,
                value_name='sales',
                var_name='d')
    )


# def add_forecast_period(df: pd.DataFrame, days: int) -> (pd.DataFrame, list):

#     max_date_id = max([int(x.strip('d_')) for x in df.columns if 'd_' in x])

#     forecast_date_cols = ['d_' + str(x) for x in range(max_date_id + 1, max_date_id + days + 1)]

#     df = df.assign(**dict.fromkeys(forecast_date_cols, 0))
    
#     return df, forecast_date_cols


def create_splits(X: pd.DataFrame, n_splits: int, test_size: int):

    dates = X['date'].drop_duplicates()

    counter=0
    for i in range(len(dates)-test_size, 0, -test_size):
        if counter == n_splits:
            break
        counter +=1

        train_dates = dates.iloc[range(0, i)].values
        test_dates = dates.iloc[range(i, i+test_size)].values
 
        yield (X[X['date'].apply(lambda x: x in train_dates)].index, 
               X[X['date'].apply(lambda x: x in test_dates)].index)


def downcast_int_cols(df: pd.DataFrame)  -> pd.DataFrame:
    for col in df.select_dtypes(include=["int"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df
    
###
#    Features
###

def add_lagged_features(X: pd.DataFrame, lags: list, shift_days: int) -> pd.DataFrame:
    
    for l in lags:
        X[f'sales_lag_{l + shift_days}'] = (X[['id', 'sales', 'd']]
                .groupby('id')['sales']
                .transform(lambda x: x.shift(l  + shift_days))
                .fillna(0))
    return X

def add_rolling(X: pd.DataFrame, aggregate: str, days: list, shift_days: int) -> pd.DataFrame:
    
    for d in days:
        X[f'rolling_{aggregate}_{d}'] = (X[['id', 'sales', 'd']]
                    .groupby(['id'])['sales']
                    .transform(lambda x: x
                               .shift(shift_days + 1)
                               .rolling(d).agg(aggregate))
                    .fillna(0))
    return X
      
def create_submission_file(X_test: pd.DataFrame, days: int) -> pd.DataFrame:
    submission_df = X_test.pivot(index='id', columns='d', values='pred_test')

    submission_df.columns = [f'F_{x}' for x in range(1, days + 1)]

    return submission_df.reset_index()

###
#    Evaluation metric
### 

# Source: https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9
def rmsse(actual: np.ndarray, predicted: np.ndarray, seasonality: int = 1):
    """ Root Mean Squared Scaled Error """
    actual = np.array(actual)
    predicted = np.array(predicted)
    q = _mse(actual, predicted) / _mse(actual[seasonality:], _naive_forecasting(actual, seasonality))
    return np.sqrt(q)

def _error(actual: np.ndarray, predicted: np.ndarray):
    """ Simple error """
    return actual - predicted

def _mse(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Squared Error """
    return np.mean(np.square(_error(actual, predicted)))

def _naive_forecasting(actual: np.ndarray, seasonality: int = 1):
    """ Naive forecasting method which just repeats previous samples """
    return actual[:-seasonality] 


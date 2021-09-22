import pandas as pd
import numpy as np

class DataPrep:
    
    def __init__(self, filepath: str, sample: int=None):
        self.filepath = filepath
        self.sample = sample
        self.sample_selection = []
    
    def parse_data(self, start_date: int=None, train: bool=True) -> pd.DataFrame:
                
        sales_df = self._parse_sales_data(start_date, train)
        
        calendar_df = self._parse_calendar_data()
        
        sales_df = (sales_df
             .merge(calendar_df, on='d', how='left')
             .sort_values(['id', 'd']))
                
        prices_df = self._parse_price_data()
        
        sales_df = (sales_df
             .merge(prices_df, on=['wm_yr_wk', 'store_id', 'item_id'], how='left')
             .drop('wm_yr_wk', axis=1)
             .assign(sell_price=lambda df: df['sell_price'].fillna(0.0)))
        
        
        sales_df = self._parse_snap(sales_df)
        
        return sales_df

    
    def _parse_calendar_data(self) -> pd.DataFrame:
        
        calendar_df = pd.read_csv(
            f'{self.filepath}/calendar.csv', 
            dtype={
                'event_type_1': 'category'
            })
        
        calendar = (calendar_df
            .assign(is_weekend=lambda df: df['weekday'].apply(lambda x: x in ['Saturday', 'Sunday']))
            .assign(d=lambda df: df['d'].apply(lambda x: int(x.strip('d_'))))
            [['snap_CA', 'snap_TX', 'snap_WI', 'is_weekend', 'wm_yr_wk', 'wday', 
              'month', 'event_type_1', 'd']])
              
        return self._downcast_cols(calendar, "int", "integer")
     
    def _parse_sales_data(self, start_date: int, train: bool) -> pd.DataFrame: 
        
        df = pd.read_csv(
            f'{self.filepath}/sales_train_evaluation.csv', 
            dtype={
                    'id': 'category',
                    'item_id': 'category',
                    'dept_id': 'category', 
                    'cat_id': 'category', 
                    'store_id': 'category', 
                    'state_id': 'category'
            })
        
        # Rename the date column to save memory
        date_cols_rename = {
             x: int(x.strip('d_')) 
             for x in df.columns 
             if 'd_' in x
        }

        df = df.rename(columns=date_cols_rename)
        
        if train and self.sample:
            df = df.sample(self.sample)
            self.sample_selection = df.index
            df = df.reset_index()
        
        if (not train) and self.sample:
            df = df.iloc[self.sample_selection]
            df = df.reset_index()
        
        X = self._create_timeseries(df, date_cols_rename.values())

        # Downcast the sales and date columns
        X['sales'] = pd.to_numeric(X['sales'], downcast='integer')
        X['d'] = pd.to_numeric(X['d'], downcast='integer')
        
        if train:
            X = X[X['d'] < start_date]
        else:
            X = X[X['d'] >= start_date]
        
        return X
    
    def _parse_price_data(self) -> pd.DataFrame:
        
        prices_df = pd.read_csv(f'{self.filepath}/sell_prices.csv', 
                                dtype={
                                    'store_id': 'category',
                                    'item_id': 'category'})

        prices_df['wm_yr_wk'] = pd.to_numeric(prices_df['wm_yr_wk'], downcast='integer')
        prices_df['sell_price'] = (pd
                                   .to_numeric(prices_df['sell_price'], downcast='float'))
        
        return prices_df

    def _create_timeseries(self, df: pd.DataFrame, date_cols: list) -> pd.DataFrame:

        return (df
                 .melt(
                    id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], 
                    value_vars=date_cols,
                    value_name='sales',
                    var_name='d')
        )
    
    def _parse_snap(self, sales: pd.DataFrame) -> pd.DataFrame:
        
        def snap_values(state):
            return sales.iloc[np.where(sales['state_id'] == state)][f'snap_{state}']

        sales['snap'] = (snap_values('CA')
             .append(snap_values('TX'), verify_integrity=True)
             .append(snap_values('WI'), verify_integrity=True)
             .sort_index())

        return sales.drop(['snap_CA', 'snap_TX', 'snap_WI'], axis=1)
    
    def _downcast_cols(self, df: pd.DataFrame, select_type: str, cast_type: str) -> pd.DataFrame:
        
        for col in df.select_dtypes(include=[select_type]).columns:
            df[col] = pd.to_numeric(df[col], downcast=cast_type)
        return df
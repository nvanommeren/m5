import pandas as pd
import numpy as np

class DataPrep:

    def __init__(self, filepath: str, sample: int=None):
        """Creates a class for reading the data and storing it in
        a memory efficient way to make sure we can make predictions
        on the full set.

        Args:
            filepath (str): Path to the data folder.
            sample (int, optional): When given it only takes a sample
            of X item_score combinations. Recommended for testing purposes
            as running on the full set takes long. Defaults to None in which
            case it uses the enire dataset.
        """
        self.filepath = filepath
        self.sample = sample
        self.sample_selection = []

    def parse_data(self, start_date: int=None, train: bool=True) -> pd.DataFrame:
        """Parses the sales, calendar and price data.

        Args:
            start_date (int, optional): When given use this start date as the
            minimum date for the sales data. Defaults to None in which case it
            takes all data.
            train (bool, optional): [description]. When True parse the
            training data, set to False for the test data.

        Returns:
            pd.DataFrame: A dataframe with a unique item, day per row, including
            relevant calendar and price information per row.
        """
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
        """ Reads and parses the price information.

        Returns:
            pd.DataFrame: Returns a dataframe with the
            sell_price, wm_yr_wk, store_id and item_id.
        """
        prices_df = pd.read_csv(f'{self.filepath}/sell_prices.csv',
                                dtype={
                                    'store_id': 'category',
                                    'item_id': 'category'})

        prices_df['wm_yr_wk'] = pd.to_numeric(prices_df['wm_yr_wk'], downcast='integer')
        prices_df['sell_price'] = (pd
                                   .to_numeric(prices_df['sell_price'], downcast='float'))

        return prices_df

    def _create_timeseries(self, df: pd.DataFrame, date_cols: list) -> pd.DataFrame:
        """Converts the sales data into a dataframe with one item_id, store_id and date
        per row.

        Args:
            df (pd.DataFrame): Sales data.
            date_cols (list): Names of the columns including dates.

        Returns:
            pd.DataFrame: Converetd dataframe.
        """
        return (df
                 .melt(
                    id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
                    value_vars=date_cols,
                    value_name='sales',
                    var_name='d')
        )

    def _parse_snap(self, sales: pd.DataFrame) -> pd.DataFrame:
        """Parses the SNAP values such that each store uses he SNAP value
        relevant for the store. Removes the redundant SNAP columns
        afterwards.

        Args:
            sales (pd.DataFrame): Dataframe including the sales.

        Returns:
            pd.DataFrame: Sales data with new SNAP value column.
        """
        def snap_values(state):
            return sales.iloc[np.where(sales['state_id'] == state)][f'snap_{state}']

        sales['snap'] = (snap_values('CA')
             .append(snap_values('TX'), verify_integrity=True)
             .append(snap_values('WI'), verify_integrity=True)
             .sort_index())

        return sales.drop(['snap_CA', 'snap_TX', 'snap_WI'], axis=1)

    def _downcast_cols(self, df: pd.DataFrame, select_type: str, cast_type: str) -> pd.DataFrame:
        """Downcast columns to make sure everything fits in memory when running
        on the full dataset, for example a int64 can in some cases be converted
        to a int16.

        Args:
            df (pd.DataFrame): Sales data.
            select_type (str): Type selected for conversion, e.g. 'int'
            cast_type (str): Type to downcast to, e.g. 'integer'.

        Returns:
            pd.DataFrame: [description]
        """
        for col in df.select_dtypes(include=[select_type]).columns:
            df[col] = pd.to_numeric(df[col], downcast=cast_type)
        return df
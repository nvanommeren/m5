import pandas as pd
import numpy as np

import matplotlib.pylab as plt


class Visuals:
    
    def __init__(self):
        pass
    
    def plot_feature_importances(self, lgbm_model: list, column_names: list):

        (pd.DataFrame(
            {'feature': column_names,
             'importance': lgbm_model.feature_importances_, 
            }
        )
        .sort_values('importance', ascending=True)
        .set_index('feature')
        .plot(kind='barh', figsize=(20, 15)));
        
    def plot_series(self, X_total: pd.DataFrame, selected_nr: int):

        selected_id = np.unique(X_total['id'])[selected_nr]

        selected_series = X_total[X_total['id'] == selected_id]

        fig, ax = plt.subplots(figsize=(18,6))

        selected_series.plot(x='d', y='sales', ax=ax, title=f"{selected_id}")
        selected_series.plot(x='d', y='pred', ax=ax);
        selected_series.plot(x='d', y='pred_test', ax=ax);

        ax.legend(["sales", "prediction", "test_prediction"], prop={'size': 14});
        
    def plot_series_zoomed(self, X_total: pd.DataFrame, selected_nr: int):

        min_plot_date = 1900

        selected_id = np.unique(X_total['id'])[selected_nr]

        selected_series = X_total[(X_total['id'] == selected_id) 
                                  & (X_total['d'] > min_plot_date)]

        fig, ax = plt.subplots(figsize=(18,6))

        selected_series.plot(x='d', y='sales', ax=ax, title=f"Zoomed plot {selected_id}")
        selected_series.plot(x='d', y='pred', ax=ax);
        selected_series.plot(x='d', y='pred_test', ax=ax);

        ax.legend(["sales", "prediction", "test_prediction"], prop={'size': 14});
        
    
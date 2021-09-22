import argparse
import numpy as np
import pandas as pd
import random

from lightgbm.sklearn import LGBMRegressor
from joblib import dump, load

from src.data_prep import DataPrep
from src.feature_transformer import FeatureTransformer
from src.cross_validator import CrossValidator

random.seed(42)

def run_pipeline(args):
    
    days_ahead = args.days_ahead
    
    shift_days = days_ahead - 1

    start_test_date = 1914
    
    data_prep = DataPrep(args.data_folder, sample=args.sample)

    train_df = data_prep.parse_data(start_test_date, train=True)
    test_df = data_prep.parse_data(start_test_date, train=False)
    
    ft = FeatureTransformer(shift_days)

    train_df = ft.calculate_timebase_features(train_df)
    X_train = ft.fit_transform(train_df)

    test_df = ft.calculate_timebase_features(test_df)
    X_test = ft.transform(test_df)

    if args.save_data:
        ft.save_data(X_test, test_df['sales'], 'test')
        ft.save_data(X_train, train_df['sales'], 'train')
    
    cv = CrossValidator(n_splits=1, test_size=28)
    
    if args.cross_validation:
        print(cv.grid_search(X_train, train_df['sales'], train_df))
    
    # Best estimator and params according to the cross_validation
    lgbm = LGBMRegressor(n_estimators=100)

    lgbm.fit(X_train, np.array(train_df['sales']))

    dump(lgbm, f'models/lgbm_100_estimators_{shift_days}.joblib') 

    train_df['pred'] = lgbm.predict(X_train)
    test_df['pred_test'] = lgbm.predict(X_test)
    
    rmsse_train = cv.rmsse(train_df['sales'], train_df['pred'])
    rmsse_test = cv.rmsse(test_df['sales'], test_df['pred_test'])
    
    print(f"RMSSE on the train set: {rmsse_train: .2f}")
    print(f"RMSSE on the test set: {rmsse_test: .2f}")
    
    if args.create_submission:
        create_submission_file(test_df)
        
    
def create_submission_file(X_test: pd.DataFrame, days: int=28) -> pd.DataFrame:
    submission_df = X_test.pivot(index='id', columns='d', values='pred_test')

    submission_df.columns = [f'F_{x}' for x in range(1, days + 1)]
    
    submission_df.to_csv('submission.csv', index=False)
    
    return submission_df.reset_index()

    
def main():
    parser = argparse.ArgumentParser(
        description="Runs the pipeline"
    )
    
    parser.add_argument(
        "--days-ahead", type=int, default=1, help="Create model for X days ahead. ",
    )

    parser.add_argument(
        "--data-folder", type=str, default='./data', 
        help="Folder containing the data", required=False
    )
    
    parser.add_argument(
        "--save-data", type=bool, default=False, 
        help="Save the processed data", required=False
    )
    
    parser.add_argument(
        "--cross-validation", type=bool, default=False, 
        help="Run cross validation", required=False
    )
    
    parser.add_argument(
        "--create-submission", type=bool, default=False,
        help="Create submisson file", required=False)
    
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Take a sample of X items from the sales data. " 
        "Recommended for testing purposes. ", required=False)
    
    args = parser.parse_args()

    run_pipeline(args)


if "__main__" == __name__:
    main()
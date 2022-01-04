"""
Read data from the csv file and split into train and test set and save in a local path

Usage: src/split_data.py --input_path=<input_path> --train_set_path=<train_set_path> --val_set_path=<val_set_path> --test_set_path=<test_set_path>

Options:
--input_path=<input_path>           path to the input raw data
--train_set_path=<train_set_path>   path to the output train data
--val_set_path=<val_set_path>       path to the output validation data
--test_set_path=<test_set_path>     path to the output test data

Example: python src/split_data.py --input_path=data/raw/airbnb.csv --train_set_path=data/processed/train_df.csv --val_set_path=data/processed/val_df.csv --test_set_path=data/processed/test_df.csv
"""

from docopt import docopt
import pandas as pd
import os
from sklearn.model_selection import train_test_split


def create_new_features(data):
    data.dropna(subset=["reviews_per_month"], inplace=True)
    data["last_review_year"] = pd.DatetimeIndex(data["last_review"]).year
    data["last_review_month"] = pd.DatetimeIndex(data["last_review"]).month
    data["last_review_date"] = pd.DatetimeIndex(data["last_review"]).day
    return data


def read_data(path):
    data = pd.read_csv(path)
    return data


def split_train_test_data(data):
    return train_test_split(data, test_size=0.2, random_state=123)


def split_train_val_data(data):
    return train_test_split(data, test_size=0.25, random_state=123)


def save_data(data, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    data.to_csv(path, index=False)


def main(input_path, train_set_path, val_set_path, test_set_path):
    df = read_data(input_path)
    train_val_df, test_df = split_train_test_data(df)
    train_val_df = create_new_features(train_val_df)
    test_df = create_new_features(test_df)
    train_df, val_df = split_train_val_data(train_val_df)
    save_data(train_df, train_set_path)
    save_data(test_df, test_set_path)
    save_data(val_df, val_set_path)


if __name__ == "__main__":
    opt = docopt(__doc__)
    main(opt["--input_path"], opt["--train_set_path"], opt["--val_set_path"], opt["--test_set_path"])


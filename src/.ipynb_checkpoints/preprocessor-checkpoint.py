"""
Reads train csv data from path, preprocess the data, build a preprocessor pipeline and output the preprocessor

Usage: preprocessor.py --train_set_path=<train_set_path>  --column_transformer_out_path=<column_transformer_out_path> 

Options:
--train_set_path=<train_set_path>                               path to the train set
--column_transformer_out_path=<column_transformer_out_path>     path to column transformer

Example:
python src/preprocessor.py --train_set_path=data/processed/train_df.csv  --column_transformer_out_path=models
"""

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    FunctionTransformer
)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

from docopt import docopt


def define_feature_types(X_train):
    numeric_features = [
        "latitude",
        "longitude",
        "price",
        "minimum_nights",
        "calculated_host_listings_count",
        "availability_365",
        "last_review_year",
        "last_review_date",
        "id",
        "host_id",
        "number_of_reviews",
        "number_of_reviews_ltm"
    ]

    categorical_features = [
        "neighbourhood_group",
        "room_type",
        "last_review_month"
    ]

    text_features = [
        "name"
    ]

    drop_features = [
        "license",
        "host_name",
        "last_review",
        "neighbourhood"
    ]

    passthrough_features = [

    ]

    target_column = "reviews_per_month"
    return numeric_features, categorical_features, text_features, drop_features, passthrough_features


def define_column_transformer(X_train,
                              numerical_features,
                              categorical_features,
                              text_features,
                              drop_features,
                              passthrough_features,
                              column_transformer_out_path):

    function_transformer = FunctionTransformer(
        np.reshape, kw_args={"newshape": -1}
    )

    pipe_text_feats = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        function_transformer,
        CountVectorizer(stop_words="english", max_features=30)
    )

    column_transformer = make_column_transformer(
        (StandardScaler(), numerical_features),
        (OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_features),
        (pipe_text_feats, text_features),
        ("drop", drop_features),
        ("passthrough", passthrough_features)
    )

    default_preprocessor_out_path = column_transformer_out_path + "/column_transformer.pkl"

    try:
        pickle.dump(column_transformer, open(default_preprocessor_out_path, "wb"))
    except:
        os.makedirs(os.path.dirname(default_preprocessor_out_path))
        pickle.dump(column_transformer, open(default_preprocessor_out_path, "wb"))


def main(train_set_path, column_transformer_out_path):
    train_df = pd.read_csv(train_set_path)
    X_train, y_train = train_df.drop(columns=["reviews_per_month"]), train_df["reviews_per_month"]
    numerical_features, categorical_features, text_features, drop_features, passthrough_features = define_feature_types(X_train)
    column_transformer = define_column_transformer(
        X_train,
        numerical_features,
        categorical_features,
        text_features,
        drop_features,
        passthrough_features,
        column_transformer_out_path
    )


if __name__ == "__main__":
    opt = docopt(__doc__)
    main(opt["--train_set_path"], opt["--column_transformer_out_path"])
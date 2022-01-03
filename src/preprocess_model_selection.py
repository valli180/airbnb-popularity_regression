"""
Reads train csv data from path, preprocess the data, build a preprocessor pipeline and output the preprocessor, give the results from cross validation, output an ensemble model

Usage: preprocess_model_selection.py --train_set_path=<train_set_path> --column_transformer=<column_transformer> 

Options:

--train_set_path=<train_set_path>  
--column_transformer=<column_transformer> 
 
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    FunctionTransformer
)

from docopt import docopt

opt = docopt(__doc__)


def save_csv(results, results_out_path, filename="/results.csv"):

    if not os.path.exists(results_out_path):
        os.makedirs(results_out_path)

    pd.DataFrame(results).to_csv(results_out_path + filename, encoding="utf-8")

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
    assert(len(numeric_features) + len(categorical_features) + len(text_features) + len(drop_features) + len(passthrough_features) +
        len([target_column])) == len(X_train.columns)
    return numeric_features, categorical_features, text_features, drop_features, passthrough_features


def define_column_transformer(X_train,categorical_features, pass_through_features, numerical_features,text_features, column_transformer_out_path):
    function_transformer = FunctionTransformer(
        np.reshape, kw_args={"newshape": -1}
    )

    pipe_text_feats = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        function_transformer,
        CountVectorizer(stop_words="english", max_features=30)
    )

    column_transformer = make_column_transformer(
        (StandardScaler(), numeric_features),
        (OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_features),
        (pipe_text_feats, text_features),
        ("drop", drop_features),
        ("passthrough", passthrough_features)
    )

    print("Column Transformer!")

    default_preprocessor_out_path = column_transformer_out_path + "/--column_transformer.pkl"
    try:
        pickle.dump(prepr, open(default_preprocessor_out_path, "wb"))
    except:
        os.makedirs(os.path.dirname(default_preprocessor_out_path))
        pickle.dump(column_transfomer, open(default_preprocessor_out_path, "wb"))
    
    print("Column Transformer dumped!\n\n")

    return column_transformer


def main(train_set_path, column_transformer):
    train_df = pd.read_csv(train_set_path)
    X_train, y_train = train_df.drop(columns = ["reviews_per_month"]), train_df["reviews_per_month"]
    numeric_features, categorical_features, text_features, drop_features, passthrough_features = define_feature_types(X_train)
    column_transformer = define_column_transformer(
        X_train,categorical_features, 
        pass_through_features, 
        numerical_features,
        text_features, 
        column_transformer_out_path
    )
    
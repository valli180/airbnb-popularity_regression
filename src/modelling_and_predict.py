"""
Reads train csv data from path, loads the column transformer, train a model and build an ensemble model

Usage: src/modelling_and_predict.py --train_set_path=<train_set_path> --val_set_path=<val_set_path> --test_set_path=<test_set_path> --column_transformer_in_path=<column_transformer_in_path> --results_out_path=<results_out_path> --model_out_path=<model_out_path>

Options:

--train_set_path=<train_set_path>                             path to the train data
--val_set_path=<val_set_path>                                 path to the validation data 
--test_set_path=<test_set_path>                               path to the test data
--column_transformer_in_path=<column_transformer_in_path>     path to column transformer
--results_out_path=<results_out_path>                         save results to csv file location

Example:
python src/modelling_and_predict.py --train_set_path=data/processed/train_df.csv --val_set_path=data/processed/val_df.csv --test_set_path=data/processed/test_df.csv --column_transformer_in_path=models/column_transformer.pkl --results_out_path=results/results.csv --model_out_path=models/stacking_model.pkl
"""

import pandas as pd
import os
import pickle

from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso

from catboost import CatBoostRegressor
from lightgbm.sklearn import LGBMRegressor
from xgboost import XGBRegressor

from sklearn.ensemble import StackingRegressor
from sklearn.pipeline import make_pipeline

from docopt import docopt
from preprocessor import dump_pkl


def save_csv(results, results_out_path, filename="/results.csv"):
    if not os.path.exists(results_out_path):
        os.makedirs(results_out_path)

    pd.DataFrame(results).to_csv(results_out_path + filename, encoding="utf-8")


def cross_val_scores(model, X_train, y_train, X_val, y_val, return_train_score):

    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)

    score_dict = {
        "r2_test": model.score(X_val, y_val),
        "mse_test": mean_squared_error(y_val, y_val_pred),
        "mape_test": mean_absolute_percentage_error(y_val, y_val_pred)
    }

    if return_train_score:
        y_train_pred = model.predict(X_train)

        score_dict["r2_train"] = model.score(X_train, y_train)
        score_dict["mse_train"] = mean_squared_error(y_train, y_train_pred)
        score_dict["mape_train"] = mean_absolute_percentage_error(y_train, y_train_pred)

    scores_result = pd.Series(score_dict)

    return model, scores_result


def get_base_models(column_transformer):
    return {
        "Dummy": make_pipeline(
            column_transformer, DummyRegressor()
        ),
        "Ridge": make_pipeline(
            column_transformer, Ridge()
        ),
        "Lasso": make_pipeline(
            column_transformer, Lasso()
        ),
        "Random_Forest_reg": make_pipeline(
            column_transformer, RandomForestRegressor(random_state=123)
        ),
        "XGBoost_reg": make_pipeline(
            column_transformer, XGBRegressor(verbosity=0)
        ),
        "lgbm_reg": make_pipeline(
            column_transformer, LGBMRegressor()
        ),
        "catBoost_reg": make_pipeline(
            column_transformer, CatBoostRegressor(verbose=0)
        ),
        'poly_ridge': make_pipeline(
            column_transformer, PolynomialFeatures(degree=2), Ridge()
        )
     }


def train(X_train, y_train, X_val, y_val, results, models):
    for model_name, model_value in models.items():
        print("Training", model_name)
        _, results[model_name] = cross_val_scores(
            model_value,
            X_train,
            y_train,
            X_val,
            y_val,
            return_train_score=True
        )
        print(model_name, "done!\n")
    return pd.DataFrame(results)


def get_stacking_model(column_transformer, X_train, y_train, X_val, y_val):
    models_stack = {
        "Ridge": make_pipeline(
            column_transformer, PolynomialFeatures(degree=2), Ridge()
        ),
        "Random_Forest_reg": make_pipeline(
            column_transformer, RandomForestRegressor()
        ),
        "lgbm_reg": make_pipeline(
            column_transformer, LGBMRegressor()
        ),
        "catBoost_reg": make_pipeline(
            column_transformer, CatBoostRegressor(verbose=0)
        )
    }
    stacking_model = StackingRegressor(list(models_stack.items()))
    return cross_val_scores(
        stacking_model,
        X_train,
        y_train,
        X_val,
        y_val,
        return_train_score=True
    )


def get_score(model, X_test, y_test):
    return model.score(X_test, y_test)


def main(train_set_path, val_set_path, test_set_path, column_transformer_in_path, results_out_path, model_out_path):

    column_transformer = pickle.load(open(column_transformer_in_path, "rb"))

    train_data = pd.read_csv(train_set_path)
    val_data = pd.read_csv(val_set_path)
    test_data = pd.read_csv(test_set_path)

    target_column = "reviews_per_month"

    X_train, y_train = train_data.drop(columns=[target_column]), train_data[target_column]
    X_val, y_val = val_data.drop(columns=[target_column]), val_data[target_column]
    X_test, y_test = test_data.drop(columns=[target_column]), test_data[target_column]

    results = {}

    models = get_base_models(column_transformer)

    results = train(X_train, y_train, X_val, y_val, results, models)

    stacking_model, stacking_results = get_stacking_model(column_transformer, X_train, y_train, X_val, y_val)

    results["Stacking"] = stacking_results

    print("Saving results!\n\n")
    save_csv(results, results_out_path, filename="/results.csv")
    print("Results saved!")

    dump_pkl(stacking_model, model_out_path)

    print("Final Score:", get_score(stacking_model, X_test, y_test))


if __name__ == "__main__":
    opt = docopt(__doc__)
    main(opt["--train_set_path"], opt["--val_set_path"], opt["--test_set_path"], opt["--column_transformer_in_path"], opt["--results_out_path"], opt["--model_out_path"])






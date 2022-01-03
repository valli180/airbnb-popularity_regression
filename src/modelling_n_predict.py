"""
Reads train csv data from path, loads the column transformer, train a model and build an ensemble model

Usage: modelling.py --train_set_path=<train_set_path> --val_set_path=<val_set_path> --test_set_path=<test_set_path> --column_transformer_out_path=<column_transformer_out_path> --results_out_path=<results_out_path> --prediction_out_path=<prediction_out_path> 

Options:

--train_set_path=<train_set_path>                             path to the train data
--val_set_path=<val_set_path>                                 path to the validation data 
--test_set_path=<test_set_path>                               path to the test data
--column_transformer_out_path=<column_transformer_out_path>   path to column transformer
--results_out_path=<results_out_path>                         save results to csv file location
--prediction_out_path=<prediction_out_path>                   save score on test data to csv
 
 
"""

from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso

from catboost import CatBoostRegressor
from lightgbm.sklearn import LGBMRegressor
from xgboost import XGBRegressor

from sklearn.ensemble import StackingRegressor

from sklearn.feature_selection import SequentialFeatureSelector

from docopt import docopt

opt = docopt(__doc__)


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

    
def base_models(column_transformer):
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
        'poly_ridge' = make_pipeline(
            column_transformer, PolynomialFeatures(degree=2), Ridge()
        )
     }
    
def train(X_train, y_train, X_val, y_val, results, model_name, model_value):
    for model_name, model_value in models.items():
        _, results[model_name] = cross_val_scores(
            model_value,
            X_train,
            y_train,
            X_val,
            y_val,
            return_train_score=True
        )
    return pd.DataFrame(results)


def stack_models(column_transformer):
    return {
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

def stacking_model():
    stacking_model = StackingRegressor(list(.items()))
    return 
    
def predict(X_test, y_test, prediction, model_name, model_value):
    model=stacking_model()


def main(train_set_path, val_set_path, test_set_path, column_transformer_out_path, results_out_path, prediction_out_path):
    
    column_transformer = pickle.load(open(column_transformer_out_path, "rb"))
    X_train, y_train = train_data.drop(columns=[target_column]), train_data[target_column]
    X_val, y_val = val_data.drop(columns=[target_column]), val_data[target_column]
    X_test, y_test = test_data.drop(columns=[target_column]), test_data[target_column]
    
    results_base = {}
    models = base_models(column_transformer)
    for model_name, model in models.items():
        print("Training", model_name)
        results_base = train(X_train, y_train, X_val, y_val, results_base, model_name, model)
        print(model_name, "done!\n")
    
    models_stack = stack_models(column_transformer)
    for model_name, model in models_stack.items():
        results_base = train(X_train, y_train, X_val, y_val, results_base, model_name, model)

    print("Saving results!\n\n")
    save_csv(results_base, results_out_path, filename="/results.csv")
    print("Results saved!")
     
    
if __name__ == "__main__":
    main(opt["--train_set_path"], opt["--val_set_path"], opt["--test_set_path"], opt["--column_transformer_out_path"], opt["--results_out_path"], opt["--prediction_out_path"])






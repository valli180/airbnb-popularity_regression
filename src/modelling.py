"""
Reads train csv data from path, loads the column transformer, train a model and build an ensemble model

Usage: modelling.py --train_set_path=<train_set_path> --column_transformer=<column_transformer> 

Options:

--train_set_path=<train_set_path>  
--score_data=<score_data> 
--column_transformer=<column_transformer> 
 
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

# Adapted from UBC Lecture Notes

def mape(true, pred):
    return 100.0 * np.mean(np.abs((pred - true) / true))

    # make a scorer function that we can pass into cross-validation
    mape_scorer = make_scorer(mape, greater_is_better=False)

    scoring_metrics = {
        "neg RMSE": "neg_root_mean_squared_error",
        "r2": "r2",
        "mape": mape_scorer,
    }

    
def base_models():
    models = {
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

    for model_name, model in models.items():
        _, results[model_name] = cross_val_scores(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            return_train_score=True
        )
    return pd.DataFrame(results)

results_base = {}

    models = get_models(preprocessor)

    for model_name, model in models.items():
        print("Training", model_name)
        results_base = train(results_base, model_name, model, X_train, y_train)
        print(model_name, "done!\n")

    print("Saving results!\n\n")
    save_csv(results_base, results_out_path, filename="/base_results.csv")
    print("Results saved!")

if __name__ == "__main__":
    opt = docopt(__doc__)
    main(opt["--train_in_path"], opt["--preprocessor_out_path"], opt["--results_out_path"])






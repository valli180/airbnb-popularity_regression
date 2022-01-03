""" Create EDA plots using the train data and save the plots as png files in the given directory

Usage: src/eda.py --train_set_path=<train_set_path> --out_dir=<out_dir>

Options:
--train_set_path=<train_set_path>   path to the train set
--out_dir=<out_dir>                 directory in which the png files of the eda plots are stored

"""

from docopt import docopt
import pandas as pd
import numpy as np
import altair as alt
alt.data_transformers.disable_max_rows()
import os

opt = docopt(__opt__)


def main(train_set_path, out_dir):
    
    train_df = pd.read_csv(train_set_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Work with Categorical features
    categorical_features_eda_final = [
        "neighbourhood_group",
        "room_type",
        "last_review_month"
    ]
    
    #Distribution of categorical features
    hist_cat = alt.Chart(train_data).mark_bar().encode(
    x= alt.X(alt.repeat(), type='nominal'),
    y=alt.Y('count()')
    ).properties(
        width=200,
        height=200
    ).repeat(categorical_features_eda_final, columns = 3, title = "Reviews per month vs Categorical features"
    )
    hist_cat.save(out_dir + "/histogram_categorical.png")
    
    #Dependence of reviews_per_month and categorical features
    neighbourhood_group = alt.Chart(train_data, title="Number of reviews vs Neighbourhood group").mark_boxplot().encode(
        x='neighbourhood_group',
        y='reviews_per_month',   
    ).properties(
    width=250,
    height=250
    )

    room_type = alt.Chart(train_data, title="Number of reviews vs room type").mark_boxplot().encode(
        x='room_type',
        y='reviews_per_month',   
    ).properties(
    width=250,
    height=250
    )

    last_review_month = alt.Chart(train_data, title="Number of reviews vs month").mark_boxplot().encode(
        x='last_review_month',
        y='reviews_per_month',   
    ).properties(
        width=250,
        height=250
    )

    final_plot = (neighbourhood_group | room_type | last_review_month)
    final_plot.save(out_dir + "/dependence_categorical.png")
    
    # Work with Numeric data
    numeric_cols = [
        "price", 
        "minimum_nights", 
        "calculated_host_listings_count",
        "availability_365", "number_of_reviews",
        "number_of_reviews_ltm"
    ]
    
    # Histogram for Numeric features
    hist_num = alt.Chart(train_data).mark_bar().encode(
        x=alt.X(alt.repeat(), type='quantitative',bin=alt.Bin(maxbins=10)),
        y=alt.Y('count()')
    ).properties(
        width=100,
        height=100
    ).repeat(numeric_cols, columns=3, title="Distribution of Numerical Features" )
    hist_num.save(out_dir + "/hist_numerical.png")
    
    # Dependence of Numeric features on reviews_per_month
    dependence_num = alt.Chart(train_data).mark_circle().encode(
        alt.X(alt.repeat(), type='quantitative'),
        y=alt.Y("reviews_per_month", type='quantitative')
    ).properties(
        width=200,
        height=200
    ).repeat(numeric_cols, columns = 3, title = "Reviews per month vs Categorical features"
    )
    dependence_num.save(out_dir + "/dependence_numerical.png")

    # Variation of reviews_per_month across years
    reviews_over_years = alt.Chart(train_data, title = "Increase in reviews over time").mark_area().encode(
                        x = "last_review_year",
                        y = "count(reviews_per_month)",
                        color = "count(reviews_per_month)")
    dependence_year = reviews_over_years + reviews_over_years.mark_square()
    dependence_year.save(out_dir + "/growth_across_year.png")
    
    #Correlation
    corr_df = (
        train_data.drop(["license", "host_name", "last_review", "neighbourhood"], axis = 1)
        .corr('spearman')
        .abs()
        .stack()
        .reset_index(name='corr'))


    corr_matrix = alt.Chart(corr_df).mark_rect().encode(
        x='level_0',
        y='level_1',
        size='corr',
        color='corr')

    corr_matrix.save(out_dir + "/Correlation_matrix.png")

if __name__ == "__main__":
    main(opt[--train_set_path], opt[--out_dir])
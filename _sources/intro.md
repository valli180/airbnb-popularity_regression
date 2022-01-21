AirBnB Dataset Final Report
============================

This is the report made for the project work in lab4 of 573-Feature and Model Selection

# Introduction

Author: Valli Akella

## Abstract:

This project is focused on applying machine learning- regression models to predict the reviews_per_month, which is being used as aproxy to determine the popularity of the listings in New York city. Airbnb can use this sort of model to predict how ppopular the future listings might become before they are even posted, perhaps it can guide the hosts to create their listings so as to attract the customers. I performed exploratory data analysis in order to visully analyze the data first and identify any patterns and trends.
I, then performed feature engineering and preprocessing to identify new features that can add value to the model prediction followed by model selection. In model selection, I trained to fit the basic models to find which one perfomed the best and hyperparameter tuning was done for those best performing models and finally I created an ensemble model with the best performing hyperparams. Finally, I concluded that it was not feasible to obtain a very high score with the features given.

## Research Question:

> The main purpose of the project was to determine if we are able to predict the popularity of a listing in Airbnb with the given features.

The popularity of Airbnb has increased over time and more and more poeple are opting to stay at an Airbnb than hotels owing to prices, flexibility, location , convenience and inclination towards home amenities. In such a scenario understanding the ways to attract the tourists may help both the owners and well as the customers and thereby increase the business for Airbnb.


## DataSet:

The dataset was secured from the Inside AirBnB website. The data behind the Inside Airbnb site is sourced from publicly available information from the Airbnb site. The data has been analyzed, cleansed and aggregated where appropriate to faciliate public discussion.
It is live data which is updated on day to day basis. The dataset had last_review column which is date when it was last reviewed. I split that column into year, month and date to check if I can get any information from that. Final features are as follows:


| Feature Name          |      Description      |
|-----------------------|-----------------------|
| `id`     | listing ID |
| `name` | name of the listing  |
| `host_id`| host ID |
| `host_name`| name of the host |
| `neighborhood_group`| location of the property  |
| `neighbourhood`| area of the propoerty |
| `latitude`| latitude coordinates|
| `longitude`| longitude coordinates |
| `room_type`|  listing space type |
| `price` | price in dollars |
| `minimum_nights`| amount of nights minimum |
| `number_of_reviews`| number of reviews |
| `last_review`| latest review|
| `reviews_per_month` | number of reviews per month |
| `calculated_host_listings_count`| number of listing per host |
| `availability_365` |number of days when listing is available for booking |
| `last_review_year`| year of last review |
| `last_review_month`| month of last review |
| `last_review_date`| date of the last review|
<figcaption align = "center"><b>Table.1 -features generated</b></figcaption>

The reviews_per_month is the target column. The number_of_reviews column has been dropped because it seems to be redundant with the target column and can mislead the prediction. Also the columns namely id(serial number of the property) and host_name dont seem to be relevant to predict the popularity.  Same way the neighbourhood would be same has the neighbourhood_group,so to avoid redundancy these have been dropped.




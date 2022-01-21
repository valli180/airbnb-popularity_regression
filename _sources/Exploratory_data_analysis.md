# Exploratory Data Analysis

 After dropping unwanted columns, the data was split into training and test sets and EDA was done on the training set.
 The dataset consists of numeric as well as categorical features. The relation between the target and the numerical columns as well as the categorical columns is as below.
 
![Numerical_features vs Reviews](../results/num_rel_plot.png)
<figcaption align = "left"><b>Fig.1 Numerical_features vs Reviews - </b></figcaption>
The numerical features in the dataset are namely latitude, longitude, price, minimum_nights, availability_365, calculated_host_listings_count, last_review_year, last_review_date, id, host_id, number_of_reviews, number_of_reviews_ltm.
The above plot shows that reviews depend on all these features.

![Categorical_features vs Reviews](../results/cat_dependence.png)
<figcaption align = "left"><b>Fig.2 Categorical_features vs Reviews - </b></figcaption>

The categorical columns are namely neighbourhood_group, room_type, last_review_month. 
From the above plot we observe that the reviews depend on the neighborhood as expected. The room types entire home/apartment and private rooms received largest number of reviews. This shows that more number of people prefer them.
Also, reviews increased in the month of October. This may be due to some seasonal effect that has to be further studied 


![Increase in Popularity with Time](../results/final_plot.png)
<figcaption align = "left"><b>Fig.3 - Line plot for Increase in Popularity with Time</b></figcaption>

In the above graph, there is an increasing trend in the number of reviews over the years. We can draw the insight that the popularity for AirBnb has increased over time. 


![Correlation Matrix](../results/corr_matrix.png)
<figcaption align = "left"><b>Fig.2 - Density plot of Page Values</b></figcaption>
An attempt to find if any of the features are correlated with each other showed that there is no such dependence.
Thereby considering all the features selected to proceed to feature engineering and model selection.
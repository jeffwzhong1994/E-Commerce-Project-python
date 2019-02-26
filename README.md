# E-Commerce-Project-python
In this project, I am using Amazon Prime Video Dataset which have more than 4000+ rows and 16 features, 
to build a prediction model to predict whether a movie is going to perform well on the platform (cvt_per_day).

In the process of getting the best prediction model, I have done:
1. Cleaning up the missing values, replacing them with NaN, and filling up the missing value with column means
2. Feature Understanding& Feature Engineering: Convert the categorical features to dummy variables, especially on the "Genres" feature,
which I combined the six least frequent genres into one for model training.
3. Data Visualization: Using both logscale due to data variation and heatmap to vividly show the dataset
4. Model Training: Use Linear Model such as Lasso/Ridge in both linear and polynomial form as well as Non-linear model Random Forest 
to train the model, fine-tune the parameters to get the optimal results.
5. Model Evaluations: Comparing five different models using evaluation metrics such as: R square/score, MSE, RMSE to decide the best model
6. Feature Importance: After evaluating the best model(Random Forest), rank which features are the best to predict the models.



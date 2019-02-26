"""
Project Name: Amazon Prime Video Data Analysis

Author: Jeff Wenlue Zhong

Date: 02/26/19

"""

#Load Packages:
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import seaborn as sns
import sklearn 
from sklearn import preprocessing
pd.set_option('display.float_format', lambda x: '%.3f' % x)

#Import Data Set:
TV = pd.read_table('TVdata.txt', sep = ',' , header = 0, lineterminator = '\n')
print(TV.head(10))
print('Number of rows in the dataset: ' + str(TV.shape[0]))
print('Number of columns in the dataset: ' + str(TV.shape[1]))

#Remove duplicate video in the 
print()
index = TV.set_index('video_id').index
print("Duplicate Entry in the Dataset: ", len(index[index.duplicated()].unique()))
if len(index[index.duplicated()].unique()) == 0:
	print("There is no duplicated video in the dataset.")
print()

#Understand the Datasets:
print (TV.describe(percentiles = [.1, .25, .5, .75, .95]))
print("Empty cell in the Dataset:")
print((TV == 0).sum())
print()
print("Data types:")
print (TV.dtypes)

"""
From the above, we can see that among 10 numerical features, there are 4 features have over 25% of 
missing data: budget, boxoffice, metacritic_score, star_category. There are 2 features have less
than 10% of data missing: imdb_votes, imdb_ratings.

Next: Explore different numerical features

"""
#Explore cumulative view time per day.
plt.hist(TV['cvt_per_day'].values, bins = range(0, 15000, 30), alpha = 0.5, color = 'r', label = 'cvt_per_day', density = True)
plt.legend(loc = 'upper right')
plt.title('Histograms of cumulative view time per day before preprocessing')
plt.xlabel('cumulative view time per day')
plt.ylabel('density')
plt.show()

#Due to large variation, use log scale instead.
plt.hist(TV['cvt_per_day'].values, log = True, bins = range(0, 15000, 30), alpha = 0.5, color = 'g', label = 'cvt_per_day_log', density = True)
plt.legend(loc = 'upper right')
plt.title('Cumulative view time per day log scale')
plt.ylabel('density')
plt.show()

#Plot Heatmap:
corr = TV[['cvt_per_day','weighted_categorical_position','weighted_horizontal_poition','release_year', 'imdb_votes', 'budget',
 'boxoffice' ,'imdb_rating','duration_in_mins', 'metacritic_score', 'star_category']].corr()
sns.heatmap(corr, cmap = 'YlGnBu')
plt.title('Numerical Features Heatmap')
plt.show()
print("<Correlation>")
print(corr)

#Understand the Categorical Features:
sns.stripplot(x ='import_id', y ='cvt_per_day', data = TV, jitter = True)
plt.title('Import ID stripplot')
plt.xlabel('Import ID')
plt.ylabel('Cumulative view time per day')
plt.show()
print(TV['import_id'].value_counts().reset_index())
print()

sns.stripplot(x ='mpaa', y ='cvt_per_day', data = TV, jitter = True)
plt.title('mpaa stripplot')
plt.xlabel('Mpaa')
plt.ylabel('Cumulative view time per day')
plt.show()
print(TV['mpaa'].value_counts().reset_index())
print()

sns.stripplot(x ='awards', y ='cvt_per_day', data = TV, jitter = True)
plt.title('Awards stripplot')
plt.xlabel('Awards')
plt.ylabel('Cumulative view time per day')
plt.show()
print(TV['awards'].value_counts().reset_index())
print()

# Distribution of splitted genres:
# Some videos belong to more than 1 genre, the genre of each video is splited that would 
# help emphasize the effect of each individual genre.
genre_split = TV['genres'].str.get_dummies(sep = ',').sum()
genre_split.sort_values(ascending = False).plot.bar()
plt.title('Splitted Genres Distribution')
plt.xlabel('Genres')
plt.ylabel('Amount')
plt.show()

"""
It can be shown from the plot that genres such as Drama, Comedy, Thriller, Horror, Action
are the most popular, while the Anime, Reality, Lifestyle, Adult, LGBT, Holiday have lowest
frequencies.  So during the feature processing, these lower frequencies genres will be grouped
together as 'Misc_gen' in 'Genres' feature.

"""

"""
Explore distribution of release year feature:

The release year of video varies through a wide range. Because the popularity of a video usually
decays over time, the release_year should be bucketed based on the release_year range.

"""
plt.figure(1)
plt.hist(TV['release_year'].values, bins = range(1920, 2017, 1), alpha = 0.5 , color = 'r', label = 'release_year', density = True)
plt.legend(loc = 'upper left')
plt.title('Histograms of release_year before data processing')
plt.xlabel('release_year')
plt.ylabel('density')
plt.show()
print(TV['release_year'].describe(percentiles = [.1, .2, .3, .4, .5, .6, .7, .8, .9]))

"""
Feature Preprocessing:

1. There are 5 categorical features: Import_id, mpaa, awards, genres, release_year. There is no missing data in them, 
they can be converted into dummy/indicators.
2. The first three have relatively small sub-types, they can be easily converted to dummies.
3. The 'Genres' have 27 different sub-types, 6 of them are rarely observed. So these six are grouped into one.
4. The release_year is bined into 10 different buckets based on the year range between 1917 and 2017.

"""

#Convert 3 categorical variables into dummy variables:
dummy_import_id = pd.get_dummies(TV['import_id']).astype(np.int64)
dummy_mpaa = pd.get_dummies(TV['mpaa']).astype(np.int64)
dummy_awards = pd.get_dummies(TV['awards']).astype(np.int64)

#Convert Genres into dummy variables:
dummy_genres = pd.DataFrame()
for genre in ['Action', 'Adult', 'Adventure', 'Animation', 'Anime', 'Comedy', 'Crime', 'Documentary', 'Drama'
          , 'Fantasy', 'Foreign/International', 'Holiday', 'Horror', 'Independent', 'Kids & Family', 'LGBT', 
          'Lifestyle', 'Music', 'Musicals','Mystery', 'Reality', 'Romance','Sci-Fi', 'Sport', 'Thriller', 'War', 
          'Western']:
    gen_lst = []
    for i in range(TV.shape[0]):
    	if genre in TV['genres'][i]:
    		gen_lst.append(1)
    	else:
    		gen_lst.append(0)
    dummy_genres[genre] = pd.Series(gen_lst).values

dummy_genres['Misc_gen'] = dummy_genres['Anime']|dummy_genres['Reality']|dummy_genres['Lifestyle']|dummy_genres['Adult']|dummy_genres['LGBT']|dummy_genres['Holiday']
dummy_genres.drop(['Anime', 'Reality', 'Lifestyle', 'Adult', 'LGBT', 'Holiday'], inplace = True, axis = 1)

#Convert Release_year into dummy variables:
bin_year = [1916, 1974, 1991, 2001, 2006, 2008, 2010, 2012, 2013, 2014, 2017]
year_range = ['1916-1974','1974-1991','1991-2001','2001-2006','2006-2008','2008-2010','2010-2012','2012-2013','2013-2014','2014-2017']
year_bin = pd.cut(TV['release_year'], bin_year, labels = year_range)
dummy_year = pd.get_dummies(year_bin).astype(np.int64)

#Get the new Dataframe based on dropping previous categorical features and add new dummies variables, check for null
TV_tmp = TV.drop(['import_id','mpaa','awards','genres','release_year'], axis = 1)
newTV = pd.concat([TV_tmp, dummy_import_id, dummy_mpaa, dummy_awards,dummy_genres,dummy_year], axis = 1)

#Print out the info of the new Dataframe:
print(newTV.shape)
print(pd.isnull(newTV).any(1).nonzero()[0])
print(newTV.head())
newTV_copy = newTV.copy()

"""
Feature space holds 4226 observations and 58 features in total.
There is No null data, making a new copy of newTV so the raw Dataframe can 
be kept before any further feature processing.

"""

"""
Handle the Missing Data:

1. Among the 10 numerical feature, 4 feature have over 25% of missing values, 2 features have less than 10% of missing data
2. There are 3242 samples that have at least one missing data.

"""
#Mark zero as NaN
newTV[['budget', 'boxoffice', 'metacritic_score', 'star_category', 'imdb_votes', 'imdb_rating']] =\
newTV[['budget', 'boxoffice', 'metacritic_score', 'star_category', 'imdb_votes', 'imdb_rating']].replace(0, np.nan)

#Count the number of NaN values in each column:
print(newTV.isnull().sum())

#Count the number videos who have at least one missing data：
print ("Video that have at least one null value: ", newTV.isnull().any(axis = 1).sum())

"""
Filling the missing values with mean value:

1. For imdb_votes and imdb_rating, they always show "null" together, since when an imdb_votes is missing,
the imdb_rating is mostly also missing.
2. For all 6 features with missing data, the attempt is to fill in with their mean value.

"""
#Fill missing values with column means:
newTV_v1 = newTV.copy()
newTV_v1['budget'].fillna(newTV_v1['budget'].mean(), inplace = True)
newTV_v1['boxoffice'].fillna(newTV_v1['boxoffice'].mean(), inplace = True)
newTV_v1['metacritic_score'].fillna(newTV_v1['metacritic_score'].mean(), inplace = True)
newTV_v1['star_category'].fillna(newTV_v1['star_category'].mean(), inplace = True)
newTV_v1['imdb_votes'].fillna(newTV_v1['imdb_votes'].mean(), inplace = True)
newTV_v1['imdb_rating'].fillna(newTV_v1['imdb_rating'].mean(), inplace = True)

#Feature Scaling:
scale_lst = ['weighted_categorical_position', 'weighted_horizontal_poition', 'budget','boxoffice', 
             'imdb_votes','imdb_rating','duration_in_mins', 'metacritic_score','star_category']

newTV_sc = newTV_v1.copy()
sc_scale = preprocessing.StandardScaler().fit(newTV_sc[scale_lst])
newTV_sc[scale_lst] = sc_scale.transform(newTV_sc[scale_lst])
print(newTV_sc.head())

#Min Max Scaling:
newTV_mm = newTV_v1.copy()
mm_scale = preprocessing.MinMaxScaler().fit(newTV_mm[scale_lst])
newTV_mm[scale_lst] = mm_scale.transform(newTV_mm[scale_lst])

#Robust Scaling:
newTV_rs = newTV_v1.copy()
rs_scale = preprocessing.RobustScaler().fit(newTV_rs[scale_lst])
newTV_rs[scale_lst] = rs_scale.transform(newTV_rs[scale_lst])

"""
Model Training:
1. 85% of the samples will be used to train all models, and 20% is reserved for test the models in 
the next sections.
2. 15% of test data will kept aside, they won't be seen by the models until final test/comparison

"""

from sklearn.model_selection import train_test_split
model_train, model_test = train_test_split(newTV_sc, test_size = 0.15, random_state = 3)
model_train_x = model_train.drop(['video_id','cvt_per_day'], axis = 1)
model_test_x = model_test.drop(['video_id','cvt_per_day'], axis = 1)
model_train_y = model_train['cvt_per_day']
model_test_y = model_test['cvt_per_day']

#Import Packages: 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

# Linear Model:
# 1. Lasso_linear_regression:

lr_train, lr_validate = train_test_split(model_train, test_size = 0.15, random_state = 0)
lr_train_x = lr_train.drop(['video_id','cvt_per_day'], axis = 1)
lr_validate_x = lr_validate.drop(['video_id','cvt_per_day'], axis = 1)
lr_train_y = lr_train['cvt_per_day']
lr_validate_y = lr_validate['cvt_per_day']

alphas = np.logspace(-10, 2.5, num = 150)
scores = np.empty_like(alphas)
opt_a = float('-inf')
max_score = float('-inf')
for i,a in enumerate(alphas):
	lasso = linear_model.Lasso()
	lasso.set_params(alpha = a)
	lasso.fit(lr_train_x, lr_train_y)
	scores[i] = lasso.score(lr_validate_x, lr_validate_y)
	if scores[i] > max_score:
		max_score = scores[i]
		opt_a = a
		lasso_save = lasso

plt.plot(alphas, scores, color = 'b', linestyle = 'dashed', marker = 'o', markerfacecolor = 'blue', markersize = 6)
plt.xlabel('alpha')
plt.ylabel('score')
plt.grid(True)
plt.title('score vs. alpha')
plt.show()
print('The Optimized alpha and score of Lasso Linear is: ', opt_a, max_score)

#Combine the validate data and training data, use the optimal alpha, re-train the model:
lasso_f = linear_model.Lasso()
lasso_f.set_params(alpha = opt_a)
lasso_f.fit(model_train_x, model_train_y)

# 2. Polynomial features:

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(2)

poly_train, poly_validate = train_test_split(model_train, test_size = 0.15, random_state = 0)
poly_train_x = poly_train.drop(['video_id','cvt_per_day'], axis = 1)
poly_validate_x = poly_validate.drop(['video_id','cvt_per_day'], axis = 1)
poly_train_xp = poly.fit_transform(poly_train_x)
poly_validate_xp = poly.fit_transform(poly_validate_x)
poly_train_y = poly_train['cvt_per_day']
poly_validate_y = poly_validate['cvt_per_day']

alphas = np.logspace(-2.6, 2.5, num = 80)
scores = np.empty_like(alphas)
opt_a = float('-inf')
max_score = float('-inf')
for i,a in enumerate(alphas):
	lasso = linear_model.Lasso()
	lasso.set_params(alpha = a)
	lasso.fit(poly_train_xp, poly_train_y)
	scores[i] = lasso.score(poly_validate_xp, poly_validate_y)
	if scores[i] > max_score:
		max_score = scores[i]
		opt_a = a
		lasso_save = lasso

plt.plot(alphas, scores, color = 'b', linestyle = 'dashed', marker = 'o', markerfacecolor = 'blue', markersize = 6)
plt.xlabel('alpha')
plt.ylabel('score')
plt.grid(True)
plt.title('score vs. alpha Polynomial')
plt.show()
print('The Optimized alpha and score of Lasso Polynomial is: ', opt_a, max_score)

#Combine the validate data and training data, use the optimal alpha, re-train the model:
poly_train_xpl = poly.fit_transform(model_train_x)
lasso_fp = linear_model.Lasso()
lasso_fp.set_params(alpha = opt_a)
lasso_fp.fit(poly_train_xpl, model_train_y)

#3. Ridge Linear Regression:
ridge_train, ridge_validate = train_test_split(model_train, test_size = 0.15, random_state = 0)
alphas = np.logspace(-10, 3, num = 150)
scores = np.empty_like(alphas)
opt_a = float('-inf')
max_score = float('-inf')
for i, a in enumerate(alphas):
	ridge = linear_model.Ridge()
	ridge.set_params(alpha = a)
	ridge.fit(lr_train_x, lr_train_y)
	scores[i] = ridge.score(lr_validate_x, lr_validate_y)
	if scores[i] > max_score:
		max_score = scores[i]
		opt_a = a
		ridge_save = ridge
plt.plot(alphas, scores, color = 'r', linestyle = 'dashed', marker = 'o', markerfacecolor = 'r', markersize = 6)
plt.xlabel('alpha')
plt.ylabel('score')
plt.grid(True)
plt.title('Score vs. alpha Ridge')
plt.show()
print(max_score, opt_a)

#Combine the validate data and training data, use the optimal alpha, re-train the model:
ridge_f = linear_model.Ridge()
ridge_f.set_params(alpha = opt_a)
ridge_f.fit(model_train_x, model_train_y)

# 4. Ridge Polynomial features:
poly = PolynomialFeatures(2)

poly_train, poly_validate = train_test_split(model_train, test_size = 0.15, random_state = 0)
poly_train_x = poly_train.drop(['video_id','cvt_per_day'], axis = 1)
poly_validate_x = poly_validate.drop(['video_id','cvt_per_day'], axis = 1)
poly_train_xp = poly.fit_transform(poly_train_x)
poly_validate_xp = poly.fit_transform(poly_validate_x)
poly_train_y = poly_train['cvt_per_day']
poly_validate_y = poly_validate['cvt_per_day']

alphas = np.logspace(-2, 2,  num = 20)
scores = np.empty_like(alphas)
opt_a = float('-inf')
max_score = float('-inf')
for i,a in enumerate(alphas):
	ridge = linear_model.Ridge()
	ridge.set_params(alpha = a)
	ridge.fit(poly_train_xp, poly_train_y)
	scores[i] = ridge.score(poly_validate_xp, poly_validate_y)
	if scores[i] > max_score:
		max_score = scores[i]
		opt_a = a
		ridge_save = ridge

plt.plot(alphas, scores, color = 'brown', linestyle = 'dashed', marker = 'o', markerfacecolor = 'blue', markersize = 6)
plt.xlabel('alpha')
plt.ylabel('score')
plt.grid(True)
plt.title('score vs. alpha Polynomial')
plt.show()
print('The Optimized alpha and score of Ridge Polynomial is: ', opt_a, max_score)

#Combine the validate data and training data, use the optimal alpha, re-train the model:
poly_train_xpl = poly.fit_transform(model_train_x)
ridge_fp = linear_model.Ridge()
ridge_fp.set_params(alpha = opt_a)
ridge_fp.fit(poly_train_xpl, model_train_y)

"""
Non-Linear Model: Random Forest

Random Forest with GridSearch cross-validation is used. The 'mean-score' is used to narrow down the parameters of 
n_estimator(number of trees in the forest) and Max_Depth (maximum depth of the tree.)

"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

rf_train, rf_test = train_test_split(model_train, test_size = 0.15, random_state = 0)
rf_train_x = rf_train.drop(['video_id', 'cvt_per_day'], axis = 1)
rf_test_x = rf_test.drop(['video_id', 'cvt_per_day'], axis = 1)
rf_train_y = rf_train['cvt_per_day']
rf_test_y = rf_test['cvt_per_day']

param_grid = {
	'n_estimators': [54, 55, 56, 57, 58, 59, 60, 62],
	'max_depth': [12, 13, 14, 15, 16, 17]
}

rf = RandomForestRegressor(random_state = 2, max_features = 'sqrt')
grid_rf = GridSearchCV(rf, param_grid, cv = 5)
grid_rf.fit(rf_train_x, rf_train_y)

#Print Best Parameters:
print(grid_rf.best_params_)
print(grid_rf.cv_results_)

#Plot the effect of different number of trees and maximum tree-depth during cross validation.
scores = grid_rf.cv_results_['mean_test_score']

n_estimators = [54, 55, 56, 57, 58, 59, 60, 62]
m_depth = [12, 13, 14, 15, 16, 17]
scores = np.array(scores).reshape(len(m_depth), len(n_estimators))
fig = plt.figure()
ax = plt.subplot(111)
for ind, i in enumerate(m_depth):
	plt.plot(n_estimators, scores[ind], '-o', label = 'n estimator.' + str(i),)

ax.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
plt.xlabel('Mean score')
plt.ylabel('Max Depth')
plt.title('Random_Forest_Mean_Score')
plt.grid(True)
plt.show()

#Add both training and validation data together as the new training data.
rf = RandomForestRegressor(random_state = 2, max_features = 'sqrt', max_depth = 14, n_estimators = 55)
rf.fit(model_train_x, model_train_y)

"""
Model Evaluation:
Test data is the reserved 15% of the whole dataset, and has never been by the above models.

1. Lasso test with linear features
2. Lasso test with polynomial features
3. ridge test with linear features
4. ridge test with polynomial features
5. Random Forest test(rf)

"""
# 1. Lasso test with linear features(lasso_f)
lasso_f_score = lasso_f.score(model_test_x, model_test_y)
pred_y = lasso_f.predict(model_test_x)

#The mean squared error and Root mean square error:
MSE_lasso_f = mean_squared_error(model_test_y, pred_y)
RMSE_lasso_f = sqrt(mean_squared_error(model_test_y, pred_y))

print("lasso_f score: ", lasso_f_score)
print("Mean square error of lasso_f: ", MSE_lasso_f)
print("Root mean squared error of lasso_f: ", RMSE_lasso_f)
print("Coefficients of lasso_f: ", lasso_f.coef_)

#2. Lasso test with polynomial features(lasso_fp)
model_test_xp = poly.fit_transform(model_test_x)
pred_y = lasso_fp.predict(model_test_xp)

lasso_fp_score = lasso_fp.score(model_test_xp, model_test_y)
MSE_lasso_fp = mean_squared_error(model_test_y, pred_y)
RMSE_lasso_fp = sqrt(mean_squared_error(model_test_y, pred_y))

print("lasso_fp score: ", lasso_fp_score)
print("Mean square error of lasso_fp: ", MSE_lasso_fp)
print("Root mean squared error of lasso_fp: ", RMSE_lasso_fp)
print("Coefficients of lasso_fp: ", lasso_fp.coef_)

#3. Ridge test with linear features:
ridge_f_score = ridge_f.score(model_test_x, model_test_y)
pred_y = ridge_f.predict(model_test_x)

#The Mean Squared Error and Root Mean Square Error:
MSE_ridge_f = mean_squared_error(model_test_y, pred_y)
RMSE_ridge_f = sqrt(mean_squared_error(model_test_y, pred_y))

print("ridge_f score: ", ridge_f_score)
print("Mean square error of ridge_f: ", MSE_ridge_f)
print("Root mean squared error of ridge_f: ", RMSE_ridge_f)
print("Coefficients of ridge_f: ", ridge_f.coef_)

#4. Ridge Test with polynomial features(ridge_fp):
model_test_xp = poly.fit_transform(model_test_x)
pred_y = ridge_fp.predict(model_test_xp)

ridge_fp_score = ridge_fp.score(model_test_xp, model_test_y)
MSE_ridge_fp = mean_squared_error(model_test_y, pred_y)
RMSE_ridge_fp = sqrt(mean_squared_error(model_test_y, pred_y))

print("ridge_fp score: ", ridge_fp_score)
print("Mean square error of ridge_fp: ", MSE_ridge_fp)
print("Root mean squared error of ridge_fp: ", RMSE_ridge_fp)
print("Coefficients of ridge_fp: ", ridge_fp.coef_)

#5. Random Forest test(rf)
rf_score = rf.score(model_test_x, model_test_y)
pred_y = rf.predict(model_test_x)
MSE_rf = mean_squared_error(model_test_y, pred_y)
RMSE_rf = sqrt(mean_squared_error(model_test_y, pred_y))

#Print out the Mean square error and root mean square error:
print('rf score: ', rf_score)
print('Mean square error of rf: ', MSE_rf)
print('Root mean squared error of rf: ', RMSE_rf)
print("Coefficients of rf:", ridge_fp.coef_)

"""
Comparsion of 5 models:
1. Max_score
2. Mean Squared error(MSE)
3. Root Mean Squared Error(RMSE)

Conclusion: random forest model returns the best prediction accuracy

"""

lst_score = [lasso_f_score, lasso_fp_score, ridge_f_score, ridge_fp_score, rf_score]
MSE_lst = [MSE_lasso_f, MSE_lasso_fp, MSE_ridge_f, MSE_ridge_fp, MSE_rf]
RMSE_lst = [RMSE_lasso_f, RMSE_lasso_fp, RMSE_ridge_f, RMSE_ridge_fp, RMSE_rf]
model_lst= ['Lasso_linear', 'Lasso polynomial', 'Ridge_linear', 'Ridge poly', 'Random Forest']

plt.figure(1)
plt.plot(model_lst, lst_score, 'ro')
plt.legend(loc = 9)
plt.legend(['r-square/score'])
plt.xlabel('model names', fontsize = 16)
plt.ylabel('score/r square', fontsize = 16)
plt.grid(True)
plt.show()

plt.figure(2)
plt.plot(model_lst, MSE_lst, 'g^')
plt.legend(loc = 9)
plt.legend(['Mean square error(MSE)'])
plt.xlabel('model names', fontsize = 16)
plt.ylabel('mean square error', fontsize =16)
plt.grid(True)
plt.show()

plt.figure(3)
plt.plot(model_lst, RMSE_lst, 'bs')
plt.legend(loc = 9)
plt.legend(['Root Mean Square Error(RMSE)'])
plt.xlabel('model names', fontsize = 16)
plt.ylabel('root mean square error', fontsize = 16)
plt.grid(True)
plt.show()

"""
Feature importance:

Random Forest(RF) shows the best prediction accuracy. Therefore, the feature importance 
will be extracted from the RF Model.

"""
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis = 0)
indices = np.argsort(importances)[::-1]
feature_name = model_test_x.columns.get_values()

#Print Feature Rankings:
print("Feature importance Ranking:")

for f in range(model_test_x.shape[1]):
	print("%d. feature %d %s (%f)" %(f+1, indices[f], feature_name[f], importances[indices[f]]))

plt.figure(1)
plt.bar(feature_name[:11], importances[indices[:11]])
plt.xticks(rotation = 90)
plt.title('Feature Importance Ranking')
plt.xlabel('Feature Names')
plt.ylabel('Importances level')
plt.show()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 11:42:41 2019

@author: Team 1
"""

###############################################################################
#1 Initial Exploratory Data Analysis
###############################################################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#file = 'birthweight.xlsx'
file1 = 'birthweight_feature_set.xlsx'

baby = pd.read_excel(file1)

print(baby.columns)
print(baby.head())
print(baby.shape)
print(baby.info())
print(baby.describe().round(2))

#Missing values
print(baby.isnull().sum())
total_baby_rows = baby.shape[0]
missing_baby = baby.isnull().sum()

missing_ratio = (missing_baby / total_baby_rows).multiply(100)
print (missing_ratio.round(2))
type (missing_ratio)

'''
meduc      30
monpre      5
npvis      68
fage        6
feduc      47
omaps       3
fmaps       3
cigs      110
drink     115
'''

'''
for col in baby:
    if baby[col].isnull().any():
        baby['m_'+col] = baby[col].isnull().astype(int)
'''

###############################################################################
#2 Missing Value Imputation
###############################################################################

#Histograms before imputation – to check the skewness so that the col can be imputed accordingly
null_cols = baby.columns[baby.isna().any()].tolist()

for col in null_cols:
    plt.figure()
    sns.distplot(baby[col], hist=False, rug=True)
      
#Loop to imput the median of the Col for each mssing value
for col in baby:
    if baby[col].isnull().any():
        baby[col] = baby[col].fillna(baby[col].median())

print(baby.isnull().sum())

###############################################################################
#3 Outlier Analysis
###############################################################################

baby1= pd.DataFrame.copy(baby)
for col in null_cols:
    plt.figure()
    sns.distplot(baby1[col], hist=False, rug=True)
    
##### 3.1 Quantiles and type of variable classification
baby_quantiles = baby1.loc[:, :].quantile([0.20,0.40,0.60,0.80,1.00])
print(baby_quantiles)                                                
                                                                                                
for col in baby1:
    print(col)          

'''
Assumed Continuous/Interval Variables - 
mage
meduc
monpre
npvis
fage
feduc
omaps
fmaps
cigs
drink
bwght

Assumed Categorical -
NA

Binary Classifiers -
male
mwhte
mblck
moth
fwhte
fblck
foth

'''


##### 3.2 Visual EDA (Histograms) to see the distributions and determine Min and Max.

plt.subplot(2, 2, 1)
sns.distplot(baby1['mage'],
#             bins = 35,
             color = 'g')

plt.xlabel('Mother Age')


########################


plt.subplot(2, 2, 2)
sns.distplot(baby1['meduc'],
#             bins = 30,
             color = 'y')

plt.xlabel('Mother Education')



########################


plt.subplot(2, 2, 3)
sns.distplot(baby1['monpre'],
#             bins = 17,
             kde = False,
             rug = True,
             color = 'orange')

plt.xlabel('Month Prenatal Care Began')



########################


plt.subplot(2, 2, 4)

sns.distplot(baby1['npvis'],
#             bins = 17,
             kde = False,
             rug = True,
             color = 'r')

plt.xlabel('Total Number Of Prenatal Visits')



plt.tight_layout()
plt.savefig('Baby Data Histograms 1 of 5.png')

plt.show()


########################
########################


plt.subplot(2, 2, 1)
sns.distplot(baby1['fage'],
#             bins = 35,
             color = 'g')

plt.xlabel('Father Age')


########################


plt.subplot(2, 2, 2)
sns.distplot(baby1['feduc'],
#             bins = 30,
             color = 'y')

plt.xlabel('Father Education')



########################


plt.subplot(2, 2, 3)
sns.distplot(baby1['omaps'],
#             bins = 17,
             kde = False,
             rug = True,
             color = 'orange')

plt.xlabel('One Minute apgar Score')



########################


plt.subplot(2, 2, 4)

sns.distplot(baby1['fmaps'],
#             bins = 17,
             kde = False,
             rug = True,
             color = 'r')

plt.xlabel('Five Minute apgar Score')



plt.tight_layout()
plt.savefig('Baby Data Histograms 2 of 5.png')

plt.show()


########################
########################


plt.subplot(1, 2, 1)
sns.distplot(baby1['cigs'],
#             bins = 35,
             color = 'g')

plt.xlabel('Avg Cigarettes Per Day')

plt.subplot(1, 2, 2)
sns.distplot(baby1['drink'],
#             bins = 35,
             color = 'g')

plt.xlabel('Avg Drinks Per Week')

plt.tight_layout()
plt.savefig('Baby Data Histograms 3 of 5.png')

plt.show()



col_list = ['mage', 
            'meduc', 
            'monpre', 
            'npvis', 
            'fage', 
            'feduc', 
            'omaps', 
            'fmaps', 
            'cigs', 
            'drink', 
            'bwght']

for col in col_list:
    
    sns.boxplot(y = 'male',
                  x = col,
                  data = baby1,                         
                  orient ="h",
                  meanline = True,
                  showmeans = True)
                     
                                          
    plt.suptitle('')
    plt.tight_layout()
    plt.show()

for cols in col_list:
    plt.figure()
    g = sns.FacetGrid(baby1, col='male')
    g.map(plt.hist, cols);   
    

##### 3.3 Outlier Flagging based on distributions and boxplots above. #########
    
my_dict = {
	"mage": ['None','None'],
    "meduc": [4,'None'],
    "monpre": ['None',6],
    "npvis": [4,30],
    "fage": ['None',50],
    "feduc": [4,'None'],
    "omaps": ['None','None'], 
    "fmaps": ['None','None'],
    "cigs": ['None','None'],
    "drink": ['None','None'],
    "bwght": [1000,'None']
};    

for col in col_list:
    plt.figure()

    sns.distplot(baby1[col], hist=False, rug=True)

    
    if (my_dict[col][0] != 'None'):
        plt.axvline(x = my_dict[col][0],
            label = 'Outlier Thresholds',
            linestyle = '--',
            color = 'red')
   
    if (my_dict[col][1] != 'None'):
        plt.axvline(x = my_dict[col][1],
            label = 'Outlier Thresholds',
            linestyle = '--',
            color = 'red')   

baby2= pd.DataFrame.copy(baby1)


##### 3.4 Creation of new columns with outliers. ##############################

#from pandas.api.types import is_numeric_dtype
for col in col_list:
    #print(is_numeric_dtype(baby1[col]))
    if (my_dict[col][0] != 'None'):
        baby2['o_'+col] = baby2[col].apply(lambda val: -1 if val < my_dict[col][0] else 0)
    if (my_dict[col][1] != 'None'):
        baby2['o_'+col] = baby2[col].apply(lambda val: 1 if val > my_dict[col][1] else 0)
    
print(baby2.shape)
for col in baby2:
    print(col)   
    

    
###############################################################################
# 4 Correlation Analysis (INITIAL)
###############################################################################

##### 4.1 Correlations ########################################################


baby3= pd.DataFrame.copy(baby1)
baby3.head()
df_corr = baby3.corr().round(2)


print(df_corr)
df_corr.loc['bwght'].sort_values(ascending = False)



##### 4.2 Correlation Heatmap ################################################

# Using palplot to view a color scheme
sns.palplot(sns.color_palette('coolwarm', 12))

fig, ax = plt.subplots(figsize=(18,18))


df_corr2 = df_corr.iloc[0:21, 0:21]

sns.heatmap(df_corr2,
            cmap = 'coolwarm',
            square = True,
            annot = True,
            linecolor = 'black',
            linewidths = 0.5)


plt.savefig('Baby Weight Correlation Heatmap.png')
plt.show()

baby3.to_excel('Baby_explored1.xlsx')
file = 'Baby_explored1.xlsx'
                        
baby4 = pd.read_excel(file)


###############################################################################
# 5 Base Model (NO FEATURE ENGINEERING)
###############################################################################

from sklearn.model_selection import train_test_split # train/test split
from sklearn.neighbors import KNeighborsRegressor # KNN for Regression
import statsmodels.formula.api as smf # regression modeling
import sklearn.metrics # more metrics for model performance evaluation
from sklearn.model_selection import cross_val_score # k-folds cross validation

###### 5.1 Multivariate Regression Model (BASE MODEL) #########################


#We observed from the heatmap the variables that had the most correlation with
#Baby weight and made sure they correspond to our expternal research as well.
#These variables are represented below:

lm_fmaps = smf.ols(formula = """bwght ~                         
                           baby4['cigs'] +
                           baby4['drink'] +
                           baby4['fage'] +
                           baby4['feduc'] +
                           baby4['mage']
                           """,
                         data = baby4)

results = lm_fmaps.fit()
print(results.summary())
print(f"""
Parameters:
{results.params.round(2)}

Summary Statistics:
R-Squared:          {results.rsquared.round(3)}
Adjusted R-Squared: {results.rsquared_adj.round(3)}
""")

predict = results.predict()
y_hat   = pd.DataFrame(predict).round(2)
resids  = results.resid.round(2)

# Plotting residuals
residual_analysis = pd.concat(
        [baby4.loc[:,'bwght'],
         y_hat,
         results.resid.round(2)],
         axis = 1)

residual_analysis.to_excel('Baby Residuals.xlsx')

sns.residplot(x = predict,
              y = baby4.loc[:,'bwght'])


plt.show()



###### 5.2 Generalization using Train/Test Split (BASE MODEL) #################

#We have made sure to drop the variables that we haven't chosen above.
baby_data   = baby4.drop(['bwght',
                          'omaps',
                          'fmaps',
                          'meduc',
                            'monpre',
                            'npvis',
                            'male',
                            'mwhte',
                            'mblck',
                            'moth',
                            'fwhte',
                            'fblck',
                            'foth',],
                                axis = 1)

baby_target = baby4.loc[:, 'bwght']


X_train, X_test, y_train, y_test = \
    train_test_split(baby_data,
                     baby_target,                           
                     test_size = 0.10,
                     random_state = 508)


print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)



###### 5.3 Using KNN  on the optimal model (BASE MODEL) #######################


training_accuracy = []
test_accuracy = []

neighbors_settings = range(1, 51)


for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsRegressor(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train)
    
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))


plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()



print(test_accuracy)

print("The optimal number of neighbors is at index", \
      test_accuracy.index(max(test_accuracy)), \
      "with an optimal score of", \
      max(test_accuracy))


###### 5.4 KNN with Optimized Number of Neighbors (BASE MODEL) ################
#Based on section 5.3, the best results for the BASE MODEL occur when k = 13 

# Building a model with k = 13
knn_reg = KNeighborsRegressor(algorithm = 'auto',
                              n_neighbors = 13)

# Fitting the model based on the training data
knn_reg_fit = knn_reg.fit(X_train, y_train)

# Scoring the model
y_score_knn_optimal = knn_reg.score(X_test, y_test)

# The score is directly comparable to R-Square
print(y_score_knn_optimal)

# Generating Predictions based on the optimal KNN model
#knn_reg_optimal_pred = knn_reg_fit.predict(X_test)

# Predicting on the X_data that the model has never seen before
y_pred = knn_reg.predict(X_test)

# Printing out prediction values for each test observation
print(f"""
Test set predictions:
{y_pred}
""")
    
# Calling the score method, which compares the predicted values to the actual
# values
y_score = knn_reg.score(X_test, y_test)

# The score is directly comparable to R-Square
print(y_score)    


###### 5.5 Prediction. Does OLS predict better than KNN? (BASE MODEL) #########


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr_fit = lr.fit(X_train, y_train)
lr_pred = lr_fit.predict(X_test)


print(f"""
Test set predictions:
{y_pred.round(2)}
""")

# Scoring the model
y_score_ols_optimal = lr_fit.score(X_test, y_test)

# The score is directly comparable to R-Square
print(y_score_ols_optimal)


# Let's compare the testing score to the training score.

print('Training Score', lr.score(X_train, y_train).round(4))
print('Testing Score:', lr.score(X_test, y_test).round(4))

# Printing model results
print(f"""
Full model KNN score:    {y_score.round(3)}
Optimal model KNN score: {y_score_knn_optimal.round(3)}
Optimal model OLS score: {y_score_ols_optimal.round(3)}
""")



###### 5.6 Outputting Model Coefficients, Predictions, and Other Metrics (BASE MODEL)


# What does our leading model look like?
pd.DataFrame(list(zip(baby_data.columns, lr.coef_)))


# How well are we predicting on each observation?
pd.DataFrame(list(zip(y_test, lr_pred)))


# R-Square (same as the score above)
lr_rsq = sklearn.metrics.r2_score(y_test, lr_pred)
print(lr_rsq)


# Mean Squared Error
lr_mse = sklearn.metrics.mean_squared_error(y_test, lr_pred)
print(lr_mse)


# Root Mean Squared Error (how far off are we on each observation?)
lr_rmse = pd.np.sqrt(lr_mse)
print(lr_rmse)


###############################################################################
# 6 Feature Engineering
###############################################################################
import numpy as np

#Our base model was good but we want to see if by creating new variables
#through feature engineering we can increase our overall Testing Score.
#For this, we created 3 new variables also based on our external research:


# 'presponsible' to corresponds to parents that have been responsible in the 
# prenatal care of their child. So, parents that started the prenatal care in
# month 1 AND have a minimum of 12 visits of prenatal vistis will be considered
# as Responsible.
baby1['presponsible'] = np.where( (baby1['monpre']==1) & (baby1['npvis']>=12 ), 1,0)


# 'pblack' refers to if BOTH parents are black. From our external research we 
# observed that black parents could have babies with lower weight.
baby1['pblack'] = np.where( (baby1['mblck']==1) & (baby1['fblck']==1 ), 1,0)


# 'oldmom' refers to the mother that have babies at extreme risk age, which is
# more than 50 years old. From the Base Model, we realized that Mother's Age
# 'mage' as a whole is not quite siginificant. But we know from external
# reserach that as the mother's age preogresses, the baby's health is more at 
# risk, potentially affecting the weight (lower weights). To confir this we 
# first created a Scatterplot:

baby.plot(kind='scatter',x='mage',y='bwght')
plt.xlabel('Mother Age')
plt.ylabel('Baby Weight (grams)')
plt.axvline(50, color='red')
plt.savefig('Scatterplot bwght+mage.png')

#The scatterplot clearly shows how at the mother's age of 50 (approx) the 
#baby weight starts to decrease. Becasue of this, we have created the following
#varaible.
baby1['oldmom'] = np.where( (baby1['mage']>50), 1,0)


###############################################################################
# 7 New Correlation Analysis (WITH FEATURE ENGINEERING)
###############################################################################

# After creating our 3 new featured engineered variables, we ran again our 
# correlations in order to update them.

##### 7.1 Correlations ########################################################


baby3= pd.DataFrame.copy(baby1)
baby3.head()
df_corr = baby3.corr().round(2)


print(df_corr)
df_corr.loc['bwght'].sort_values(ascending = False)



##### 7.2 Correlation Heatmap ################################################

# Using palplot to view a color scheme
sns.palplot(sns.color_palette('coolwarm', 12))

fig, ax = plt.subplots(figsize=(18,18))


df_corr2 = df_corr.iloc[0:21, 0:21]

sns.heatmap(df_corr2,
            cmap = 'coolwarm',
            square = True,
            annot = True,
            linecolor = 'black',
            linewidths = 0.5)


plt.savefig('Baby Weight Correlation Heatmap.png')
plt.show()

baby3.to_excel('Baby_explored1.xlsx')
file = 'Baby_explored1.xlsx'
                        
baby4 = pd.read_excel(file)

# As believed, in the correlation heatmap we can see that 
# our new variable 'oldmom' has negative correlation of 0.5, which confirms 
# that affects the baby weight.

###############################################################################
# 8 Model A (WITH FEATURE ENGINEERING)
###############################################################################

# This a new model including ALL of the feature engineered variables to 
# see if we improve our testing score. We keep the original variables:
    # 'cigs'
    # 'drink'
    # 'fage'
    # 'feduc'
    
# We took out the original variable 'mage' to include the new variable 'oldmom'.

###### 8.1 Multivariate Regression Model (Model A) ############################


lm_fmaps = smf.ols(formula = """bwght ~                         
                           baby4['cigs'] +
                           baby4['drink'] +
                           baby4['fage'] +
                           baby4['feduc'] +
                           baby4['presponsible'] +
                           baby4['pblack'] +
                           baby4['oldmom']
                           """,
                         data = baby4)

results = lm_fmaps.fit()
print(results.summary())
print(f"""
Parameters:
{results.params.round(2)}

Summary Statistics:
R-Squared:          {results.rsquared.round(3)}
Adjusted R-Squared: {results.rsquared_adj.round(3)}
""")

predict = results.predict()
y_hat   = pd.DataFrame(predict).round(2)
resids  = results.resid.round(2)

# Plotting residuals
residual_analysis = pd.concat(
        [baby4.loc[:,'bwght'],
         y_hat,
         results.resid.round(2)],
         axis = 1)

residual_analysis.to_excel('Baby Residuals.xlsx')

sns.residplot(x = predict,
              y = baby4.loc[:,'bwght'])


plt.show()



###### 8.2 Generalization using Train/Test Split (Model A) ###################

baby_data   = baby4.drop(['bwght',
                          'omaps',
                          'fmaps',
                          'meduc',
                            'monpre',
                            'npvis',
                            'male',
                            'mwhte',
                            'mblck',
                            'moth',
                            'fwhte',
                            'fblck',
                            'foth',
                            'mage'],
                                axis = 1)

baby_target = baby4.loc[:, 'bwght']


X_train, X_test, y_train, y_test = \
    train_test_split(baby_data,
                     baby_target,                           
                     test_size = 0.10,
                     random_state = 508)


print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)



###### 8.3 Using KNN  on the optimal model (MODEL A) ##########################


training_accuracy = []
test_accuracy = []

neighbors_settings = range(1, 51)


for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsRegressor(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train)
    
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))


plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()



print(test_accuracy)

print("The optimal number of neighbors is at index", \
      test_accuracy.index(max(test_accuracy)), \
      "with an optimal score of", \
      max(test_accuracy))


###### 8.4 KNN with Optimized Number of Neighbors (MODEL A) ###################
#The best results occur when k = 13 

# Building a model with k = 13
knn_reg = KNeighborsRegressor(algorithm = 'auto',
                              n_neighbors = 13)

# Fitting the model based on the training data
knn_reg_fit = knn_reg.fit(X_train, y_train)

# Scoring the model
y_score_knn_optimal = knn_reg.score(X_test, y_test)

# The score is directly comparable to R-Square
print(y_score_knn_optimal)

# Generating Predictions based on the optimal KNN model
#knn_reg_optimal_pred = knn_reg_fit.predict(X_test)

# Predicting on the X_data that the model has never seen before
y_pred = knn_reg.predict(X_test)

# Printing out prediction values for each test observation
print(f"""
Test set predictions:
{y_pred}
""")
    
# Calling the score method, which compares the predicted values to the actual
# values
y_score = knn_reg.score(X_test, y_test)

# The score is directly comparable to R-Square
print(y_score)    


###### 8.5 Prediction. Does OLS predict better than KNN? (MODEL A) ############


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr_fit = lr.fit(X_train, y_train)
lr_pred = lr_fit.predict(X_test)


print(f"""
Test set predictions:
{y_pred.round(2)}
""")

# Scoring the model
y_score_ols_optimal = lr_fit.score(X_test, y_test)

# The score is directly comparable to R-Square
print(y_score_ols_optimal)


# Let's compare the testing score to the training score.

print('Training Score', lr.score(X_train, y_train).round(4))
print('Testing Score:', lr.score(X_test, y_test).round(4))

# Printing model results
print(f"""
Full model KNN score:    {y_score.round(3)}
Optimal model KNN score: {y_score_knn_optimal.round(3)}
Optimal model OLS score: {y_score_ols_optimal.round(3)}
""")



###### 8.6 Outputting Model Coefficients, Predictions, and Other Metrics (MODEL A)


# What does our leading model look like?
pd.DataFrame(list(zip(baby_data.columns, lr.coef_)))


# How well are we predicting on each observation?
pd.DataFrame(list(zip(y_test, lr_pred)))


# R-Square (same as the score above)
lr_rsq = sklearn.metrics.r2_score(y_test, lr_pred)
print(lr_rsq)


# Mean Squared Error
lr_mse = sklearn.metrics.mean_squared_error(y_test, lr_pred)
print(lr_mse)


# Root Mean Squared Error (how far off are we on each observation?)
lr_rmse = pd.np.sqrt(lr_mse)
print(lr_rmse)

###### 8.7 Final Results (MODEL A) ############################################
# The overall scores improves with Model A. Compared to the Base Model (secion 5):
    # R-squared increased from 0.712 (BASE) to 0.720 (MODEL A)
    # Training Score increased from 0.716 (BASE) to 0.722 (MODEL A)
    # Testing Score increased from 0.651 (BASE) to 0.661 (MODEL A)
    

###############################################################################
# 9 Model B (WITH FEATURE ENGINEERING)
###############################################################################

# In the previous model we used all of the 3 new variables created. In this
# model we explude 'plack' (if both parents are black) variable to determine
# if we get better resutls. We exclude it becasue fo a low correlation.

###### 9.1 Multivariate Regression Model (MODEL B) ############################


lm_fmaps = smf.ols(formula = """bwght ~                         
                           baby4['cigs'] +
                           baby4['drink'] +
                           baby4['fage'] +
                           baby4['feduc'] +
                           baby4['presponsible'] +
                           baby4['oldmom']
                           """,
                         data = baby4)

results = lm_fmaps.fit()
print(results.summary())
print(f"""
Parameters:
{results.params.round(2)}

Summary Statistics:
R-Squared:          {results.rsquared.round(3)}
Adjusted R-Squared: {results.rsquared_adj.round(3)}
""")

predict = results.predict()
y_hat   = pd.DataFrame(predict).round(2)
resids  = results.resid.round(2)

# Plotting residuals
residual_analysis = pd.concat(
        [baby4.loc[:,'bwght'],
         y_hat,
         results.resid.round(2)],
         axis = 1)

residual_analysis.to_excel('Baby Residuals.xlsx')

sns.residplot(x = predict,
              y = baby4.loc[:,'bwght'])


plt.show()



###### 9.2 Generalization using Train/Test Split (MODEL B) ####################

baby_data   = baby4.drop(['bwght',
                          'omaps',
                          'fmaps',
                          'meduc',
                            'monpre',
                            'npvis',
                            'male',
                            'mwhte',
                            'mblck',
                            'moth',
                            'fwhte',
                            'fblck',
                            'foth',
                            'mage',
                            'pblack'],
                                axis = 1)

baby_target = baby4.loc[:, 'bwght']


X_train, X_test, y_train, y_test = \
    train_test_split(baby_data,
                     baby_target,                           
                     test_size = 0.10,
                     random_state = 508)


print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)



###### 9.3 Using KNN  on the optimal model (MODEL B) ##########################


training_accuracy = []
test_accuracy = []

neighbors_settings = range(1, 51)


for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsRegressor(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train)
    
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))


plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()



print(test_accuracy)

print("The optimal number of neighbors is at index", \
      test_accuracy.index(max(test_accuracy)), \
      "with an optimal score of", \
      max(test_accuracy))


###### 9.4 KNN with Optimized Number of Neighbors (MODEL B) ###################
#The best results occur when k = 14 

# Building a model with k = 14
knn_reg = KNeighborsRegressor(algorithm = 'auto',
                              n_neighbors = 14)

# Fitting the model based on the training data
knn_reg_fit = knn_reg.fit(X_train, y_train)

# Scoring the model
y_score_knn_optimal = knn_reg.score(X_test, y_test)

# The score is directly comparable to R-Square
print(y_score_knn_optimal)

# Generating Predictions based on the optimal KNN model
#knn_reg_optimal_pred = knn_reg_fit.predict(X_test)

# Predicting on the X_data that the model has never seen before
y_pred = knn_reg.predict(X_test)

# Printing out prediction values for each test observation
print(f"""
Test set predictions:
{y_pred}
""")
    
# Calling the score method, which compares the predicted values to the actual
# values
y_score = knn_reg.score(X_test, y_test)

# The score is directly comparable to R-Square
print(y_score)    


###### 9.5 Prediction. Does OLS predict better than KNN? (MODEL B) ############


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr_fit = lr.fit(X_train, y_train)
lr_pred = lr_fit.predict(X_test)


print(f"""
Test set predictions:
{y_pred.round(2)}
""")

# Scoring the model
y_score_ols_optimal = lr_fit.score(X_test, y_test)

# The score is directly comparable to R-Square
print(y_score_ols_optimal)


# Let's compare the testing score to the training score.

print('Training Score', lr.score(X_train, y_train).round(4))
print('Testing Score:', lr.score(X_test, y_test).round(4))

# Printing model results
print(f"""
Full model KNN score:    {y_score.round(3)}
Optimal model KNN score: {y_score_knn_optimal.round(3)}
Optimal model OLS score: {y_score_ols_optimal.round(3)}
""")



###### 9.6 Outputting Model Coefficients, Predictions, and Other Metrics (MODEL B)


# What does our leading model look like?
pd.DataFrame(list(zip(baby_data.columns, lr.coef_)))


# How well are we predicting on each observation?
pd.DataFrame(list(zip(y_test, lr_pred)))


# R-Square (same as the score above)
lr_rsq = sklearn.metrics.r2_score(y_test, lr_pred)
print(lr_rsq)


# Mean Squared Error
lr_mse = sklearn.metrics.mean_squared_error(y_test, lr_pred)
print(lr_mse)


# Root Mean Squared Error (how far off are we on each observation?)
lr_rmse = pd.np.sqrt(lr_mse)
print(lr_rmse)

###### 9.7 Final Results (MODEL B) ############################################
# With Model B, only the Testin score improved, but not for much.
# Compared to Model A (secion 8):
    # R-squared remained the same at 0.720
    # Training Score remained the same at 0.722
    # Testing Score increased from 0.661 (MODEL A) to 0.665 (MODEL A)


###############################################################################
# 10 Model C (WITH FEAUTRE ENGINEERING)
###############################################################################

# The reusltus obtained in the prevuous model barely improved. Becasue of this
# we created a final model that includes the 2 new variables that based on the 
# external research should have the highest impact.
#
# We included 'oldmom' and 'pblack' to see if a combination of variables that
# take into account high-risk age for mothers and parent's race will have a 
# higher impact.

###### 10.1 Multivariate Regression Model (MODEL C) ###########·###############


lm_fmaps = smf.ols(formula = """bwght ~                         
                           baby4['cigs'] +
                           baby4['drink'] +
                           baby4['fage'] +
                           baby4['feduc'] +
                           baby4['pblack'] +
                           baby4['oldmom']
                           """,
                         data = baby4)

results = lm_fmaps.fit()
print(results.summary())
print(f"""
Parameters:
{results.params.round(2)}

Summary Statistics:
R-Squared:          {results.rsquared.round(3)}
Adjusted R-Squared: {results.rsquared_adj.round(3)}
""")

predict = results.predict()
y_hat   = pd.DataFrame(predict).round(2)
resids  = results.resid.round(2)

# Plotting residuals
residual_analysis = pd.concat(
        [baby4.loc[:,'bwght'],
         y_hat,
         results.resid.round(2)],
         axis = 1)

residual_analysis.to_excel('Baby Residuals.xlsx')

sns.residplot(x = predict,
              y = baby4.loc[:,'bwght'])


plt.show()



###### 10.2 Generalization using Train/Test Split (MODEL C) ###################

baby_data   = baby4.drop(['bwght',
                          'omaps',
                          'fmaps',
                          'meduc',
                            'monpre',
                            'npvis',
                            'male',
                            'mwhte',
                            'mblck',
                            'moth',
                            'fwhte',
                            'fblck',
                            'foth',
                            'presponsible',
                            'mage'],
                                axis = 1)

baby_target = baby4.loc[:, 'bwght']


X_train, X_test, y_train, y_test = \
    train_test_split(baby_data,
                     baby_target,                           
                     test_size = 0.10,
                     random_state = 508)


print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)



###### 10.3 Using KNN  on the optimal model (MODEL C) #########################


training_accuracy = []
test_accuracy = []

neighbors_settings = range(1, 51)


for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsRegressor(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train)
    
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))


plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()



print(test_accuracy)

print("The optimal number of neighbors is at index", \
      test_accuracy.index(max(test_accuracy)), \
      "with an optimal score of", \
      max(test_accuracy))


###### 10.4 KNN with Optimized Number of Neighbors (MODEL C) ##################
#The best results occur when k = 14

# Building a model with k = 14
knn_reg = KNeighborsRegressor(algorithm = 'auto',
                              n_neighbors = 14)

# Fitting the model based on the training data
knn_reg_fit = knn_reg.fit(X_train, y_train)

# Scoring the model
y_score_knn_optimal = knn_reg.score(X_test, y_test)

# The score is directly comparable to R-Square
print(y_score_knn_optimal)

# Generating Predictions based on the optimal KNN model
knn_reg_optimal_pred = knn_reg_fit.predict(X_test)

# Predicting on the X_data that the model has never seen before
y_pred = knn_reg.predict(X_test)

# Printing out prediction values for each test observation
print(f"""
Test set predictions:
{y_pred}
""")
    
# Calling the score method, which compares the predicted values to the actual
# values
y_score = knn_reg.score(X_test, y_test)

# The score is directly comparable to R-Square
print(y_score)    


###### 10.5 Prediction. Does OLS predict better than KNN? (MODEL C) ###########


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr_fit = lr.fit(X_train, y_train)
lr_pred = lr_fit.predict(X_test)


print(f"""
Test set predictions:
{y_pred.round(2)}
""")

# Scoring the model
y_score_ols_optimal = lr_fit.score(X_test, y_test)

# The score is directly comparable to R-Square
print(y_score_ols_optimal)


# Let's compare the testing score to the training score.

print('Training Score:', lr.score(X_train, y_train).round(4))
print('Testing Score:', lr.score(X_test, y_test).round(4))

# Printing model results
print(f"""
Full model KNN score:    {y_score.round(3)}
Optimal model KNN score: {y_score_knn_optimal.round(3)}
Optimal model OLS score: {y_score_ols_optimal.round(3)}
""")



###### 10.6 Outputting Model Coefficients, Predictions, and Other Metrics (MODEL C)


# What does our leading model look like?
pd.DataFrame(list(zip(baby_data.columns, lr.coef_)))


# How well are we predicting on each observation?
pd.DataFrame(list(zip(y_test, lr_pred)))


# R-Square (same as the score above)
lr_rsq = sklearn.metrics.r2_score(y_test, lr_pred)
print(lr_rsq)


# Mean Squared Error
lr_mse = sklearn.metrics.mean_squared_error(y_test, lr_pred)
print(lr_mse)


# Root Mean Squared Error (how far off are we on each observation?)
lr_rmse = pd.np.sqrt(lr_mse)
print(lr_rmse)


###### 10.7 Final Results (MODEL C) ############################################
# The Testing Score improved with Model C. THIS IS OUR BEST MODEL.
# Compared to Model B (secion 9):
    # R-squared remained the same at 0.720
    # Training Score remained the same at 0.722
    # Testing Score increased from 0.665 (MODEL B) to 0.675 (MODEL C)


###### 10.8 Storing Model Predictions and Summary (MODEL C) ###################

# We can store our predictions as a dictionary.
model_predictions_df = pd.DataFrame({'Actual' : y_test,
                                     'KNN_Predicted': knn_reg_optimal_pred,
                                     'OLS_Predicted': lr_pred})



model_predictions_df.to_excel('A1 Final Model Predictions - Team 1 - Section 3.xlsx')




model_results = pd.DataFrame({'Model' : ['Full model KNN score', 'Optimal model KNN score', 'Optimal model OLS score'],
                              'Summary': [y_score.round(3), y_score_knn_optimal.round(3), y_score_ols_optimal.round(3)],
                              '' : ['','',''],
                              'Training Score' : [lr.score(X_train, y_train).round(4), '', ''],
                              'Testing Score' : [lr.score(X_test, y_test).round(4), '', ''],
                              'R-Squared' : [results.rsquared.round(3), '', ''],
                              'Adjusted R-Squared' : [results.rsquared_adj.round(3), '', '']
                              })


model_results.to_excel('A1 Final Model Summary - Team 1 - Section 3.xlsx')


###############################################################################
# 11 Model D (WITH A SUBSET OF THE ORIGNAL DATA)
###############################################################################

# Seeing how the age of mothers made our previous models more significant, we
# we decided to create one final model with a SUBSET of the original data, 
# taking only the observations where 'mage' is equal or higher than 45 years.
# We intend to observer if by limiting the data only to mothers with high-risk
# age, our testing and prediting scores will go up.
    
baby5 = baby3[baby3['mage']>=45]
print(baby5.shape)
#np.where( (baby4['mage']>60), 1,0)
    
    
###### 11.1 Multivariate Regression Model (MODEL D) ############################





# Building a Regression Base Model
lm_fmaps = smf.ols(formula = """bwght ~ 
                           baby5['mage'] +
                           baby5['cigs'] +
                           baby5['drink'] +
                           baby5['feduc'] +
                           baby5['fage']
                           """,
                         data = baby5)

results = lm_fmaps.fit()
print(results.summary())
print(f"""
Parameters:
{results.params.round(2)}

Summary Statistics:
R-Squared:          {results.rsquared.round(3)}
Adjusted R-Squared: {results.rsquared_adj.round(3)}
""")

predict = results.predict()
y_hat   = pd.DataFrame(predict).round(2)
resids  = results.resid.round(2)

# Plotting residuals
residual_analysis = pd.concat(
        [baby5.loc[:,'bwght'],
         y_hat,
         results.resid.round(2)],
         axis = 1)

residual_analysis.to_excel('Baby Residuals.xlsx')

sns.residplot(x = predict,
              y = baby5.loc[:,'bwght'])


plt.show()


###### 11.2 Generalization using Train/Test Split (MODEL D) ###################

baby_data   = baby5.drop(['bwght',
                          'omaps',
                          'fmaps',
                          'meduc',
                            'monpre',
                            'npvis',
                            'male',
                            'mwhte',
                            'mblck',
                            'moth',
                            'fwhte',
                            'fblck',
                            'foth'],
                                axis = 1)

baby_target = baby5.loc[:, 'bwght']


X_train, X_test, y_train, y_test = \
    train_test_split(baby_data,
                     baby_target,                           
                     test_size = 0.10,
                     random_state = 508)


print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)


###### 11.3 Using KNN  on the optimal model (MODEL D) #########################


training_accuracy = []
test_accuracy = []

neighbors_settings = range(1, 40)


for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsRegressor(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train)
    
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))


plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()



print(test_accuracy)

print("The optimal number of neighbors is at index", \
      test_accuracy.index(max(test_accuracy)), \
      "with an optimal score of", \
      max(test_accuracy))

###### 11.4 KNN with Optimized Number of Neighbors (MODEL D) ##################
#The best results occur when k = 3

# Building a model with k = 3
knn_reg = KNeighborsRegressor(algorithm = 'auto',
                              n_neighbors = 3)

# Fitting the model based on the training data
knn_reg_fit = knn_reg.fit(X_train, y_train)

# Scoring the model
y_score_knn_optimal = knn_reg.score(X_test, y_test)

# The score is directly comparable to R-Square
print(y_score_knn_optimal)

# Generating Predictions based on the optimal KNN model
#knn_reg_optimal_pred = knn_reg_fit.predict(X_test)

# Predicting on the X_data that the model has never seen before
y_pred = knn_reg.predict(X_test)

# Printing out prediction values for each test observation
print(f"""
Test set predictions:
{y_pred}
""")
    
# Calling the score method, which compares the predicted values to the actual
# values
y_score = knn_reg.score(X_test, y_test)

# The score is directly comparable to R-Square
print(y_score)    

###### 11.5 Prediction. Does OLS predict better than KNN? (MODEL D) ###########

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr_fit = lr.fit(X_train, y_train)
lr_pred = lr_fit.predict(X_test)


print(f"""
Test set predictions:
{y_pred.round(2)}
""")

# Scoring the model
y_score_ols_optimal = lr_fit.score(X_test, y_test)

# The score is directly comparable to R-Square
print(y_score_ols_optimal)


# Let's compare the testing score to the training score.

print('Training Score', lr.score(X_train, y_train).round(4))
print('Testing Score:', lr.score(X_test, y_test).round(4))

# Printing model results
print(f"""
Full model KNN score:    {y_score.round(3)}
Optimal model KNN score: {y_score_knn_optimal.round(3)}
Optimal model OLS score: {y_score_ols_optimal.round(3)}
""")

###### 11.6 Final Results (MODEL D) ############################################

# Even though this model produces a higher R-suqared and higher Training Score
# compared to our best model (Model C), it return a much lower Testing SCore 
# and therfore the gap between Training and Testing scores is higher.

# Because of this, this IS NOT A GOOD model and MODEL C IS STILL THE BEST ONE.
# Compared to Model C:
    # R-squared increases from 0.720 (Model C) to 0.787 (Model D)
    # Training Score increases from 0.722 (Model C) to 0.806 (Model D)
    # Testing Score DESCREASES from 0.675 (Model C) to 0.623 (Model D)
    

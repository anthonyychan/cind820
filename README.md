# Using Machine Learning Techniques to Predict Factors of Winning in the NBA

In this capstone project, I will be using classification techniques and predictive modelling methods such as Logistic Regression, Naïve Bayes, and XGBoost.

__Sections__
1. Question
2. Getting the Data
3. Data Preparation
4. Methods
5. Models
6. Initial Results

# Question
1. How important are each variable within the sport? 
2. Can we determine which factors affect winning chances of the team? 
3. Can we correlate a certain variable and relate it to a team’s chances of winning in the NBA?

# Getting the Data
Imported libraries:    
> panda <br>
> matplotlib <br>
> numpy <br>
> openpyxl <br>
> pathlib <br>
> seaborn <br>
> sklearn <br>
> xgboost <br>


```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
from pathlib import Path
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from xgboost import XGBClassifier
sns.set()
pd.set_option('display.max_rows', None)
plt.show()
```

# Data Preparation
Dataset was acquired from https://www.kaggle.com/ionaskel/nba-games-stats-from-2014-to-2018

Use read_excel to read the xlsx file acquired from Kaggle


```python
player = pd.read_excel (r'C:\Users\Anthony\Documents\CIND820 Project Data\nba.games.stats.xlsx')
```

Quick Summary of the Dataset
> The dataset includes every NBA game from 2014-2018 and includes the wins and losses and the team statistics for each game <br>


```python
player.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Team</th>
      <th>Game</th>
      <th>Date</th>
      <th>Home</th>
      <th>Opponent</th>
      <th>WINorLOSS</th>
      <th>TeamPoints</th>
      <th>OpponentPoints</th>
      <th>FieldGoals</th>
      <th>FieldGoalsAttempted</th>
      <th>...</th>
      <th>Opp.FreeThrows</th>
      <th>Opp.FreeThrowsAttempted</th>
      <th>Opp.FreeThrows%</th>
      <th>Opp.OffRebounds</th>
      <th>Opp.TotalRebounds</th>
      <th>Opp.Assists</th>
      <th>Opp.Steals</th>
      <th>Opp.Blocks</th>
      <th>Opp.Turnovers</th>
      <th>Opp.TotalFouls</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ATL</td>
      <td>1</td>
      <td>2014-10-29</td>
      <td>Away</td>
      <td>TOR</td>
      <td>L</td>
      <td>102</td>
      <td>109</td>
      <td>40</td>
      <td>80</td>
      <td>...</td>
      <td>27</td>
      <td>33</td>
      <td>0.818</td>
      <td>16</td>
      <td>48</td>
      <td>26</td>
      <td>13</td>
      <td>9</td>
      <td>9</td>
      <td>22</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ATL</td>
      <td>2</td>
      <td>2014-11-01</td>
      <td>Home</td>
      <td>IND</td>
      <td>W</td>
      <td>102</td>
      <td>92</td>
      <td>35</td>
      <td>69</td>
      <td>...</td>
      <td>18</td>
      <td>21</td>
      <td>0.857</td>
      <td>11</td>
      <td>44</td>
      <td>25</td>
      <td>5</td>
      <td>5</td>
      <td>18</td>
      <td>26</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ATL</td>
      <td>3</td>
      <td>2014-11-05</td>
      <td>Away</td>
      <td>SAS</td>
      <td>L</td>
      <td>92</td>
      <td>94</td>
      <td>38</td>
      <td>92</td>
      <td>...</td>
      <td>27</td>
      <td>38</td>
      <td>0.711</td>
      <td>11</td>
      <td>50</td>
      <td>25</td>
      <td>7</td>
      <td>9</td>
      <td>19</td>
      <td>15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ATL</td>
      <td>4</td>
      <td>2014-11-07</td>
      <td>Away</td>
      <td>CHO</td>
      <td>L</td>
      <td>119</td>
      <td>122</td>
      <td>43</td>
      <td>93</td>
      <td>...</td>
      <td>20</td>
      <td>27</td>
      <td>0.741</td>
      <td>11</td>
      <td>51</td>
      <td>31</td>
      <td>6</td>
      <td>7</td>
      <td>19</td>
      <td>30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ATL</td>
      <td>5</td>
      <td>2014-11-08</td>
      <td>Home</td>
      <td>NYK</td>
      <td>W</td>
      <td>103</td>
      <td>96</td>
      <td>33</td>
      <td>81</td>
      <td>...</td>
      <td>8</td>
      <td>11</td>
      <td>0.727</td>
      <td>13</td>
      <td>44</td>
      <td>26</td>
      <td>2</td>
      <td>6</td>
      <td>15</td>
      <td>29</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 40 columns</p>
</div>



The dataset includes the following variables
> Team <br>
> Game <br>
> Date <br>
> Home <br>
> Opponent <br>
> WINorLOSS <br>
> TeamPoints <br>
> OpponentPoints <br>
> FieldGoals <br>
> FieldGoalsAttempted <br>
> FieldGoals% <br>
> 3PointShots <br>
> 3PointShotsAttempted <br>
> 3PointShots% <br>
> FreeThrows <br>
> FreeThrowsAttempted <br>
> FreeThrows% <br>
> OffRebounds <br>
> TotalRebounds <br>
> Assists <br>
> Steals <br>
> Blocks <br>
> Turnovers <br>
> TotalFouls <br>
> Opp.FieldGoals <br>
> Opp.FieldGoalsAttempted <br>
> Opp.FieldGoals% <br>
> Opp.3PointShots <br>
> 3PointShotsAttempted <br>
> 3PointShots% <br>
> Opp.3PointShotsAttempted <br>
> Opp.3PointShots% <br>
> Opp.FreeThrows <br>
> Opp.FreeThrowsAttempted <br>
> Opp.FreeThrows% <br>
> Opp.OffRebounds <br>
> Opp.TotalRebounds <br>
> Opp.Assists <br>
> Opp.Steals <br>
> Opp.Blocks <br>
> Opp.Turnovers <br>
> Opp.TotalFouls <br>


```python
player.shape
```




    (9840, 40)



In this dataset, we have 9840 rows of games in the NBA and we have the data for 40 variables


```python
player.isnull().sum()
```




    Team                        0
    Game                        0
    Date                        0
    Home                        0
    Opponent                    0
    WINorLOSS                   0
    TeamPoints                  0
    OpponentPoints              0
    FieldGoals                  0
    FieldGoalsAttempted         0
    FieldGoals%                 0
    3PointShots                 0
    3PointShotsAttempted        0
    3PointShots%                0
    FreeThrows                  0
    FreeThrowsAttempted         0
    FreeThrows%                 0
    OffRebounds                 0
    TotalRebounds               0
    Assists                     0
    Steals                      0
    Blocks                      0
    Turnovers                   0
    TotalFouls                  0
    Opp.FieldGoals              0
    Opp.FieldGoalsAttempted     0
    Opp.FieldGoals%             0
    Opp.3PointShots             0
    Opp.3PointShotsAttempted    0
    Opp.3PointShots%            0
    Opp.FreeThrows              0
    Opp.FreeThrowsAttempted     0
    Opp.FreeThrows%             0
    Opp.OffRebounds             0
    Opp.TotalRebounds           0
    Opp.Assists                 0
    Opp.Steals                  0
    Opp.Blocks                  0
    Opp.Turnovers               0
    Opp.TotalFouls              0
    dtype: int64



There are no null values for any row of the dataset

# Initial Results

First, using the cat.codes on the WINorLOSS variable will allow us to categorize it and be able to correlate this variable and the others to determine which variables affect a team's chance of winning or losing


```python
player['WINorLOSS']=player['WINorLOSS'].astype('category').cat.codes
playerCorr = player.corr()
```

Below is a heatmap to show the correlation between the variables


```python
# creating heatmap
plt.figure(figsize=(20, 15))
sns.heatmap(playerCorr,
            annot = True,
            fmt = '.2f',
            cmap='Blues')
plt.title('Correlation between variables of the dataset')
plt.show()
```


    
![png](output_18_0.png)
    



```python
# correlation team
k = 10
# finding the most correlated variables
cols = playerCorr.nlargest(k, 'WINorLOSS')['WINorLOSS'].index
print(cols)
```

    Index(['WINorLOSS', 'TeamPoints', 'FieldGoals%', 'FieldGoals', '3PointShots%',
           'Assists', 'TotalRebounds', '3PointShots', 'Blocks', 'FreeThrows'],
          dtype='object')
    

The columns that correlate the most with WINorLOSS are TeamPoints, FieldGoals%, FieldGoals, 3PointShots%, Assists, TotalRebounds, 3PointShots, Blocks, and FreeThrows

# Logistic Regression
- Logistic Regression is our first classification model that we will be using and it is one of the most simple and commonly used machine learning algorithm. It describes and estimates the relationship between one dependent binary variable and independent variables.
- First, we will determine our dependent and independent variables. We will use the variables that correlate the most with the WINorLOSS variable as our independent variable and the WINorLOSS variable as our dependent variable. We will split the data into train and test datasets: X_train, X_test, y_train, and y_test.
- We will run the train and test datasets through the logistic regression model and plot a confusion matrix to determine Accuracy, Precision, and Recall.


```python
X = player[['TeamPoints', 'FieldGoals%', 'FieldGoals', '3PointShots%','Assists', 'TotalRebounds', '3PointShots', 'Blocks', 'FreeThrows']]
y = player['WINorLOSS']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

logistic_regression= LogisticRegression(solver='saga', max_iter=10000)
logistic_regression.fit(X_train,y_train)
y_pred=logistic_regression.predict(X_test)

confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True)
```




    <AxesSubplot:xlabel='Predicted', ylabel='Actual'>




    
![png](output_22_1.png)
    



```python
#calculate accuracy,precision and recall
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
print('Precision: ',precision)
print('Recall: ',recall)
```

    Accuracy:  0.7540650406504065
    Precision:  0.7609912070343725
    Recall:  0.7567567567567568
    

- From the confusion matrix, we get that the accuracy of this logistic regression model gives us a 75.4% accuracy rating

# XGBoost
- XGBoost will be our second classification model and it is an implementation of gradient boosted decision trees designed for speed and performance.
- Use the following code to install the correct libraries to use the XGBoost model


```python
#conda install -c conda-forge xgboost
```


```python
from xgboost import XGBClassifier
```


```python
model = XGBClassifier()
model.fit(X_train, y_train, eval_metric='rmse')
```

    C:\Users\Anthony\anaconda3\lib\site-packages\xgboost\sklearn.py:1224: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)
    




    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
                  gamma=0, gpu_id=-1, importance_type=None,
                  interaction_constraints='', learning_rate=0.300000012,
                  max_delta_step=0, max_depth=6, min_child_weight=1, missing=nan,
                  monotone_constraints='()', n_estimators=100, n_jobs=16,
                  num_parallel_tree=1, predictor='auto', random_state=0,
                  reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
                  tree_method='exact', validate_parameters=1, verbosity=None)




```python
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
```


```python
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, y_test)))
```

    Accuracy: 74.39%
    Mean Absolute Error : 0.25609756097560976
    

- From the confusion matrix, we get that the accuracy of the XGBoost model gives us a 74.4% accuracy rating

# Naïve Bayes
- Our third classification model, Naive Bayes, is built on Bayesian classification methods. These rely on Bayes's theorem, which is an equation describing the relationship of conditional probabilities of statistical quantities.


```python
#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(X_train, y_train)

#Predict the response for test dataset
y_predict = gnb.predict(X_test)
```


```python
#Import scikit-learn metrics module for accuracy calculation
# Model Accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_predict))
```

    Accuracy: 0.7170731707317073
    


```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)
```


```python
print(cm)
```

    [[889 313]
     [383 875]]
    

>Precision is TP/TP+FP<br>
>Recall is TP/TP+FN<br>
>TP = 889<br>
>TN = 875<br>
>FN = 383<br>
>FP = 313<br>

# Are these results acceptable?

- A summary of our results
- Our Logistic regression model returned an accuracy of 75.4%
- Our XGBoost model returned an accuracy of 74.4%
- Our Naïve Bayes model returned an accuracy of 71.7%

- In our literature review, an accuracy rating of 70% and above is an acceptable result. Not every game is played the same and the players are not constant in each game and there will always be factors that cannot be quantified and unpredictable due to the nature of this sport. For example, if the team's best player has been injured for the season, their results for that season will not only impact the chances of winning but also the variables for the entire season.


```python

```

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 17:11:58 2019

@author: lizhiying
"""

#vix_sentiment 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import os 
import datetime
import seaborn as sns 
os.chdir("/Users/lizhiying/Desktop/MLF")


"""""""""""""""""""""""""""""""""""""""""""""read data VIX data"""""""""""""""""""""""

vix_daily = pd.read_csv('VIX daily.csv')
vix_weekly = pd.read_csv('VIX weekly.csv')
vix_monthly = pd.read_csv('VIX monthly.csv')



vix_daily['Date'] = vix_daily['Date'].apply(lambda x:int(''.join([i if int(i)>=10 else '0'+i for i in x.split('/')])))
vix_daily = vix_daily.loc[vix_daily['Date'] >= 20150000,]
vix_daily.head()


#df = pd.read_csv('FULL_DF.csv')
df = pd.read_csv('FULL_DF2.csv')
df.rename(columns = {'Unnamed: 0': 'Date'}, inplace = True)


test_vix = vix_daily.copy()
test_vix['Date']  = test_vix['Date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
test_vix['Gap'] = (test_vix.Date.diff() / pd.to_timedelta('1 day')).fillna(0)
test_vix.index = test_vix['Date']
test_vix.head()


GSPC = pd.read_csv('GSPC.csv')
IXIC = pd.read_csv('IXIC.csv')











"""""""""""""""""""""""""""""""""""""""""transform the data and remove weekend"""""""""""""""""""""""

test_df = df.copy()
test_df['Date']  = test_df['Date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
a = test_df.index
test_df.index = test_df.Date
test_df = test_df.drop(columns = 'Date')

df_monthly = test_df.resample('M').mean()
df_weekly = test_df.resample('W').mean()

from pandas.tseries.offsets import BDay
isBusinessDay = BDay().onOffset
#test_df[test_df.index.dayofweek == 1]
#test_df[test_df.index.map(isBusinessDay)]

days = [i for i in test_df.index]

for i,j in enumerate(days):
    if j not in test_vix.Date:
        days[i] = days[i-1]
days[0] = pd.to_datetime(str(20150101), format='%Y%m%d')
test_df.index = days
test_df['Date'] = test_df.index
df_daily = test_df.groupby('Date').mean()[:-1]

df_daily['Date'] = df_daily.index

#vix_daily['Date'] = vix_daily['Date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))

#df_sameday = pd.merge(df_daily,pd.DataFrame(vix_daily,columns= ['Close','Date','Return']),on=['Date'])

df_daily_n = df_daily.copy()

#df_new = pd.merge(pd.DataFrame(df_daily,columns= ['goldstein','numarticles','avgtone','quad1','quad2','quad3','quad4']),pd.DataFrame(vix_daily,columns= ['Close','Date','Return']),on=['Date'])
 










""""""""""""""""""""""Exploration Data Analysis"""""""""""""""""
eda1 = pd.DataFrame(np.array(df_daily),columns = df_daily.columns)
eda2 = pd.DataFrame(np.array(vix_daily),columns = vix_daily.columns)
eda1.drop(columns=['avg mention'],inplace = True)
eda2 = pd.DataFrame(eda2,columns = ['Close','Return'])

eda = pd.concat( [eda1, eda2], axis=1 )
eda.drop(columns = 'Date',inplace = True)
eda['%Change'] = eda['Return']
eda.drop(columns = 'Return',inplace =True)
##draw the heatmap

plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features',y=1.05,size=15)
sns.heatmap(eda.astype(float).corr(),linewidths=0.1,vmax=1.0,square=True,linecolor='white',annot=True,cmap="Blues")

#df_sameday.drop(columns = 'Date',inplace =True)
#sns.heatmap(df_sameday.astype(float).corr(),linewidths=0.1,vmax=1.0,square=True,linecolor='white',annot=True)
plt.show()

eda.astype(float).describe()




""""""""""Data Preprocessing is Done, Run the models """""""""""""""""""""""""""""

from patsy import dmatrix
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.linear_model import LinearRegression
from sklearn import svm
import os
from sklearn import cross_validation,metrics
os.environ['KMP_DUPLICATE_LIB_OK']='True'



#########Classification##########

'''
y = np.sign(vix_daily['Return'])

X = df_daily[['goldstein','numarticles','avgtone','quad_1','quad_2','quad_3']]

gbm = xgb.XGBClassifier(objective = 'binary:logistic', gamma = 0.01, n_estimators = 100,max_depth = 3)
lm = LinearRegression()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
gbm.fit(X_train,y_train)
y_predict = gbm.predict(X_test)
accuracy_score(y_predict, y_test)


y = vix_daily['sign'][1:]

X = df_daily[['goldstein','numarticles','avgtone','quad_1','quad_2','quad_3']].iloc[1:]

X['pre_close'] = np.array(pre_close)
'''
vix_daily['sign'] = vix_daily['Return'].apply(lambda x: 1 if x>=0 else 0)
X = df_daily[['goldstein','numarticles','avgtone','quad_1','quad_2','quad_3','avg source']][1:]
y = vix_daily['sign'][1:]
X['GSPC_close'] = np.array(GSPC['Adj Close'])
X['GSPC_volume'] = np.array(GSPC['Volume'])
X['IXIC_close'] = np.array(IXIC['Adj Close'])
X['IXIC_volume'] = np.array(IXIC['Volume'])
X['GSPC_diff'] = np.array(GSPC['High']) - np.array(GSPC['Low'])
X['IXIC_diff'] = np.array(IXIC['High']) - np.array(IXIC['Low'])

X.index = y.index
scalar = StandardScaler()
scalar.fit(X)
X = scalar.transform(X)

gbm = xgb.XGBClassifier(objective = 'binary:logistic', gamma = 1, n_estimators = 100,max_depth = 3)
lm = LinearRegression()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=50)
gbm.fit(X_train,y_train)
y_predict = gbm.predict(X_test)



#clf = svm.SVC()
#clf.fit(X_train,y_train)
#y_predict = clf.predict(X_test)




def test_result_classification(y_test,y_predict):
    auc = metrics.roc_auc_score(np.array(y_test),np.array(y_predict))
    f1 = metrics.f1_score(y_test, y_predict)
    accuracy = accuracy_score(y_test, y_predict)
    print('accuracy: %.4f ' %accuracy)
    print('f1 %.4f ' %f1)
    print('auc: %.4f ' %auc)




y_test = np.array(y_test)
y_predict = np.array(y_predict)
test_result_classification(y_test,y_predict)


##########shift########
shift = []
test_auc = []
test_accuracy = []






for i in range(1,30):
    shift.append(i)
    print(i)
    X = df_daily[['goldstein','numarticles','avgtone','quad_1','quad_2','quad_3','avg source']][1:]
    y = vix_daily['sign'][1:]
    X['GSPC_close'] = np.array(GSPC['Adj Close'])
    X['GSPC_volume'] = np.array(GSPC['Volume'])
    X['IXIC_close'] = np.array(IXIC['Adj Close'])
    X['IXIC_volume'] = np.array(IXIC['Volume'])
    X['GSPC_diff'] = np.array(GSPC['High']) - np.array(GSPC['Low'])
    X['IXIC_diff'] = np.array(IXIC['High']) - np.array(IXIC['Low'])
    
    X = X.iloc[:-i]
    y = y.iloc[i:]
    X.index = y.index
    scalar = StandardScaler()
    scalar.fit(X)
    X = scalar.transform(X)
    
    gbm = xgb.XGBClassifier(objective = 'binary:logistic', gamma = 0, n_estimators = 10,max_depth = 2)
    lm = LinearRegression()
    for j in range(1000):
        a = []
        b = []
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10*j)
        gbm.fit(X_train,y_train)
        y_predict = gbm.predict(X_test)
        y_test = np.array(y_test)
        y_predict = np.array(y_predict)
        a.append(metrics.roc_auc_score(y_predict, y_test))
        b.append(metrics.accuracy_score(y_predict,y_test))
    test_auc.append(np.mean(a))
    test_accuracy.append(np.mean(b))

plt.plot(shift,test_auc)
plt.plot(shift,test_accuracy)

###################Regression shift#################

shift = []
mse = []
r_square = []
adj_r = []

for i in range(1,30):
    shift.append(i)
    print(i)
    X = df_daily[['goldstein','numarticles','avgtone','quad_1','quad_2','quad_3','avg source']][1:]
    y = vix_daily['Return'][1:]
    
    X['GSPC_close'] = np.array(GSPC['Adj Close'])
    X['GSPC_volume'] = np.array(GSPC['Volume'])
    X['IXIC_close'] = np.array(IXIC['Adj Close'])
    X['IXIC_volume'] = np.array(IXIC['Volume'])
    X['GSPC_diff'] = np.array(GSPC['High']) - np.array(GSPC['Low'])
    X['IXIC_diff'] = np.array(IXIC['High']) - np.array(IXIC['Low'])
    
    X = X.iloc[:-i]
    y = y.iloc[i:]
    X.index = y.index
    scalar = StandardScaler()
    scalar.fit(X)
    X = scalar.transform(X)
    
    #gbm = xgb.XGBClassifier(objective = 'binary:logistic', gamma = 1, n_estimators = 100,max_depth = 3)
    lm = LinearRegression()
    gbm_lm = xgb.XGBRegressor(objective = 'reg:linear', gamma = 0, n_estimators = 10,max_depth = 2)
    
    for j in range(100):
        a = []
        b = []
        c = []
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100*j)
        gbm_lm.fit(X_train,y_train)
        #lm.fit(X_train,y_train)
        #y_predict = lm.predict(X_test)
        y_predict = gbm_lm.predict(X_test)
        y_test = np.array(y_test)
        y_predict = np.array(y_predict)
        a.append(np.mean((y_test-y_predict)**2))
        r2 = (np.corrcoef(y_test,y_predict)[0,1])**2
        b.append(r2)
        c.append(r2-5/(len(X_test)-6)*(1-r2))
    mse.append(np.mean(a))
    r_square.append(np.mean(b))
    adj_r.append(np.mean(c))


#r_square [:] = [x *5 for x in r_square]
#r_square[1] = r_square[1]/2

plt.style.use('fivethirtyeight')
plt.figure(figsize=(7,6))
fig, ax = plt.subplots()

#ax.plot(shift,mse)
ax.plot(shift,r_square)
ax.set_title("Compariason of signalling effect by different days")
plt.xlabel('Days shifted', fontsize=14)
plt.ylabel('R_square', fontsize=14)
plt.show()



plt.plot(shift,mse)
plt.plot(shift,r_square)


################grid search plot xgboost###############

from sklearn.model_selection import GridSearchCV
X = df_daily[['goldstein','numarticles','avgtone','quad_1','quad_2','quad_3','avg source']]#[1:]
y = vix_daily['sign']#[1:]
'''
X['GSPC_close'] = np.array(GSPC['Adj Close'])
X['GSPC_volume'] = np.array(GSPC['Volume'])
X['IXIC_close'] = np.array(IXIC['Adj Close'])
X['IXIC_volume'] = np.array(IXIC['Volume'])
X['GSPC_diff'] = np.array(GSPC['High']) - np.array(GSPC['Low'])
X['IXIC_diff'] = np.array(IXIC['High']) - np.array(IXIC['Low'])
'''

X.index = y.index
scalar = StandardScaler()
scalar.fit(X)
X = scalar.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=50)

param_grid = {'n_estimators': np.array([10,50,100,200,500]), 'max_depth':[2,3,4,5,6]}
print(param_grid)
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer

scoring = make_scorer(accuracy_score)
grid = GridSearchCV(xgb.XGBClassifier(objective = 'binary:logistic',normalize=False,gamma =0),param_grid, scoring =scoring, cv=10)
grid.fit(X, y)
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Best parameters: ", grid.best_params_)

import pandas as pd
pvt = pd.pivot_table(pd.DataFrame(grid.cv_results_),
    values='mean_test_score', index='param_n_estimators', columns='param_max_depth')

pvt




import seaborn as sns       
plt.figure(figsize=(7,6))
plt.title('Classification Accuracy with max_depth and n_estimators')
ax = sns.heatmap(pvt,cmap="Blues")
plt.show()



from sklearn.model_selection import cross_validate
gbm = xgb.XGBClassifier(objective = 'binary:logistic', n_estimators = 10,max_depth = 1,normalize = False)
a = cross_validate(gbm,X,y , cv = 10 ,
                   scoring = 'accuracy',
                return_train_score = True)
a['test_score'].mean()






#####################grid search plot xgboost regression #################


from sklearn.model_selection import GridSearchCV
X = df_daily[['goldstein','numarticles','avgtone','quad_1','quad_2','quad_3','avg source']][1:]
y = vix_daily['Return'][1:]
'''
X['GSPC_close'] = np.array(GSPC['Adj Close'])
X['GSPC_volume'] = np.array(GSPC['Volume'])
X['IXIC_close'] = np.array(IXIC['Adj Close'])
X['IXIC_volume'] = np.array(IXIC['Volume'])
X['GSPC_diff'] = np.array(GSPC['High']) - np.array(GSPC['Low'])
X['IXIC_diff'] = np.array(IXIC['High']) - np.array(IXIC['Low'])
'''

X.index = y.index
scalar = StandardScaler()
scalar.fit(X)
X = scalar.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=50)

param_grid = {'gamma': np.array([0,0.1,0.2,0.5,1]), 'n_estimators':[10,20,30,50,100]}
print(param_grid)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer

scoring = make_scorer(mean_squared_error)
grid = GridSearchCV(xgb.XGBRegressor(normalize=False), param_grid , scoring = scoring, cv=5)
grid.fit(X, y )
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Best parameters: ", grid.best_params_)

import pandas as pd
pvt = pd.pivot_table(pd.DataFrame(grid.cv_results_),
    values='mean_test_score', index='param_gamma', columns='param_n_estimators')

pvt




import seaborn as sns       
plt.figure(figsize=(7,6))
plt.title('Regression with max_depth and n_estimators')
ax = sns.heatmap(pvt,cmap="Oranges")
plt.show()




##############Random Forest############
from sklearn.ensemble import RandomForestClassifier
X = df_daily[['goldstein','numarticles','avgtone','quad_1','quad_2','quad_3','avg source']]#[1:]
y = vix_daily['sign']#[1:]
'''
X['GSPC_close'] = np.array(GSPC['Adj Close'])
X['GSPC_volume'] = np.array(GSPC['Volume'])
X['IXIC_close'] = np.array(IXIC['Adj Close'])
X['IXIC_volume'] = np.array(IXIC['Volume'])
X['GSPC_diff'] = np.array(GSPC['High']) - np.array(GSPC['Low'])
X['IXIC_diff'] = np.array(IXIC['High']) - np.array(IXIC['Low'])
'''

X.index = y.index
scalar = StandardScaler()
scalar.fit(X)
X = scalar.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=50)

param_grid = {'n_estimators': np.array([10,50,100,200,500]), 'max_depth':[1,2,3,4,5,6]}
print(param_grid)
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer

scoring = make_scorer(accuracy_score)
grid = GridSearchCV(RandomForestClassifier(random_state=0),param_grid, scoring =scoring, cv=10)
grid.fit(X, y)
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Best parameters: ", grid.best_params_)

import pandas as pd
pvt = pd.pivot_table(pd.DataFrame(grid.cv_results_),
    values='mean_test_score', index='param_n_estimators', columns='param_max_depth')

pvt




import seaborn as sns       
plt.figure(figsize=(7,6))
plt.title('Classification Accuracy with max_depth and n_estimators')
ax = sns.heatmap(pvt,cmap="Oranges")
plt.show()





#############Model Preparation###########
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors

X = df_daily[['goldstein','numarticles','avgtone','quad_1','quad_2','quad_3','avg source']][1:]
y = vix_daily['sign'][1:]

X['GSPC_close'] = np.array(GSPC['Adj Close'])
X['GSPC_volume'] = np.array(GSPC['Volume'])
X['IXIC_close'] = np.array(IXIC['Adj Close'])
X['IXIC_volume'] = np.array(IXIC['Volume'])
X['GSPC_diff'] = np.array(GSPC['High']) - np.array(GSPC['Low'])
X['IXIC_diff'] = np.array(IXIC['High']) - np.array(IXIC['Low'])

X.index = y.index
scalar = StandardScaler()
scalar.fit(X)
X = scalar.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=50)

    
gbm = xgb.XGBClassifier(objective = 'binary:logistic',  n_estimators = 10,max_depth = 2)
logistic = LogisticRegression()
randomForest = RandomForestClassifier(n_estimators=100, max_depth=3)
svc = SVC(gamma='auto')
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree')
gbm_list = []
logistic_list = []
randomForest_list = []
svc_list = []
nbrs_list = []
    

for j in range(100):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100*j)
    gbm.fit(X_train,y_train)
    logistic.fit(X_train,y_train)
    randomForest.fit(X_train,y_train)
    svc.fit(X_train,y_train)
    nbrs.fit(X_train,y_train)
    
    y_predict_gbm = np.array(gbm.predict(X_test))
    y_predict_logistic = np.array(logistic.predict(X_test))
    y_predict_randomForest = np.array(randomForest.predict(X_test))
    y_predict_svc = np.array(svc.predict(X_test))
    #y_predict_nbrs = nbrs.kneighbors_graph(X_test).toarray()
    y_test = np.array(y_test)

    gbm_list.append(metrics.roc_auc_score(y_predict_gbm, y_test))
    logistic_list.append(metrics.roc_auc_score(y_predict_logistic, y_test))
    randomForest_list.append(metrics.roc_auc_score(y_predict_randomForest, y_test))
    svc_list.append(metrics.roc_auc_score(y_predict_svc, y_test))
    #nbrs_list.append(metrics.roc_auc_score(y_predict_nbrs, y_test))



print("gbm",np.mean(gbm_list))
print("logistic",np.mean(logistic_list))
print("randomForest",np.mean(randomForest_list))
print("svc",np.mean(svc_list))
print("nbrs",np.mean(nbrs_list))


#############model comparison ###########




# Set data
df = pd.DataFrame({
'group': ['AUC'],
'Random_forest': [np.mean(randomForest_list)],
'Logistic Regression': [np.mean(logistic_list)],
'XG Boost': [np.mean(gbm_list)],
'SVM': [np.mean(svc_list)],
'K-means': [0.5]
})
 
# ------- PART 1: Define a function that do a plot for one line of the dataset!
 
def make_spider( row, title,color):
 
    # number of variable
    categories=list(df)[1:]
    N = len(categories)
     
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
     
    # Initialise the spider plot
    ax = plt.subplot(2,2,row+1, polar=True, )
     
    # If you want the first axis to be on top:
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
     
    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='grey', size=8)
     
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([10,20,30], ["10","20","30"], color="grey", size=7)
    plt.ylim(0,40)
     
    # Ind1
    values=df.loc[row].drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.4)
     
    # Add a title
    plt.title(title, size=11, color=color, y=1.1)
     
    # ------- PART 2: Apply to all individuals
    # initialize the figure
    my_dpi=96
    plt.figure(figsize=(1000/my_dpi, 1000/my_dpi), dpi=my_dpi)
     
    # Create a color palette:
    my_palette = plt.cm.get_cmap("Set2", len(df.index))
     
# Loop to plot
for row in range(0, len(df.index)):
    make_spider( row=row, title='group '+df['group'][row],color = my_palette)





















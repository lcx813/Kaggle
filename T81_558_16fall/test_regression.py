
# coding: utf-8

# In[1]:

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


from scipy import stats
from scipy.stats import norm, skew #for some statistics


pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points


#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8")) #check the files available in the directory


# In[2]:

import pandas as pd
import io
import requests
import numpy as np
from sklearn import metrics



train=pd.read_csv("./t81_558_train.csv",na_values=['NA','?'])
test=pd.read_csv("./t81_558_test.csv",na_values=['NA','?'])


# In[3]:

train.head(5)
train.shape


# In[4]:

test.head(5)
test.shape


# In[5]:

#check the numbers of samples and features
print("The train data size before dropping Id feature is : {} ".format(train.shape))
print("The test data size before dropping Id feature is : {} ".format(test.shape))

#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

#check again the data size after dropping the 'Id' variable
print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 
print("The test data size after dropping Id feature is : {} ".format(test.shape))


# In[6]:

for a in ['a','b','c','d','e','f','g']:

    fig, ax = plt.subplots()
    ax.scatter(x = train[a], y = train['outcome'])
    plt.ylabel('outcome', fontsize=13)
    plt.xlabel(a, fontsize=13)
    plt.show()




# In[7]:

#Deleting outliers
#train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
#train.shape

#Check the graphic again
#fig, ax = plt.subplots()
#ax.scatter(train['GrLivArea'], train['SalePrice'])
#plt.ylabel('SalePrice', fontsize=13)
#plt.xlabel('GrLivArea', fontsize=13)
#plt.show()


# In[8]:

sns.distplot(train['outcome'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['outcome'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('outcome distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['outcome'], plot=plt)
plt.show()
print(train['outcome'])


# In[9]:

#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
#train["SalePrice"] = np.log1p(train["SalePrice"])

#Check the new distribution 
#sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
#(mu, sigma) = norm.fit(train['SalePrice'])
#print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
#plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
#            loc='best')
#plt.ylabel('Frequency')
#plt.title('SalePrice distribution')

#Get also the QQ-plot
#fig = plt.figure()
#res = stats.probplot(train['SalePrice'], plot=plt)
#plt.show()

#print(train['SalePrice'])


# In[10]:

ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.outcome.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['outcome'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))
print(y_train)

print(train['outcome'])






train = all_data[:ntrain]
test = all_data[ntrain:]


# In[22]:

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


# In[23]:

numStages = 3
ypred = np.zeros((train.shape[0], numStages))


# In[24]:

ypred.shape


# In[25]:

train.shape


# In[26]:

test.shape


# In[27]:

y_train.shape


# In[28]:

y_train


# In[29]:

#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)



# In[30]:

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))


# In[31]:

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))


# In[32]:

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)


# In[33]:

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)


# In[34]:

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)


# In[35]:

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)






def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))




import pickle
import math        

def mm_model_fit():
    T = 2
    for t in range(T):
        print("Start: Iteration %d \n" % (t))
        
        for stage in range(numStages):
            print("Stage %d \n" % (stage))
            target_stage = y_train - ypred[:, np.arange(numStages)!=stage].sum(axis=1)
            print('target_stage')
            print(target_stage)
            
            
            if stage == 0:
                reg = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
                reg.fit(train.values,target_stage)
                #print(lasso.coef_)
                ypred[:,stage] = reg.predict(train.values)
                print(ypred)
                pickle.dump(reg, open("stage_"+str(stage), 'wb'))
    
                
                if t == 0:
                    print("first step score (RMSE): {}".format(rmsle(y_train,ypred[:,stage])))
                    
                    
            elif stage == 1:
                
                    
                reg = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
        
                
                #reg = RandomForestRegressor()
    
                reg.fit(train.values,target_stage)
                ypred[:,stage] = reg.predict(train.values)
                print(ypred)
                
                pickle.dump(reg, open("stage_"+str(stage), 'wb'))
                
            
            else:
                if t == T-1:
                    break
                reg = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
                
                reg.fit(train.values,target_stage)
                ypred[:,stage] = reg.predict(train.values)
                print(ypred)
                
                pickle.dump(reg, open("stage_"+str(stage), 'wb'))
                
    print("final score (RMSE): {}".format(rmsle(y_train,ypred.sum(axis=1))))
                
                
score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))




mm_model_fit()     


ypred = np.zeros((test.shape[0], numStages))
for stage in range(numStages):
    
    if stage == 0:
        loaded_model = pickle.load(open("stage_"+str(stage), 'rb'))
        ypred[:,stage] = loaded_model.predict(test.values)
    elif stage == 1:
        loaded_model = pickle.load(open("stage_"+str(stage), 'rb'))
        ypred[:,stage] = loaded_model.predict(test.values)
    else:
        loaded_model = pickle.load(open("stage_"+str(stage), 'rb'))
        ypred[:,stage] = loaded_model.predict(test.values)
    
    
final_ypred = ypred.sum(axis=1)

   
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['outcome'] = np.expm1(final_ypred)
sub.to_csv('submission.csv',index=False)

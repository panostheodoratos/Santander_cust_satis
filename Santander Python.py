
# coding: utf-8

# In[18]:


#Import libraries:
import pandas as pd
import numpy as np
from sklearn import model_selection, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.model_selection import train_test_split
from sklearn import cross_validation
from sklearn.preprocessing import Imputer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
import seaborn as sns
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt2


# In[19]:


from scipy import integrate
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif,chi2
from sklearn.preprocessing import Binarizer, scale
from sklearn.utils import resample


# In[2]:


#Reading Data
Data = pd.read_csv("/Users/panagiotistheodoratos/Downloads/train.csv",sep=",",low_memory=False)


# Data.info()

# In[5]:


Data.describe()


# In[22]:


#Checking Target
(len(Data['TARGET']) - len(Data[ Data['TARGET']==0 ]['TARGET']) )/len(Data['TARGET']) * 100
 


# It seems that it is an unbalanced problem as only 3.95% belongs to unsatisfied category

# In[7]:


#Checking nas
Data.isnull().sum()


# In[8]:


#Imputation with median in case anything is missing
im = Imputer(strategy='median')
im.fit(Data)
Datatemp = im.transform(Data)
Data=pd.DataFrame(Datatemp, columns=Data.columns)
Data.head()


# In[9]:


# remove duplicated columns
def rmv_dublicated(df_train):
    remove = []
    c = df_train.columns
    for i in range(len(c)-1):
        v = df_train[c[i]].values
        for j in range(i+1,len(c)):
            if np.array_equal(v,df_train[c[j]].values):
                remove.append(c[j])
    print("Duplicated columes: ",remove)
    df_train.drop(remove, axis=1, inplace=True)
    # Let's look at the size of the train dataset
    print("After simple preprocess, train:  nrows %d, ncols %d" % df_train.shape)


# In[10]:


rmv_dublicated(Data)  


# In[11]:


# remove constant columns
def rmv_constant(df_train):
    remove = []
    for col in df_train.columns:
        if df_train[col].std() == 0:
            remove.append(col)
    print("Constant columes: ",remove)
    df_train.drop(remove, axis=1, inplace=True)
               
rmv_constant(Data) 


#     Running non parametric correlations to check target relationship witht the rest of the varibles

# In[12]:


Cordf = Data[Data.columns].corr(method='spearman')['TARGET'].sort_values()
Cordf = Cordf.to_frame().reset_index()
Cordf['Abs Corr.'] = [abs(x) for x in Cordf['TARGET']]
Cordf.sort_values('Abs Corr.', inplace=True, ascending=False)
Cordf.reset_index(inplace=True,drop=True)
Cordf


# In[13]:


#Separating Target 
X=Data[Data.columns.difference(['TARGET','Unnamed: 0', 'ID'])]
y=Data['TARGET']


# In[14]:


# Check what features are selected  based on chi2 and f_classif
p = 3 # Third Quantile
 
X_bin = Binarizer().fit_transform(scale(X))
selectChi2 = SelectPercentile(chi2, percentile=p).fit(X_bin, y)
selectF_classif = SelectPercentile(f_classif, percentile=p).fit(X, y)
 
chi2_selected = selectChi2.get_support()
chi2_selected_features = [ f for i,f in enumerate(X.columns) if chi2_selected[i]]
print('Chi2 selected {} features {}.'.format(chi2_selected.sum(),
   chi2_selected_features))
f_classif_selected = selectF_classif.get_support()
f_classif_selected_features = [ f for i,f in enumerate(X.columns) if f_classif_selected[i]]
print('F_classif selected {} features {}.'.format(f_classif_selected.sum(),
   f_classif_selected_features))
selected = chi2_selected & f_classif_selected
print('Chi2 & F_classif selected {} features'.format(selected.sum()))
features = [ f for f,s in zip(X.columns, selected) if s]
print (features)


#     As it is an unbalanced problem we will fit a Random Forest with a balanced class weight 

# In[23]:


#Train and test split and run of random forest with balanced
columns=['TARGET', 'ID']
X=Data.drop(columns, axis=1)
y=Data['TARGET']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=12)
clf = RandomForestClassifier(random_state=10,n_estimators=300,class_weight="balanced")
clf.fit(X_train,y_train)


# In[25]:


def print_results(X_train, X_test, y_train, y_test,clf):
    # Create actual  names for  each predicted category
    prediction_train = clf.predict(X_train)
    prediction_test = clf.predict(X_test)
    # Create confusion matrix train
    print(" Confusion matrix on train:\n",
          pd.crosstab(y_train, prediction_train, rownames=[' Actual Unsat.'], colnames=['Predicted Unsat.']))
    # Create confusion matrix test
    print("\n", "Confusion matrix on test:\n",
          pd.crosstab(y_test, prediction_test, rownames=[' Actual Unsat.'], colnames=['Predicted Unsat.']))
    print("\n", "Train precision score :", '%.2f' % (precision_score(y_train.values, prediction_train) * 100), "%")
    print(" Train recall score    :", '%.2f' % (recall_score(y_train.values, prediction_train) * 100), "%")
    print(" Test precision score  :", '%.2f' % (precision_score(y_test.values, prediction_test) * 100), "%")
    print(" Test recall score     :", '%.2f' % (recall_score(y_test.values, prediction_test) * 100), "%")
    print(" Accuracy on train     :", '%.2f' % (metrics.accuracy_score(y_train, prediction_train) * 100), "%")
    print(" Accuracy on test      :", '%.2f' % (metrics.accuracy_score(y_test, prediction_test) * 100), "%")
   # AUC and Gini train
    predstrainprob = clf.predict_proba(X_train)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_train, predstrainprob, pos_label=1)
    print(" AUC Train:", metrics.auc(fpr, tpr))
    print(" Gini Train:", 2 * metrics.auc(fpr, tpr) - 1)
    # AUC and Gini test
    predstestprob = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, predstestprob, pos_label=1)
    print(" AUC Test:", metrics.auc(fpr, tpr))
    print(" Gini Test:", 2 * metrics.auc(fpr, tpr) - 1)
 
print_results(X_train, X_test, y_train, y_test,clf)


# In[26]:


def create_roc_curve(y_test,X_test,clf):
    prob_test=clf.predict_proba(X_test)[:, 1]
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, prob_test, pos_label=1)
    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
    plt.title('Receiver Operating Characteristic Test')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
             label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.005, 1.0])
    plt.ylim([-0.005, 1.0])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
create_roc_curve(y_test,X_test,clf)    


# In[27]:


def cap_curve(y_test,X_test,clf):
    y_values=y_test
    y_preds_proba=clf.predict_proba(X_test)[:, 1]
    num_pos_obs = int(np.sum(y_values))
    num_count = len(y_values)
    rate_pos_obs = float(num_pos_obs) / float(num_count)
    ideal = pd.DataFrame({'x':[0,rate_pos_obs,1],'y':[0,1,1]})
    xx = np.arange(num_count) / float(num_count - 1)
    y_cap = np.c_[y_values,y_preds_proba]
    y_cap_df_s=pd.DataFrame(data=y_cap)
    y_cap_df_s.index.names=['index']
    y_cap_df_s=y_cap_df_s.sort_values([1], ascending=False).reset_index('index', drop=True)
    yy = np.cumsum(y_cap_df_s[0]) / float(num_pos_obs)
    yy = np.append([0], yy[0:len(y_values)-1]) #add the first curve point (0,0) : for xx=0 we have yy=0
    percent = 0.5
    row_index = np.trunc(num_count * percent)
    row_index=int(row_index)
    val_y1 = yy[row_index]
    val_y2 = yy[row_index+1]
    if val_y1 == val_y2:
        val = val_y1*1.0
    else:
        val_x1 = xx[row_index]
        val_x2 = xx[row_index+1]
        val = val_y1 + ((val_x2 - percent)/(val_x2 - val_x1))*(val_y2 - val_y1)
    sigma_ideal = 1 * xx[num_pos_obs - 1 ] / 2 + (xx[num_count - 1] - xx[num_pos_obs]) * 1
    sigma_model = integrate.simps(yy,xx)
    sigma_random = integrate.simps(xx,xx)
    ar_value = (sigma_model - sigma_random) / (sigma_ideal - sigma_random)
    #ar_label = 'ar value = %s' % ar_value
    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    ax.plot(ideal['x'],ideal['y'], color='grey', label='Perfect Model')
    ax.plot(xx,yy, color='red', label='User Model')
#ax.scatter(xx,yy, color='red')
    ax.plot(xx,xx, color='blue', label='Random Model')
    ax.plot([percent, percent], [0.0, val], color='green', linestyle='--', linewidth=1)
    ax.plot([0, percent], [val, val], color='green', linestyle='--', linewidth=1, label=str(val*100)+'% of positive obs at '+str(percent*100)+'%')
    plt2.xlim(0, 1.02)
    plt2.ylim(0, 1.25)
    plt2.title("CAP Curve - a_r value ="+str(ar_value))
    plt2.xlabel('% of the data')
    plt2.ylabel('% of positive obs')
    plt2.legend()
    plt2.show()
cap_curve(y_test,X_test,clf)    


# In[28]:


def prob_plot(X_train,X_test,clf):
    fig,axs=plt.subplots(1,2)
    predstrainprob= clf.predict_proba(X_train)[:,1]
    Probtrain=pd.Series(predstrainprob)
    Probtrain.plot(kind='hist',title='Predicted Prob. on train',ax=axs[0])
    predstestnprob= clf.predict_proba(X_test)[:,1]
    Probtest=pd.Series(predstestnprob)
    Probtest.plot(kind='hist',title='Predicted Prob. on test',ax=axs[1],color='red')
prob_plot(X_train,X_test,clf)


# In[29]:


def decile_cut(y_train,x_train,clf,cut):
    y_values=y_train.values
    y_preds_proba=clf.predict_proba(x_train)[:, 1]
    y_cap = np.c_[y_values,y_preds_proba]
    tempdf=pd.DataFrame(data=y_cap)
    tempdf=tempdf.sort_values([1], ascending=False)#.reset_index('index', drop=True)
    tempdf['Decile']=pd.qcut(tempdf[1], cut,duplicates='drop')
    tempdf.rename(columns={0 : 'TARGET SUM', 1 : 'Prob'},inplace=True)
    tempdf.groupby('Decile').agg({'TARGET SUM': np.sum, 'Decile': np.size})
    results=pd.DataFrame(tempdf.groupby('Decile').agg({'TARGET SUM': np.sum, 'Decile': np.size}))
    results.rename(columns={'Decile' : 'Obs Count'},inplace=True)
    results['Penetration %']=results['TARGET SUM']/results['Obs Count']*100
    print(results)
decile_cut(y_train,X_train,clf,10)    


# In[30]:


decile_cut(y_test,X_test,clf,10)  


# In[31]:


def imp_plot(clf,X_train):
    feat_importances = pd.Series(clf.feature_importances_, index=X_train.columns)
    feat_importances = feat_importances.nlargest(20)
    feat_importances=feat_importances.sort_values()
    feat_importances.plot(kind='barh')
    plt.xlabel('Importance of features')
    plt.ylabel('Features')
    plt.title('Importance of each feature')               
 
imp_plot(clf,X_train)


#     From the model diagnostics it is obvious that the model is overfitting.
#     Training results look good but this is not the case with the test.
#     We can try to create more cases of the unbalanced category and optimise the parameters.

# In[32]:


############## Balancing ##############
X_train2 = X_train.copy()
X_test2 = X_test.copy()
X_train2 = X_train2.assign(TARGET=y_train.values)
X_test2 = X_test2.assign(TARGET=y_test.values)
# Separate majority and minority classes
df_majority = X_train2[X_train2.TARGET==0]
df_minority = X_train2[X_train2.TARGET==1]
# Upsample minority class
df_minority_upsampled = resample(df_minority,
                                 replace=True,     # sample with replacement
                                 n_samples=25000,    # to match majority class
                                 random_state=123)# reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
train=df_upsampled
train.TARGET.value_counts()


# In[33]:


#Run the same forest, changing some parameters to optimise it
clf2 = RandomForestClassifier(random_state=10,n_estimators=300,max_features=150,max_depth=8)
clf2.fit(train.loc[:, train.columns != 'TARGET'], train['TARGET'])


# In[34]:


print_results(X_train=train.loc[:, train.columns != 'TARGET'],X_test=X_test2.loc[:, X_test2.columns != 'TARGET'],y_train=train['TARGET'],y_test=X_test2['TARGET'],clf=clf2)


# In[35]:


create_roc_curve(y_test=X_test2['TARGET'],X_test=X_test2.loc[:, X_test2.columns != 'TARGET'],clf=clf2)  


# In[36]:


cap_curve(y_test=X_test2['TARGET'],X_test=X_test2.loc[:, X_test2.columns != 'TARGET'],clf=clf2)


# In[37]:


prob_plot(X_train=train.loc[:, train.columns != 'TARGET'],X_test=X_test2.loc[:, X_test2.columns != 'TARGET'],clf=clf2)


# In[38]:


decile_cut(y_train=train['TARGET'],x_train=train.loc[:, train.columns != 'TARGET'],clf=clf2,cut=10)  


# In[39]:


decile_cut(y_train=X_test2['TARGET'],x_train=X_test2.loc[:, X_test2.columns != 'TARGET'],clf=clf2,cut=10)    


# In[40]:


imp_plot(clf=clf2,X_train=train.loc[:, train.columns != 'TARGET'])       


#     After balancing and optimising some of the parameters the results have been improved on the test.
#     In addition, we can perform grid search with cross-validays to optimise even more the parameters.
#     Furthermore, the rank of variable importance has changed.
#     In any case the optomised model should be tested in an out of time sample to assest it's stability.

# In[ ]:


#Choose all predictors except target & IDcols
predictors = [x for x in train.columns if x not in ['TARGET', 'IDcol']]
param_test1 = {'max_features':list(range(30,181,60))}
gsearch1 = GridSearchCV(estimator = RandomForestClassifier(random_state=10,n_estimators=300,max_depth=8),
param_grid = param_test1, scoring='roc_auc',n_jobs=-1,iid=False, cv=5)
gsearch1.fit(train[predictors],train['TARGET'])    


# In[ ]:


gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


# In[ ]:


param_test2 = {'max_depth':list(range(5,16,2)), 'min_samples_split':list(range(3800,4801,200))}
gsearch2 = GridSearchCV(estimator = RandomForestClassifier(n_estimators=400, max_features='sqrt', random_state=10),
param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch2.fit(train[predictors],train['TARGET'])
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_


#     Grid search was taking a lot of time to execute thus I stopped it. I would expect a further improvent of 10% in
#     the final results.
#
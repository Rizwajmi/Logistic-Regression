#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model  import Ridge,Lasso,RidgeCV, LassoCV, ElasticNet, ElasticNetCV, LogisticRegression
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
import seaborn as sns
import pickle


# # LASSO

# __1.LASSO, short for Least Absolute Shrinkage and Selection Operator, is a statistical formula whose main purpose is the     feature selection and regularization of data models.
# 
# __2.The LASSO method regularizes model parameters by shrinking the regression coefficients, reducing some of them to zero.   The feature selection phase occurs after the shrinkage, where every non-zero value is selected to be used in the model.     This method is significant in the minimization of prediction errors that are common in statistical models.
# 
# __3.LASSO offers models with high prediction accuracy. The accuracy increases since the method includes shrinkage of         coefficients, which reduces variance and minimizes bias. It performs best when the number of observations is low and the   number of features is high. It heavily relies on parameter λ, which is the controlling factor in shrinkage. The larger λ   becomes, then the more coefficients are forced to be zero.
# 
# __4.Lasso regression is used in machine learning to prevent overfitting.
# 
LassoCV takes one of the parameter input as “cv” which represents number of folds to be considered while applying cross-validation.
# # RIDGE

# __1.Ridge regression is a technique used to eliminate multicollinearity in data models.
# 
# __2.In a case where observations are fewer than predictor variables, ridge regression is the most appropriate technique.
# 
# __3.Ridge regression constraint variables form a circular shape when plotted, unlike the LASSO plot, which forms a diamond shape.

# # ElasticNet

# __1.The elastic net method performs variable selection and regularization simultaneously.
# 
# __2.The elastic net technique is most appropriate where the dimensional data is greater than the number of samples used.
# 
# __3.Groupings and variables selection are the key roles of the elastic net technique.

# In[ ]:





# In[3]:


df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")


# In[4]:


df


# In[5]:


ProfileReport(df)


# In[6]:


df.isnull().sum()


# In[7]:


df['BMI'] = df['BMI'].replace(0 , df['BMI'].mean())


# In[8]:


df.columns


# In[7]:


df['BloodPressure'] = df['BloodPressure'].replace(0,df['BloodPressure'].mean())


# In[8]:


df['Insulin'] = df['Insulin'].replace(0,df['Insulin'].mean())


# In[9]:


df['Glucose'] = df['Glucose'].replace(0,df['Glucose'].mean())


# In[10]:


df['SkinThickness'] = df['SkinThickness'].replace(0,df['SkinThickness'].mean())


# In[9]:


ProfileReport(df)


# In[10]:


fig ,ax  = plt.subplots(figsize = (20,20))
sns.boxplot(data = df , ax = ax)


# In[11]:


q = df['Insulin'].quantile(.70)
df_new = df[df['Insulin'] < q]


# In[12]:


df_new


# In[13]:


fig ,ax  = plt.subplots(figsize = (20,20))
sns.boxplot(data = df_new , ax = ax)


# In[14]:


q = df['Pregnancies'].quantile(.98)
df_new = df[df['Pregnancies'] < q]

q = df_new['BMI'].quantile(.99)
df_new = df_new[df_new['BMI']< q]

q = df_new['SkinThickness'].quantile(.99)
df_new = df_new[df_new['SkinThickness']< q]

q = df_new['Insulin'].quantile(.95)
df_new = df_new[df_new['Insulin']< q]

q = df_new['DiabetesPedigreeFunction'].quantile(.99)
df_new = df_new[df_new['DiabetesPedigreeFunction']< q]


q = df_new['Age'].quantile(.99)
df_new = df_new[df_new['Age']< q]


# In[15]:


def outlier_removal(self,data):
        def outlier_limits(col):
            Q3, Q1 = np.nanpercentile(col, [75,25])
            IQR= Q3-Q1
            UL= Q3+1.5*IQR
            LL= Q1-1.5*IQR
            return UL, LL

        for column in data.columns:
            if data[column].dtype != 'int64':
                UL, LL= outlier_limits(data[column])
                data[column]= np.where((data[column] > UL) | (data[column] < LL), np.nan, data[column])

        return data


# In[16]:


df_new


# In[50]:


fig ,ax  = plt.subplots(figsize = (20,20))
sns.boxplot(data = df_new , ax = ax)


# In[44]:


fig ,ax  = plt.subplots(figsize = (20,20))
sns.boxplot(data = df , ax = ax)


# In[17]:


ProfileReport(df_new)


# In[18]:


df_new


# In[19]:


y = df_new['Outcome']
y


# In[20]:


X = df_new.drop(columns=['Outcome'])


# In[21]:


X


# In[22]:


scalar = StandardScaler()
ProfileReport(pd.DataFrame(scalar.fit_transform(X)))
X_scaled = scalar.fit_transform(X)


# In[23]:


df_new_scalar = pd.DataFrame(scalar.fit_transform(df_new))
fig ,ax  = plt.subplots(figsize = (20,20))
sns.boxplot(data = df_new_scalar , ax = ax)


# In[24]:


X_scaled


# In[25]:


y


# In[26]:


def vif_score(x):
    scaler = StandardScaler()
    arr = scaler.fit_transform(x)
    return pd.DataFrame([[x.columns[i], variance_inflation_factor(arr,i)] for i in range(arr.shape[1])], columns=["FEATURE", "VIF_SCORE"])


# In[27]:


vif_score(X)


# In[28]:


x_train, x_test, y_train, y_test = train_test_split(X_scaled , y , test_size = .20 , random_state = 144)


# In[29]:


x_train


# In[30]:


x_test


# In[31]:


x_test[0]


# In[32]:


logr_liblinear = LogisticRegression(verbose=1,solver='liblinear')


# In[37]:


logr=logr_liblinear.fit(x_train,y_train )


# In[38]:


logr.predict_proba([x_test[1]])


# In[39]:


logr.predict([x_test[1]])


# In[40]:


logr.predict_log_proba([x_test[1]])


# In[41]:


type(y_test)


# In[42]:


y_test.iloc[1]


# In[43]:


y_test


# In[44]:


logr = LogisticRegression(verbose=1)


# In[45]:


logr.fit(x_train,y_train)


# In[46]:


logr_liblinear


# In[47]:


logr


# In[48]:


y_pred_liblinear = logr_liblinear.predict(x_test)
y_pred_liblinear


# In[49]:


y_pred_default = logr.predict(x_test)


# In[50]:


y_pred_default


# In[51]:


confusion_matrix(y_test,y_pred_liblinear)


# In[52]:


confusion_matrix(y_test,y_pred_default)


# In[53]:


def model_eval(y_true,y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
    accuracy=(tp+tn)/(tp+tn+fp+fn)
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    specificity=tn/(fp+tn)
    F1_Score = 2*(recall * precision) / (recall + precision)
    result={"Accuracy":accuracy,"Precision":precision,"Recall":recall,'Specficity':specificity,'F1':F1_Score}
    return result
model_eval(y_test,y_pred_liblinear)


# In[54]:


model_eval(y_test,y_pred_default)


# In[55]:


auc = roc_auc_score(y_test,y_pred_liblinear)


# In[56]:


roc_auc_score(y_test,y_pred_default)


# In[57]:


fpr, tpr, thresholds  = roc_curve(y_test,y_pred_liblinear)


# In[58]:


plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





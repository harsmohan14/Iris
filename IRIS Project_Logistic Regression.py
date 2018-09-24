
# coding: utf-8

# In[1]:


import numpy as np
import datetime
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats import kurtosis
import scipy.stats as stat
get_ipython().magic('matplotlib inline')
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn import metrics


# In[2]:


iris = load_iris()


# In[3]:


irisdata = pd.DataFrame(iris.data)
irisdata.head()


# In[4]:


featureNames = iris.feature_names


# In[5]:


irisdata.columns = featureNames
irisdata.head()


# In[6]:


iristarget = pd.DataFrame(iris.target)
iristarget.head()


# In[7]:


print (pd.unique(iristarget[0]))


# In[8]:


irisdata.corr()


# In[9]:


irisdata.describe()


# In[10]:


irisdata['sepal length (cm)'].plot()


# In[11]:


plt.boxplot(irisdata['sepal length (cm)'])
plt.show()


# In[12]:


irisdata['sepal length (cm)'].hist()


# In[13]:


q = irisdata['sepal length (cm)'].quantile(0.99)
len(irisdata[irisdata['sepal length (cm)']<=q])
#This shows most of the data is in 3rd quartile, hence no need of outlier treatment in this column


# In[14]:


q = irisdata['sepal width (cm)'].quantile(0.99)
len(irisdata[irisdata['sepal width (cm)']>q])


# In[15]:


irisdata['Target'] = iristarget[0]
irisdata.plot(kind='scatter',x='sepal width (cm)',y='Target')


# In[16]:


irisdata.head()


# In[17]:


iristrain, iristest = train_test_split(irisdata,test_size=0.3,random_state=43)


# In[18]:


iristrain.head()


# In[19]:


iristest.shape


# In[20]:


irisXtrain = iristrain[((iristrain['Target']==1) | (iristrain['Target']==2))].iloc[:,0:4]
irisXtrain.shape


# In[21]:


irisYtrain = iristrain[((iristrain['Target']==1) | (iristrain['Target']==2))].Target
irisYtrain.shape


# In[22]:


irisXtest = iristest[((iristest['Target']==1) | (iristest['Target']==2))].iloc[:,0:4]
irisXtest.shape


# In[23]:


irisYtest = iristest[((iristest['Target']==1) | (iristest['Target']==2))].Target
irisYtest.shape


# In[25]:


from sklearn import linear_model


# In[30]:


reg = linear_model.LogisticRegression()


# In[31]:


irismodel = reg.fit(irisXtrain, irisYtrain)


# In[32]:


predicted = irismodel.predict(irisXtest)
predicted


# In[33]:


accuracy = reg.score(irisXtrain, irisYtrain)
accuracy


# In[34]:


probs = irismodel.predict_proba(irisXtest)
probs


# In[35]:


#examine coefficients
#irismodel.coef_
#pd.DataFrame(zip(irisXtrain.columns, np.transpose(irismodel.coef_[0])))


# In[36]:


a = metrics.classification_report(irisYtest,predicted)
print (a)


# In[37]:


print (metrics.confusion_matrix(irisYtest, predicted))


# In[38]:


#probs[:,1]
#Important to set a threshold like 0.6 below as some of the probabilities are too less
predictedclass = [2 if (x>0.6) else 1 for x in probs[:,1]]


# In[39]:


print (metrics.classification_report(irisYtest,predictedclass))


# In[40]:


print (metrics.confusion_matrix(irisYtest, predictedclass))


# In[41]:


#fpr-false positive rate
#tpr-true positive rate
fpr, tpr, thresholds = metrics.roc_curve(irisYtest,predicted,pos_label=1)


# In[42]:


print (thresholds)


# In[43]:


len(irisYtest)


# In[44]:


plt.plot(fpr,tpr)
plt.show()


# In[2]:


#Using KNN algo
from sklearn.neighbors import KNeighborsClassifier


# In[4]:


knn = KNeighborsClassifier()
knn


# In[45]:


knn.fit(irisXtrain, irisYtrain)


# In[46]:


predictedknn = knn.predict(irisXtest)


# In[47]:


print (metrics.classification_report(irisYtest, predictedknn))


# In[81]:


#penalty=l1 makes it lasso regularization
irismodel1 = linear_model.LogisticRegression(penalty='l1')
irismodel1
#LogisticRegressionCV can also be used which uses cross validation for better results.
#Eg-clf = LogisticRegressionCV(n_jobs=2, penalty='l1', solver='liblinear', cv=10, scoring = ‘accuracy’, random_state=0)


# In[46]:


reg1 = irismodel1.fit(irisXtrain,irisYtrain)


# In[47]:


reg1.coef_


# In[48]:


predicted1 = reg1.predict(irisXtest)


# In[49]:


print (metrics.confusion_matrix(irisYtest, predicted1))


# In[50]:


print (metrics.classification_report(irisYtest, predictedclass))


# In[42]:


#Doing same thing using Naive Bayes


# In[43]:


from sklearn.naive_bayes import GaussianNB


# In[44]:


NBmodel = GaussianNB()


# In[189]:


regNB = NBmodel.fit(irisXtrain,irisYtrain)


# In[190]:


predictedNB = regNB.predict(irisXtest)


# In[194]:


#regNB.predict_proba(irisXtest)
print (metrics.classification_report(irisYtest, predictedNB))


# In[195]:


print (metrics.confusion_matrix(irisYtest, predictedNB))


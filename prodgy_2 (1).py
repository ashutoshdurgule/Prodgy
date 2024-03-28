#!/usr/bin/env python
# coding: utf-8

# # Titanic Classification

# In[198]:


titanic=pd.read_csv('train.csv')
titanic.head()


# In[199]:


titanic.shape


# In[200]:


sns.countplot(x='Survived', data=titanic)


# In[201]:


sns.countplot(x='Survived',hue='Sex', data=titanic, palette='winter')


# In[202]:


sns.countplot(x='Survived',hue='Pclass', data=titanic, palette='PuBu')


# In[203]:


titanic['Age'].plot.hist()


# In[204]:


titanic['Fare'].plot.hist(bins=20, figsize=(10,5))


# In[205]:


sns.countplot(x='SibSp', data=titanic, palette='rocket')


# In[206]:


titanic['Parch'].plot.hist()


# In[207]:


sns.countplot(x='Parch', data=titanic, palette='summer')


# In[208]:


titanic.isnull().sum()


# In[209]:


sns.heatmap(titanic.isnull(), cmap='spring')


# In[210]:


sns.boxplot(x='Pclass', y='Age', data=titanic)


# In[278]:


titanic.head()


# In[279]:


print(titanic.columns)


# In[280]:


sns.heatmap(titanic.isnull(), cbar=False)


# In[284]:


titanic.isnull().sum()


# In[285]:


titanic.head(2)


# In[286]:


pd.get_dummies(titanic['Sex']).head()


# In[283]:


sex= pd.get_dummies(titanic['Sex'], drop_first=True)
sex.head(3)


# In[287]:


embark= pd.get_dummies(titanic['Embarked'])


# In[288]:


embark.head(3)


# In[289]:


embark=pd.get_dummies(titanic['Embarked'], drop_first=True)


# In[290]:


embark.head(3)


# In[291]:


Pcl=pd.get_dummies(titanic['Pclass'], drop_first=True)
Pcl.head(3)


# In[292]:


titanic=pd.concat([titanic, sex, embark, Pcl], axis=1)


# In[293]:


titanic.head(3)


# In[294]:


titanic.drop(['Name','PassengerId','Pclass',"Ticket",'Sex','Embarked'], axis=1, inplace=True)


# In[295]:


titanic.head(3)


# # Train Data

# In[329]:


X=titanic.drop('Survived',axis=1)
y=titanic['Survived']


# In[330]:


from sklearn.model_selection import train_test_split


# In[332]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=4)


# In[341]:


X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)


# # Logistic Regression

# In[342]:


from sklearn.linear_model import LogisticRegression
lm=LogisticRegression()


# In[343]:


lm.fit(X_train, y_train)


# In[344]:


prediction = lm.predict(X_test)


# In[347]:


from sklearn.metrics import classification_report


# In[348]:


from sklearn.metrics import classification_report


# In[349]:


from sklearn.metrics import confusion_matrix


# In[350]:


confusion_matrix(y_test, prediction)


# In[351]:


from sklearn.metrics import accuracy_score


# In[352]:


accuracy_score(y_test, prediction)


# we have the accuracy of 79% which is quite good and the model can predict the data accurately

# In[ ]:





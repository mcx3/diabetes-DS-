#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats 
import statsmodels.api as sm


# In[5]:


# first dataset
diabetes = pd.read_csv('/Users/manisha/Desktop/diabetesspring.csv')

# second dataset
c = pd.read_csv('/Users/manisha/Desktop/adults-with-diabetes-per-100-lghc-indicator-3 (1).csv')
ca = c.drop(columns = ['Geography', 'Lower 95% CL', 'Upper 95% CL', 'Standard Error'])

# third dataset along with some data cleaning 
url = 'https://raw.githubusercontent.com/mcx3/diabetes-DS-/main/Diabetes.csv'
df = pd.read_csv(url)
m = df.drop([0,1])
n = m.drop(columns = ['Unnamed: 7'])
f = n.drop([21])
d = f.rename(columns = {'Diagnosed Diabetes; Gender; All Ages; Age-Adjusted Percentage; National':'Year','Unnamed: 1':'Male %','Unnamed: 2':'Female %', 'Unnamed: 3':'Age 0-44','Unnamed: 4':'Age 45-64','Unnamed: 5':'Age 65-74','Unnamed: 6':'Age 75+'})
us = d.apply(pd.to_numeric)


# In[13]:


diabetes.isnull().sum()


# In[15]:


ca.isnull().sum()


# In[16]:


us.isnull().sum()


# **First I am going to work with the first dataset, labeled diabetes. **

# In[17]:


sns.pairplot(diabetes, hue = "Outcome")


# In[8]:


diabetes.describe()


# In[10]:


cond1 = (diabetes["Outcome"] == 1)
yes = cond1.sum()

cond2 = diabetes["Outcome"]
no = cond2.count()

print("There are", yes, "out of", no, "participants in the dataset who have diabetes.")
print("This means that", yes/no * 100, "% of participants in this study have diabetes.")


# In[31]:


result = stats.linregress(diabetes["Glucose"], diabetes["Insulin"])
plt.scatter(diabetes["Glucose"], diabetes["Insulin"], c = "lavender")
plt.plot(diabetes["Glucose"], result.intercept + result.slope * diabetes["Glucose"], color = "midnightblue")
plt.xlabel("Glucose")
plt.ylabel("Insulin")
plt.title("Glucose VS Insulin")
plt.show()

print("P Value: ", result.pvalue)
print("R Squared: ", result.rvalue ** 2)


# In[36]:


sns.lmplot(x = "Glucose", y = "Insulin", data = diabetes, fit_reg = True, hue = "Outcome", legend = False)

plt.legend(title = "Diabetes", loc = "upper left", labels = ["No", "Yes"])
plt.title("Glucose VS Insulin")
plt.show()


# In[48]:


figure = plt.figure(figsize = (15,7))
sns.distplot(diabetes[diabetes['Outcome'] == 0]["Glucose"], color = 'lightskyblue',label = 'No') 
sns.distplot(diabetes[diabetes['Outcome'] == 1]["Glucose"], color = 'lightpink',label = 'Yes') 
plt.title('Glucose', fontsize = 16)
plt.legend()
plt.show()


# In[49]:


figure = plt.figure(figsize = (15,7))
sns.distplot(diabetes[diabetes['Outcome'] == 0]["Insulin"], color = 'lightskyblue',label = 'No') 
sns.distplot(diabetes[diabetes['Outcome'] == 1]["Insulin"], color = 'lightpink',label = 'Yes') 
plt.title('Insulin', fontsize = 16)
plt.legend()
plt.show()


# In[51]:


figure = plt.figure(figsize = (15,7))
sns.distplot(diabetes[diabetes['Outcome'] == 0]["Pregnancies"], color = 'lightskyblue',label = 'No') 
sns.distplot(diabetes[diabetes['Outcome'] == 1]["Pregnancies"], color = 'lightpink',label = 'Yes') 
plt.title('Pregnancies', fontsize = 16)
plt.legend()
plt.show()


# In[52]:


figure = plt.figure(figsize = (15,7))
sns.distplot(diabetes[diabetes['Outcome'] == 0]["BloodPressure"], color = 'lightskyblue',label = 'No') 
sns.distplot(diabetes[diabetes['Outcome'] == 1]["BloodPressure"], color = 'lightpink',label = 'Yes') 
plt.title('Blood Pressure', fontsize = 16)
plt.legend()
plt.show()


# In[53]:


figure = plt.figure(figsize = (15,7))
sns.distplot(diabetes[diabetes['Outcome'] == 0]["SkinThickness"], color = 'lightskyblue',label = 'No') 
sns.distplot(diabetes[diabetes['Outcome'] == 1]["SkinThickness"], color = 'lightpink',label = 'Yes') 
plt.title('Skin Thickness', fontsize = 16)
plt.legend()
plt.show()


# In[54]:


figure = plt.figure(figsize = (15,7))
sns.distplot(diabetes[diabetes['Outcome'] == 0]["BMI"], color = 'lightskyblue',label = 'No') 
sns.distplot(diabetes[diabetes['Outcome'] == 1]["BMI"], color = 'lightpink',label = 'Yes') 
plt.title('BMI', fontsize = 16)
plt.legend()
plt.show()


# In[63]:


result = stats.linregress(diabetes["Glucose"], diabetes["BMI"])
plt.scatter(diabetes["Glucose"], diabetes["BMI"], c = "lightsteelblue")
plt.plot(diabetes["Glucose"], result.intercept + result.slope * diabetes["Glucose"], color = "midnightblue")
plt.xlabel("Glucose")
plt.ylabel("BMI")
plt.title("Glucose VS BMI")
plt.show()

print()

print("P Value: ", result.pvalue)
print("R Squared: ", result.rvalue ** 2)


# In[64]:


sns.lmplot(x = "Glucose", y = "BMI", hue = "Outcome", data = diabetes,
           markers=["o", "x"], palette="Set1", legend = False)
plt.legend(title = "Diabetes", loc = "upper left", labels = ["No", "Yes"])
plt.title("Glucose VS BMI")
plt.show()


# In[65]:


figure = plt.figure(figsize = (15,7))
sns.distplot(diabetes[diabetes['Outcome'] == 0]["Age"], color = 'lightskyblue',label = 'No') 
sns.distplot(diabetes[diabetes['Outcome'] == 1]["Age"], color = 'lightpink',label = 'Yes') 
plt.title('Age', fontsize = 16)
plt.legend()
plt.show()


# In[11]:


plt.hist(diabetes["Age"], color = "lavenderblush", edgecolor = "black")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Frequency of Ages")
plt.show()


# **The second dataset: labeled ca**

# In[12]:


ca


# In[27]:


# want to see the different stratas
print(ca['Strata'].unique())


# In[28]:


total = ca.loc[ca['Strata'] == 'Total population']
race = ca.loc[ca['Strata'] == 'Race-Ethnicity']
age = ca.loc[ca['Strata'] == 'Age']
education = ca.loc[ca['Strata'] == 'Education']
income = ca.loc[ca['Strata'] == 'Income']
sex = ca.loc[ca['Strata'] == 'Sex']


# Now I am going to create pivot tables and graphs to help visualize the differences 

# In[42]:


# starting with total population
print(total.pivot_table(index = "Year", values = "Percent", columns = "Strata Name"))
total.pivot_table(index = "Year", values = "Percent", columns = "Strata Name").plot()
plt.ylabel("Percent")
plt.xlabel("Year")
plt.title("Percent of Diabetes In Population Per Year")
plt.show()


# In[61]:


# race
print(race.pivot_table(index = "Year", values = "Percent", columns = "Strata Name"))
race.pivot_table(index = "Year", values = "Percent", columns = "Strata Name").plot(figsize=(15,7))
plt.xlabel("Year")
plt.ylabel("Percent")
plt.title("Percent of Diabetes By Ethnicity Per Year")
plt.show()


# In[60]:


print(age.pivot_table(index = "Year", values = "Percent", columns = "Strata Name"))
age.pivot_table(index = "Year", values = "Percent", columns = "Strata Name").plot(figsize=(15,7))
plt.xlabel("Year")
plt.ylabel("Percent")
plt.title("Percent of Diabetes By Age Per Year")
plt.show()


# In[59]:


print(education.pivot_table(index = "Year", values = "Percent", columns = "Strata Name"))
education.pivot_table(index = "Year", values = "Percent", columns = "Strata Name").plot(figsize=(15,7))
plt.xlabel("Year")
plt.ylabel("Percent")
plt.title("Percent of Diabetes By Education Per Year")
plt.show()


# In[65]:


print(income.pivot_table(index = "Year", values = "Percent", columns = "Strata Name"))
income.pivot_table(index = "Year", values = "Percent", columns = "Strata Name").plot(figsize=(15,7))

plt.xlabel("Year")
plt.ylabel("Percent")
plt.title("Percent of Diabetes By Income Per Year")
plt.show()


# In[66]:


print(sex.pivot_table(index = "Year", values = "Percent", columns = "Strata Name"))
sex.pivot_table(index = "Year", values = "Percent", columns = "Strata Name").plot(figsize=(15,7))

plt.xlabel("Year")
plt.ylabel("Percent")
plt.title("Percent of Diabetes By Gender Per Year")
plt.show()


# Now I will work on the third dataset: us

# In[67]:


us


# In[73]:


us.pivot_table(index = "Year", values = ["Male %", "Female %"]).plot(figsize=(15,7))

plt.xlabel("Year")
plt.ylabel("Percent")
plt.title("Percent of Diabetes By Gender Per Year")
plt.show()


# In[74]:


us.pivot_table(index = "Year", values = ["Age 0-44", "Age 45-64", "Age 65-74", "Age 75+"]).plot(figsize=(15,7))

plt.xlabel("Year")
plt.ylabel("Percent")
plt.title("Percent of Diabetes By Age Per Year")
plt.show()


# In[ ]:





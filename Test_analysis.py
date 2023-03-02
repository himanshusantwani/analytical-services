#!/usr/bin/env python
# coding: utf-8

# In[111]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[112]:


x=np.linspace(0,5,11)
y=x**3


# In[113]:


tips = sns.load_dataset('tips')


# In[114]:


tips.head()


# In[115]:


sns.distplot(tips['total_bill'],kde=False)


# In[116]:


tips['total_bill'].value_counts()


# In[117]:


tips['total_bill'].idxmax()


# In[118]:


tips[tips['total_bill']==tips['total_bill'].max()]


# In[119]:


tips['sex'].value_counts()


# In[120]:


sns.pairplot(tips[['tip','sex']])


# In[121]:


tips.head()


# In[122]:


tips[['total_bill','tip']].corr()


# In[123]:


tips[tips['size']==tips['size'].max()]


# In[124]:


tips['day'].value_counts()


# In[125]:


tips[tips['day']== 'Sat']


# In[126]:


tips.info()


# In[127]:


weekend_ord = tips[(tips['day']=='Sat') | (tips['day']=='Sun')]


# In[128]:


weekend_ord


# In[129]:


tips['day'].nunique()


# In[130]:


tips['day'].value_counts()


# In[131]:


tips.loc[tips['total_bill'].idxmin()]


# In[132]:


sum(tips[tips['size']>2]['total_bill'].value_counts()==1)


# In[133]:


tips[tips['day'].apply(lambda x:x in ['Sat','Sun'])]


# In[134]:


sns.distplot(tips['total_bill'],kde=False,bins=40)


# In[135]:


sns.jointplot(x='total_bill',y='tip',data=tips, kind='reg')


# In[136]:


sns.jointplot(x='total_bill',y='tip',data=tips, kind='kde')


# In[137]:


sns.jointplot(x='total_bill',y='tip',data=tips, kind='hex')


# In[138]:


sns.jointplot(x='total_bill',y='tip',data=tips, kind ='resid')


# In[139]:


sns.pairplot(tips,hue='sex')


# In[140]:


sns.rugplot(tips['total_bill'])


# In[141]:


sns.kdeplot(tips['total_bill'])


# In[142]:


sns.barplot(x='sex',y='total_bill',data=tips)


# In[143]:


tips.groupby('sex').mean()['total_bill']


# In[144]:


sns.countplot(x='sex',data=tips)


# In[145]:


sns.boxplot(x='day',y='total_bill',data=tips,hue='smoker')


# In[146]:


tips.head(3).info()


# In[147]:


sns.boxplot(x='sex',y='total_bill',data=tips,hue='smoker')


# In[148]:


sns.catplot(x='total_bill', y='tip',data=tips,kind='violin',hue='sex')


# In[149]:


sns.swarmplot(x='day',y='total_bill',data=tips,hue='sex')


# In[150]:


sns.violinplot(x='day',y='total_bill',data=tips,hue='sex')


# In[151]:


sns.stripplot(x='day',y='total_bill',data=tips,hue='sex',jitter=True)


# In[152]:


tf= tips.corr()


# In[153]:


plt.figure(figsize=(8,7))
sns.heatmap(tf,annot=True)


# In[154]:


sns.clustermap(tf,cmap='magma',standard_scale=1)


# In[157]:


g=sns.PairGrid(tips)
g.map_diag(sns.distplot)
g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot)


# In[188]:


g = sns.FacetGrid(data=tips, col='time',row='smoker')
g.map(sns.scatterplot,'total_bill','tip')


# In[162]:


tips.head()


# In[178]:


plt.figure(figsize = (10,9), dpi = 100)
sns.lmplot(x='total_bill', y='tip',data=tips,hue='sex', markers=['o','v'])


# In[187]:


sns.lmplot(x='total_bill', y ='tip', data = tips, col = 'sex', hue = 'sex', palette = 'magma', size = 10)


# In[191]:


sns.lmplot(x='total_bill',y='tip',data=tips,col='day',hue='sex')


# In[192]:


sns.lmplot(x='total_bill',y='tip',data=tips,col='day',hue='sex',aspect = 0.6)


# In[196]:


sns.lmplot(x='total_bill',y='tip',data=tips,col='day',hue='sex',aspect = 0.8,size = 8)


# In[212]:


plt.figure(figsize=(5,4))
sns.set_style('darkgrid')
sns.countplot(x='sex',data=tips)


# In[222]:


sns.set_context('poster',font_scale=0.4)
sns.countplot(x='sex',data=tips,palette = 'rainbow')


# In[223]:


data2 = sns.load_dataset('titanic')


# In[224]:


g = sns.FacetGrid(data=data2, col='sex')
g.map(sns.distplot,'age')


# In[ ]:





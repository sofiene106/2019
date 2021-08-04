#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as st
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.stats import chi2_contingency
from collections import Counter
from math import sqrt


# In[2]:


df = pd.read_csv(r'C:\Users\sofi\Downloads\wetransfer-3ab4ca\2019.csv',sep=';', encoding = 'ISO-8859-1',
                     names=['ID_ORDER', 'DATE-ADD','TIME-ADD','LOCATION','ID_CUSTOMER','LAST_NAME','FIRST_NAME','YX_LIBELLE','BIRTH_YEAR','TELEX','EMAIL','ADRESS','POSTAL_CODE','CITY','ITEM_CODE','CC_LIBELLE','CC_LIBELLE_1','DESIGNATION','PVTTC','QTEFACT','PUTTCNET','MLR_REMISE','GTR_LIBELLE'],
                     dtype={'ID_ORDER':int,'DATE-ADD':object,'TIME-ADD':object,'LOCATION':int,'ID_CUSTOMER':object,'LAST_NAME':object,'FIRST_NAME':object,'YX_LIBELLE':object,'BIRTH_YEAR':object,'TELEX':object,'EMAIL':object,'ADRESS':object,'POSTAL_CODE':object,'CITY':object,'ITEM_CODE':object,'CC_LIBELLE':object,'CC_LIBELLE_1':object})

df


# In[3]:


df.info()


# In[4]:


df.isnull().sum()


# In[5]:


df.replace({'0000-00-00': np.nan},inplace=True)


# In[6]:


df.isnull().sum()


# In[7]:


#convert DataFrame column to date-time:'GP_DATEPIECE'
df['DATE-ADD'] = pd.to_datetime(df['DATE-ADD'], format = '%Y-%m-%d')
df['DATE-ADD']= pd.to_datetime(df['DATE-ADD'], errors='ignore')


# In[8]:


df['TIME-ADD'] = pd.to_datetime(df['TIME-ADD'])


# In[9]:


# Create the dictionary for  ETABLISSEMENT
etab = {44:'Djerba',28:'Sasio Geant',31:'Blue Island Carrefour',62:'Central Park',16:'Sousse',8:'Lafayette',130:'Blue Island Palmarium',50:'Sasio Carrefour',24:'Bizerte',40:'Ennasr',56:'Sasio Manzah VI',14:'Blue Island Zephyr',61:'La Soukra',63:'Sasio Menzah V',54:'Nabeul',37:'Sasio Zephyr',134:'Sasio Palmarium',64:'Nabeul',65:'Blue Island Manar',36:'Sasio Manar',25:'Blue Island Djerba',52:'Blue Island Menzah VI',66:'Mehdia',42:'Lac 2',67:'Sfax',68:'Monastir',51:'Blue Island Menzah V',69:'El Kef',35:'Kairouan',15:'Sasio Mseken',27:'Sasio Mseken',60:'Sasio Djerba',18:'Kelibia',41:'Ksar Hellal',74:'Hammamet'}
df['LOCATION'] = df['LOCATION'].map(etab)
#display the first 5 lines
df.isnull().sum()


# In[10]:


df['DATE-ADD'] = pd.to_datetime(df['DATE-ADD'])

df['SEASON'] = (df['DATE-ADD'].dt.month - 1) // 3
df['SEASON'] += (df['DATE-ADD'].dt.month == 3)&(df['DATE-ADD'].dt.day>=20)
df['SEASON'] += (df['DATE-ADD'].dt.month == 6)&(df['DATE-ADD'].dt.day>=21)
df['SEASON'] += (df['DATE-ADD'].dt.month == 9)&(df['DATE-ADD'].dt.day>=23)
df['SEASON'] -= 3*((df['DATE-ADD'].dt.month == 12)&(df['DATE-ADD'].dt.day>=21)).astype(int)


# In[11]:


season={0:'Winter',1:'Spring',2:'Summer',3:'Autumn'}

df['SEASON'] = df['SEASON'].map(season)
df.isnull().sum()


# In[12]:


df = df[['ID_ORDER', 'DATE-ADD','TIME-ADD','SEASON','LOCATION','ID_CUSTOMER','LAST_NAME','FIRST_NAME','YX_LIBELLE','BIRTH_YEAR','TELEX','EMAIL','ADRESS','POSTAL_CODE','CITY','ITEM_CODE','CC_LIBELLE','CC_LIBELLE_1','DESIGNATION','PVTTC','QTEFACT','PUTTCNET','MLR_REMISE','GTR_LIBELLE']]


# In[13]:


df['COVID']='Pre-Covid'
df = df[['ID_ORDER', 'DATE-ADD','TIME-ADD','SEASON','COVID','LOCATION','ID_CUSTOMER','LAST_NAME','FIRST_NAME','YX_LIBELLE','BIRTH_YEAR','TELEX','EMAIL','ADRESS','POSTAL_CODE','CITY','ITEM_CODE','CC_LIBELLE','CC_LIBELLE_1','DESIGNATION','PVTTC','QTEFACT','PUTTCNET','MLR_REMISE','GTR_LIBELLE']]


# In[14]:


df['BIRTH_YEAR'] = pd.to_datetime(df['BIRTH_YEAR'],errors='coerce')
df.isnull().sum()


# In[16]:


df.BIRTH_YEAR[df.BIRTH_YEAR.dt.year > 2019] = np.nan
df.BIRTH_YEAR[df.BIRTH_YEAR.dt.year < 1918] = np.nan
df.isnull().sum()


# In[17]:


df['BIRTH_YEAR'].fillna((df['BIRTH_YEAR'].mean()), inplace=True)


# In[18]:


df.isnull().sum()


# In[19]:


df['BIRTH_YEAR'] = pd.to_datetime(df['BIRTH_YEAR'],errors='coerce')


# In[20]:


df['AGE'] = df['DATE-ADD'].dt.year- df['BIRTH_YEAR'].dt.year


# In[21]:


df['AGE'].max()


# In[22]:


df['AGE'].min()


# In[23]:


max_AGE=df['AGE'].max()
min_AGE=df['AGE'].min()


# In[24]:


cut_age = ['-25 ans','25-40ans', '40-65ans','+65 ans']
cut_bins =[min_AGE, 25, 40, 65, max_AGE]
df['AGE_SEGMENT'] = pd.cut(df['AGE'], bins=cut_bins, labels = cut_age)


# In[25]:


df['AGE_SEGMENT'].value_counts()


# In[26]:


df = df[['ID_ORDER', 'DATE-ADD','TIME-ADD','SEASON','COVID','LOCATION','ID_CUSTOMER','LAST_NAME','FIRST_NAME','YX_LIBELLE','BIRTH_YEAR','AGE','AGE_SEGMENT','TELEX','EMAIL','ADRESS','POSTAL_CODE','CITY','ITEM_CODE','CC_LIBELLE','CC_LIBELLE_1','DESIGNATION','PVTTC','QTEFACT','PUTTCNET','MLR_REMISE','GTR_LIBELLE']]


# In[27]:


df['CONFINEMENT']='NO'
df['CURFEW']='NO'
df = df[['ID_ORDER', 'DATE-ADD','TIME-ADD','SEASON','COVID','CONFINEMENT','CURFEW','LOCATION','ID_CUSTOMER','LAST_NAME','FIRST_NAME','YX_LIBELLE','BIRTH_YEAR','AGE','TELEX','EMAIL','ADRESS','POSTAL_CODE','CITY','ITEM_CODE','CC_LIBELLE','CC_LIBELLE_1','DESIGNATION','PVTTC','QTEFACT','PUTTCNET','MLR_REMISE','GTR_LIBELLE']]


# In[28]:


df['P_MONTH']=0
df['P_MONTH']+=(df['DATE-ADD'].dt.day>10)
df['P_MONTH']+=(df['DATE-ADD'].dt.day>20)
df['P_MONTH']=df['P_MONTH'].astype(int)


# In[29]:


month={0:'Start_of_Month',1:'Middle_of_Month',2:'End_of_Month'}
df['P_MONTH'] = df['P_MONTH'].map(month)


# In[30]:


df = df[['ID_ORDER', 'DATE-ADD','TIME-ADD','SEASON','P_MONTH','COVID','CONFINEMENT','CURFEW','LOCATION','ID_CUSTOMER','LAST_NAME','FIRST_NAME','YX_LIBELLE','BIRTH_YEAR','AGE','TELEX','EMAIL','ADRESS','POSTAL_CODE','CITY','ITEM_CODE','CC_LIBELLE','CC_LIBELLE_1','DESIGNATION','PVTTC','QTEFACT','PUTTCNET','MLR_REMISE','GTR_LIBELLE']]


# In[31]:


from datetime import date
import calendar


df['P_WEEK']=df['DATE-ADD'].dt.strftime('%A')
week={'Monday':'Start_of_Week','Tuesday':'Start_of_Week','Wednesday':'Mid_Week','Thursday':'Mid_Week','Friday':'Mid_Week','Saturday':'Week_End','Sunday':'Week_end'}
df['P_WEEK'] = df['P_WEEK'].map(week)


# In[32]:


df = df[['ID_ORDER', 'DATE-ADD','TIME-ADD','SEASON','P_MONTH','P_WEEK','COVID','CONFINEMENT','CURFEW','LOCATION','ID_CUSTOMER','LAST_NAME','FIRST_NAME','YX_LIBELLE','BIRTH_YEAR','AGE','TELEX','EMAIL','ADRESS','POSTAL_CODE','CITY','ITEM_CODE','CC_LIBELLE','CC_LIBELLE_1','DESIGNATION','PVTTC','QTEFACT','PUTTCNET','MLR_REMISE','GTR_LIBELLE']]


# In[33]:


df['TOTAL_QUANTITY'] = df.groupby(['ID_ORDER'])['DATE-ADD'].transform('count')


# In[34]:


df2= pd.read_csv(r'C:\Users\sofi\Downloads\train_code_postal.csv')
df2 = df2.rename(columns={'code postal': 'POST_CODE'})


# In[35]:


hour=df['TIME-ADD'].dt.hour
hour


# In[36]:


# Create the dictionary for Hour 

H = {0:'Early_morning', 1:'Early_morning', 2:'Early_morning', 3:'Early_morning',4:'Early_morning',5:'Early_morning',6:'Early_morning',7:'Early_morning',8:'Early_morning',9:'Late_morning',10:'Late_morning',11:'Late_morning',12:'Early_afternoon',13:'Early_afternoon',14:'Early_afternoon',15:'Late_afternoon',16:'Late_afternoon',17:'Late_afternoon',18:'Evening',19:'Evening',20:'Evening',21:'Night',22:'Night',23:'Night'}
# Use the dictionary to map the 'Hour'
df['HOUR'] = hour.map(H)
#display the first 5 lines
df.isnull().sum()


# In[37]:


df = df[['ID_ORDER', 'DATE-ADD','TIME-ADD','HOUR','SEASON','P_MONTH','P_WEEK','COVID','CONFINEMENT','CURFEW','LOCATION','ID_CUSTOMER','LAST_NAME','FIRST_NAME','YX_LIBELLE','BIRTH_YEAR','AGE','TELEX','EMAIL','ADRESS','POSTAL_CODE','CITY','ITEM_CODE','CC_LIBELLE','CC_LIBELLE_1','DESIGNATION','PVTTC','QTEFACT','PUTTCNET','MLR_REMISE','GTR_LIBELLE','TOTAL_QUANTITY']]


# In[38]:


df['TIME-ADD'] = pd.Series([val.time() for val in df['TIME-ADD']])
df


# In[39]:


new_row2070 = {'POST_CODE':'2070', 'Delegation':'LA MARSA', 'poverty rate':2.2, 'zone':'urbaine','orientation':'nord-est'}
df2 = df2.append(new_row2070, ignore_index=True)


# In[40]:


df['POSTAL_CODE']=df['POSTAL_CODE'].apply(str)
df2['POST_CODE']=df2['POST_CODE'].apply(str)


# In[41]:


def get_region(code, train):
  try:
    if str(code) in str(train['POST_CODE'].unique()):
        return str(train[train['POST_CODE']== str(code)]['orientation'].values[0])
  except Exception :
    print (str(code))


# In[42]:


df['region'] = df.apply(lambda x: get_region(x.POSTAL_CODE, df2), axis=1)


# In[43]:


df['region'].value_counts()


# In[44]:


df['POSTAL_CODE'][df['region'].isnull()]


# In[45]:


def get_area(code, train):
  try:
    if str(code) in str(train['POST_CODE'].unique()):
        return str(train[train['POST_CODE']== str(code)]['zone'].values[0])
  except Exception :
    print (str(code))


# In[46]:


df['area'] = df.apply(lambda x: get_area(x.POSTAL_CODE, df2), axis=1)


# In[47]:


df = df.rename(columns={'region': 'REGION'})
df = df.rename(columns={'area': 'AREA'})


# In[48]:


cut_class = ['A', 'B', 'C+','C-', 'D', 'E']
cut_bins =[0, 2, 8, 18, 26, 42, 55]
df2['class'] = pd.cut(df2['poverty rate'], bins=cut_bins, labels = cut_class)
df2


# In[49]:


df2.isnull().sum()


# In[57]:


DF=pd.DataFrame()
DF['LOC']=df['LOCATION']


# In[58]:


# Create the dictionary for  ETABLISSEMENT
POS = {'Djerba':4175,'Sasio Geant':2032,'Blue Island Carrefour':2046,'Central Park':1000,'Sousse':4000,'Lafayette':1002,'Blue Island Palmarium':1000,'Sasio Carrefour':2046,'Bizerte':7000,'Ennasr':2083,'Sasio Manzah VI':2091,'Blue Island Zephyr':2070,'La Soukra':2035,'Sasio Menzah V':2091,'Nabeul':8000,'Sasio Zephyr':2070,'Sasio Palmarium':1000,'Nabeul':8000,'Blue Island Manar':2092,'Sasio Manar':2092,'Blue Island Djerba':4180,'Blue Island Menzah VI':2091,'Mehdia':5100,'Lac 2':1053,'Sfax':3100,'Monastir':5000,'Blue Island Menzah V':2091,'El Kef':7100,'Kairouan':3100,'Sasio Mseken':4070,'Sasio Mseken':4070,'Sasio Djerba':4180,'Kelibia':8090,'Ksar Hellal':5070,'Hammamet':8050}
DF['CODE'] = DF['LOC'].map(POS)
#display the first 5 lines
DF


# In[59]:


DF.isnull().sum()


# In[60]:


def get_class22(code, train):
  try:
    if str(code) in str(train['POST_CODE'].unique()):
      return str(train[train['POST_CODE']== str(code)]['class'].values[0])
  except Exception :
    print (str(code))


# In[61]:


DF['CLASS'] = DF.apply(lambda x: get_class22(x.CODE, df2), axis=1)


# In[62]:


DF.isnull().sum()


# In[63]:


DF = DF.rename(columns={'CLASS': 'class2'})


# In[ ]:





# In[64]:


def get_class(code,code2, train,train2):
  try:
    if str(code) in str(train['POST_CODE'].unique()):
      return str(train[train['POST_CODE']== str(code)]['class'].values[0])
    else : 
        return str(train2[train2['LOC']== str(code2)]['class2'].values[0])
  except Exception :
    print (str(code))


# In[65]:


df['CLASS'] = df.apply(lambda x: get_class(x.POSTAL_CODE,x.LOCATION,df2,DF), axis=1)


# In[66]:


df.isnull().sum()


# In[67]:


df['LOCATION'][df['CLASS'].isnull()]


# In[68]:


df.loc[50685,['CLASS']] = 'C-'
df.loc[50819,['CLASS']] = 'C-'
df.loc[53426,['CLASS']] = 'C-'
df.loc[53449,['CLASS']] = 'C-'
df.loc[451964,['CLASS']] = 'C-'
df.loc[197401,['CLASS']] = 'B'
df.loc[199761,['CLASS']] = 'B'
df.loc[265954,['CLASS']] = 'B'
df.loc[198648,['CLASS']] = 'A'
df.loc[198650,['CLASS']] = 'A'
df.loc[199439,['CLASS']] = 'A'


# In[69]:


df.isnull().sum()


# In[70]:


df['CLASS'].value_counts()


# In[71]:


df = df.rename(columns={'BIRTH_YEAR': 'BIRTHDAY'})
df = df.rename(columns={'ITEM_CODE': 'PRODUCT_ID'})
df = df.rename(columns={'CC_LIBELLE_1': 'PRODUCT_NAME'})
df = df.rename(columns={'CC_LIBELLE': 'LABEL'})
df = df.rename(columns={'DESIGNATION': 'COLOR'})
df = df.rename(columns={'PVTTC': 'PRODUCT_PRICE'})
df = df.rename(columns={'PUTTCNET': 'PRODUCT_PRICE_AFTER_REDUCTION'})
df = df.rename(columns={'QTEFACT': 'PRODUCT_QUANTITY'})
df = df.rename(columns={'MLR_REMISE': 'REDUCTION_PERCENT'})
df = df.rename(columns={'GTR_LIBELLE': 'REDUCTION_TYPE'})
df = df.rename(columns={'YX_LIBELLE': 'CUSTOMER_DESCRIPTION'})


# In[72]:


df.to_csv(r'C:\Users\sofi\Desktop\Data2019.csv', index = False)


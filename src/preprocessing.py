import pandas as pd
import numpy as np
df=pd.read_csv("LBW_Dataset.csv")
#Finding the no of missing values in each column
print(df.isnull().sum())
df.shape
#Droping the null rows of Age
df.dropna(subset = ["Age"], inplace=True)
#Droping the null rows of residence
df.dropna(subset = ["Residence"], inplace=True)
#Droping the null rows of residence
df.dropna(subset = ['Education'], inplace=True)
#Updating the nan values of column - delivery phase with the column mode
df['Delivery phase'].fillna(df['Delivery phase'].mode()[0], inplace=True)
#cleaning weight column
uW=[]
for i,row in df.iterrows():
    if(pd.isnull(row['Weight'])):
        dfa=df[df['Age']==row['Age']]     
        uW.append(dfa['Weight'].mean())      #mean weight of columns with same age as the missing data
    else:
        uW.append(row['Weight'])
df['Weight']=uW
#cleaning BP column
ubp=[]             #weight and age tend to affect BP
for index,row in df.iterrows():
    if(pd.isnull(row['BP'])):
        dfa=df[df['Age']==row['Age']]   #selecting columns with same age
        dfa=dfa.iloc[(dfa['Weight']-row['Weight']).abs().argsort()[:4]]     #selecting 4 columns with weights closer to the weight of our selected row
        ubp.append(dfa['BP'].mean())   
    else:
        ubp.append(row['BP'])
df['BP']=ubp
#Updating the nan values of column - HB with the column mean
df['HB'].fillna(value=df['HB'].mean(), inplace=True)
print(df.isnull().sum())
df.shape
df.to_csv("clean.csv")
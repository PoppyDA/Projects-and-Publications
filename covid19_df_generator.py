# -*- coding: utf-8 -*-
# %%
"""
Created on Wed Mar 10 13:45:26 2021

@author: crius
"""
#Working with PowerBI has led to some interesting revelations, namely that 
#there is power in multiple databases, really one for each metric, as opposed 
#to one master database.


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk
import scipy as sp
from github import Github

# %% Get Github info

#g = Github(login_or_token="744b9b2909eb0d410cee18276934510ea8b1273a")
#repo = g.get_repo("PoppyDA/covid19")

#gcode_url = repo.get_contents("geocode_data.csv").download_url

#COVID19 County data

# %% Generate Master Dataframes

url_covid = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv"
url_gcode = "https://raw.githubusercontent.com/PoppyDA/COVID19DATA/main/geocode_data.csv"

df = pd.read_csv(url_covid)
gcode_df = pd.read_csv('data/geocode_data.csv')

#%%%
#Preliminary Transformations

#df.rename(columns = {'fips':'county'}, inplace=True)

df = df[df.county != 'Unknown']
df = df.sort_values(['fips','date'])
df = df.set_index(['state','fips'])
#df = df[df.county != 'Unknown']# Drops all rows with unknown counties
#df['locations'] = df['county'] + df['state']


prev_index = df.index[0]
prev_cases = 0
prev_deaths = 0

new_cases = []
new_deaths = []

for index, row in df.iterrows():
    
    
    if index == prev_index:
        
        new_cases.append(row['cases'] - prev_cases)
        new_deaths.append(row['deaths'] - prev_deaths)
    
    
    else:
        
        new_cases.append(row['cases'])
        new_deaths.append(row['deaths'])
        
    prev_index = index
    prev_cases = row['cases']
    prev_deaths = row['deaths']
    

#Total case/death numbers occasionally drop for a given county as 
#COVID data is shuffled around. Instead of indicating a negative change in 
#total number of cases/deaths--so far no one has become undead--we set negative values to zero. 
#This would reflect a change of zero new cases/deaths for the county.
df = df.reset_index(level=['state','fips'])

df['new_cases'] = [0 if val < 0 else val for val in new_cases]
df['new_deaths'] = [0 if val < 0 else val for val in new_deaths]
#df['county'] = [county + ' County' for county in df.county]


df= df.dropna()

# %% create useful dictionaries and functions

fips_dict =  dict(zip(gcode_df.fips,zip(gcode_df.state, gcode_df.county)))
lat_long_dict = dict(zip(gcode_df.fips,zip(gcode_df.latitude, gcode_df.longitude, gcode_df.state)))
dict_1 = dict({'lat':2, 'long':3})

dates = sorted(df.date.unique())
def lat_long_lookup(fip,lat_long):
    
    try:
        
        return fips_dict[fip][dict_1[lat_long]]
    
    except:
        
        pass

#Useful for sort function
def get_key(item):
    return item[0]

def get_county(fip):
    
    try:
        
        return fips_dict[fip][1]
    
    except:
        
        pass

def get_state(fip):
    
    try:
        
        return fips_dict[fip][0]
    
    except:
        
        pass

def get_date_index(date):
    return dates.index(date)
'''
# %% Generate latitude and longitude dataframe

lat_long_df = pd.DataFrame()
lat_long_df['latitude'] = [lat_long_lookup(county,'lat') for county in df.county]
lat_long_df['longitude'] = [lat_long_lookup(county,'long') for county in df.county]
'''
# %% Rolling Average Data

'''
Generate rolling average data for covid predictions
'''

roll_avg_cases = []
roll_avg_deaths = []
#county = 'Abbeville County'
for fip in df.fips.unique():
    data = df.loc[df['fips']==fip]
    avg_cases = data.new_cases.rolling(7,min_periods=1).mean()
    avg_deaths = data.new_deaths.rolling(7,min_periods=1).mean()
    
    roll_avg_cases = np.concatenate((roll_avg_cases,avg_cases)).tolist()
    roll_avg_deaths = np.concatenate((roll_avg_deaths, avg_deaths)).tolist()

df['roll_avg_cases'] = roll_avg_cases
df['roll_avg_deaths'] = roll_avg_deaths


# %% Initialize

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern, WhiteKernel

past_days = 31
window_size = 7
future_days = 10




# %% Get Prediction Data based on past 30 days.


kernel = C()*Matern()
#gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

x_train = list(range(1,past_days + 1))
x = np.linspace(1,len(x_train)+future_days+1,100)

#x = np.atleast_2d(x)
x_train = x_train[-past_days::window_size]
x_train = np.atleast_2d(x_train).T

#First sum all the columns into one single column that is the total for the region
#e.g. State vs county

#y_train = dataset.sum(axis=1)


#Takes training data, and prediction range, outputs predictions

def get_GPR(y_train):
    
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gpr.fit(x_train,y_train)
    
    return gpr.predict(np.atleast_2d(x).T,return_std=True)
    

predictions = []

for fip in df.fips.unique():
    
    data = df.loc[df['fips'] == fip]
    county = get_county(fip)
   
    #get past 30 days of rolling average data
    y_train_c = data.roll_avg_cases[-past_days::window_size]
    y_train_d = data.roll_avg_deaths[-past_days::window_size]
    
    y_train_c = np.atleast_2d(y_train_c).ravel()
    y_train_d = np.atleast_2d(y_train_d).ravel()
    
    pred_cases, sig_cases = get_GPR(y_train_c)
    pred_deaths, sig_deaths = get_GPR(y_train_d)
    
    pred_cases = pred_cases.tolist()
    pred_deaths= pred_deaths.tolist()
    
    sig_cases= sig_cases.tolist()
    sig_deaths = sig_deaths.tolist()
    
    zip_data = list(zip(x,[fip]*len(pred_cases), pred_cases, pred_deaths ,sig_cases,sig_deaths))
    
    #print(len(predictions))
    predictions.append(zip_data)


# %% consolidate prediction data
predictions = np.array(predictions)
shape = predictions.shape

predictions = predictions.reshape(shape[0]*shape[1],shape[2])


# %% Prediction Dataframes

new_cases_pred = pd.DataFrame(predictions,columns=['x_vals','fip','predict_cases','predict_deaths','sigma_cases','sigma_deaths'])
#new_cases_sig = pd.DataFrame(predictions_sig,columns = 'Sigmas')

new_cases_pred['county'] = new_cases_pred.fip.apply(get_county)
new_cases_pred['state'] = new_cases_pred.fip.apply(get_state)


# %% Save to csv files

df.to_csv('data/covid19_dat.csv')
new_cases_pred.to_csv('data/covid_pred_dat.csv')






















import os
import boto3
import numpy as np
import pandas as pd
import time

import argparse
import logging
import pathlib
import requests
import tempfile



logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()    
    
    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/day.csv"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)

    logger.debug("Reading downloaded data.")
    bike = pd.read_csv(fn, sep=',')
    os.unlink(fn)
    
    #Rename the columns
    bike.rename(columns={'instant':'rec_id','dteday':'datetime','yr':'year','mnth':'month','weathersit':'weather_condition',
                       'hum':'humidity','cnt':'total_count'},inplace=True)

    #Type casting the datetime and numerical attributes to category

    bike['datetime']=pd.to_datetime(bike.datetime, format="%d-%m-%Y")
    bike['season']=bike.season.astype('category')
    bike['year']=bike.year.astype('category')
    bike['month']=bike.month.astype('category')
    bike['holiday']=bike.holiday.astype('category')
    bike['weekday']=bike.weekday.astype('category')
    bike['workingday']=bike.workingday.astype('category')
    bike['weather_condition']=bike.weather_condition.astype('category')

    #TODO - Add quality check to test for Nulls
    
    #create dataframe for outliers
    wind_hum=pd.DataFrame(bike,columns=['windspeed','humidity'])
     #Cnames for outliers                     
    cnames=['windspeed','humidity']       
                      
    for i in cnames:
        q75,q25=np.percentile(wind_hum.loc[:,i],[75,25]) # Divide data into 75%quantile and 25%quantile.
        iqr=q75-q25 #Inter quantile range
        min=q25-(iqr*1.5) #inner fence
        max=q75+(iqr*1.5) #outer fence
        wind_hum.loc[wind_hum.loc[:,i]<min,:i]=np.nan  #Replace with NA
        wind_hum.loc[wind_hum.loc[:,i]>max,:i]=np.nan  #Replace with NA
    #Imputating the outliers by mean Imputation
    wind_hum['windspeed']=wind_hum['windspeed'].fillna(wind_hum['windspeed'].mean())
    wind_hum['humidity']=wind_hum['humidity'].fillna(wind_hum['humidity'].mean())

    #Replacing the imputated windspeed
    bike['windspeed']=bike['windspeed'].replace(wind_hum['windspeed'])
    #Replacing the imputated humidity
    bike['humidity']=bike['humidity'].replace(wind_hum['humidity'])
    
    #Create a new dataset 
    features=bike[['season','month','year','weekday','holiday','workingday','weather_condition','humidity','temp','windspeed']]
    #categorical attributes
    cat_attributes=['season','holiday','workingday','weather_condition','year']
    encoded_features=pd.get_dummies(features,columns=cat_attributes)
    logger.info(f"Shape of transfomed dataframe:: {encoded_features.shape}")

    
    pd.DataFrame(encoded_features).to_csv(f"{base_dir}/processed/data.csv", header=False, index=False)


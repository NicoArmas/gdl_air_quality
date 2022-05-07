import pandas as pd
import numpy as np
import datetime as dt


def delete_columns(df, columns):
    df.drop(columns, axis = 1, inplace = True)


def merge_date_time(df, date_column, time_column = None, set_index = False):
    if not time_column:
        df['DateTime'] = pd.to_datetime(df[date_column])
    else: 
        df['DateTime'] = pd.to_datetime(df[date_column] + ' ' + df[time_column])
    
    delete_columns(df, [date_column, time_column])
    
    if set_index:
        df.set_index('DateTime', drop=True, inplace=True, append=False)


def data_asNodes(df):
    new_df = (df.assign(labels = df.groupby(level = 0).cumcount())
            .groupby([df.index,'labels']).first()
            .unstack('labels')
            .sort_index(axis =1,level = 1)  #qua level era 1
            .droplevel(1,axis = 1))
    
    return new_df
    
    
def clean_dataset(df, measurement):
  delete_columns(df, ['Parameter Code', 
                    'POC', 'Datum', 'Date Local', 'Time Local', 'Units of Measure',
                    'MDL', 'Uncertainty', 'Qualifier', 'Method Type', 'Method Code', 
                    'Method Name', 'State Name', 'County Name', 'Date of Last Change'])

  merge_date_time(df, 'Date GMT', 'Time GMT', True)

  delete_columns(df, ['Latitude', 'Longitude', 'Parameter Name']) 

  df.rename(columns={'Sample Measurement': measurement}, inplace=True)

  return df


def change_indexes_1(df):
  return  df.reset_index().set_index(['DateTime', 'State Code', 'County Code', 'Site Num'])


def change_indexes_2(df):
  return df.reset_index().set_index(['DateTime'])


def join_dfs(df1, df2):
  #return df1.join(df2, on=['DateTime', 'State Code', 'County Code', 'Site Num'])
  return pd.merge(df1, df2, on=['DateTime', 'State Code', 'County Code', 'Site Num'])


def obtain_id(row):
    sc = str(row['State Code'])
    cc = str(row['County Code'])
    sn = str(row['Site Num'])

    return sc + '-' + cc + '-' + sn

def final_operations(final):
  final.drop(['State Code', 'County Code', 'Site Num'], axis = 1, inplace = True)
  new_final = final.groupby([final.index, 'uniqueid']).first().unstack(['uniqueid']).sort_index(axis=1, level=1)
  #new_final = new_final.columns.swaplevel(i = 0, j = -1)
  new_final = new_final.swaplevel(i = 0, j = -1, axis=1)

  return new_final



#OPEN DFS for all elements

#PM25
#df_PM25_2018 = pd.read_csv(r'./data/PM25/PM25_2018.csv')
#df_PM25_2019 = pd.read_csv(r'./data/PM25/PM25_2019.csv')
#df_PM25_2020 = pd.read_csv(r'./data/PM25/PM25_2020.csv')

#SO2
#df_SO2_2018 = pd.read_csv(r'./data/SO2/SO2_2018.csv')
#df_SO2_2019 = pd.read_csv(r'./data/SO2/SO2_2019.csv')
#df_SO2_2020 = pd.read_csv(r'./data/SO2/SO2_2020.csv')

#NO2
#df_NO2_2018 = pd.read_csv(r'./data/NO2/NO2_2018.csv')
#df_NO2_2019 = pd.read_csv(r'./data/NO2/NO2_2019.csv')
#df_NO2_2020 = pd.read_csv(r'./data/NO2/NO2_2020.csv')

# #CO
#df_CO_2018 = pd.read_csv(r'./data/CO/CO_2018.csv')
#df_CO_2019 = pd.read_csv(r'./data/CO/CO_2019.csv')
#df_CO_2020 = pd.read_csv(r'./data/CO/CO_2020.csv')

# #PM10
#df_PM10_2018 = pd.read_csv(r'./data/PM10/PM10_2018.csv')
#df_PM10_2019 = pd.read_csv(r'./data/PM10/PM10_2019.csv')
#df_PM10_2020 = pd.read_csv(r'./data/PM10/PM10_2020.csv')

# #WIND
#df_WIND_2018 = pd.read_csv(r'./data/WIND/WIND_2018.csv')
#df_WIND_2019 = pd.read_csv(r'./data/WIND/WIND_2019.csv')
#df_WIND_2020 = pd.read_csv(r'./data/WIND/WIND_2020.csv')

# #TEMPERATURE
#df_TEMP_2018 = pd.read_csv(r'./data/TEMPERATURE/TEMP_2018.csv')
#df_TEMP_2019 = pd.read_csv(r'./data/TEMPERATURE/TEMP_2019.csv')
#df_TEMP_2020 = pd.read_csv(r'./data/TEMPERATURE/TEMP_2020.csv')

# #PRESSURE
#df_PRE_2018 = pd.read_csv(r'./data/PRESSURE/PRESS_2018.csv')
#df_PRE_2019 = pd.read_csv(r'./data/PRESSURE/PRESS_2019.csv')
#df_PRE_2020 = pd.read_csv(r'./data/PRESSURE/PRESS_2020.csv')



#df_PM25 = pd.concat([df_PM25_2018, df_PM25_2019, df_PM25_2020])
#df_SO2 = pd.concat([df_SO2_2018, df_SO2_2019, df_SO2_2020])
#df_NO2 = pd.concat([df_NO2_2018, df_NO2_2019, df_NO2_2020])
#df_CO = pd.concat([df_CO_2018, df_CO_2019, df_CO_2020])
#df_PM10 = pd.concat([df_PM10_2018, df_PM10_2019, df_PM10_2020])
#df_WIND = pd.concat([df_WIND_2018, df_WIND_2019, df_WIND_2020])
#df_TEMP = pd.concat([df_TEMP_2018, df_TEMP_2019, df_TEMP_2020])
#df_PRE = pd.concat([df_PRE_2018, df_PRE_2019, df_PRE_2020]) 


#df_PM25 = clean_dataset(df_PM25, 'PM25')
#df_SO2 = clean_dataset(df_SO2, 'SO2')
#df_NO2 = clean_dataset(df_NO2, 'NO2')
#df_CO = clean_dataset(df_CO, 'CO')
#df_PM10 = clean_dataset(df_PM10, 'PM10')
#df_WIND = clean_dataset(df_WIND, 'WIND')
#df_TEMP = clean_dataset(df_TEMP, 'TEMP')
#df_PRE = clean_dataset(df_PRE, 'PRE')


#df_PM25 = change_indexes_1(df_PM25)
#df_SO2 = change_indexes_1(df_SO2)
#df_NO2 = change_indexes_1(df_NO2)
#df_CO = change_indexes_1(df_CO)
#df_PM10 = change_indexes_1(df_PM10)
#df_WIND = change_indexes_1(df_WIND)
#df_TEMP = change_indexes_1(df_TEMP)
#df_PRE = change_indexes_1(df_PRE)

#df_PRE.to_csv('df_PRE.csv')

# df_PM25 = pd.read_csv('df_PM25.csv') 
# df_SO2 = pd.read_csv('df_SO2.csv') 
# df_NO2 = pd.read_csv('df_NO2.csv') 
# df_CO = pd.read_csv('df_CO.csv') 
# df_PM10 = pd.read_csv('df_PM10.csv') 
# df_WIND = pd.read_csv('df_WIND.csv') 
# df_TEMP = pd.read_csv('df_TEMP.csv') 
# df_PRE = pd.read_csv('df_PRE.csv') 


# final = join_dfs(df_PM25, df_SO2)
# final = join_dfs(final, df_NO2)
# final = join_dfs(final, df_CO)
# final = join_dfs(final, df_PM10)
# final = join_dfs(final, df_WIND)
# final = join_dfs(final, df_TEMP)
# final = join_dfs(final, df_PRE)

# final.to_csv('final.csv')

#to have the proper dataset: (~30 sec)

import pandas as pd

final = pd.read_csv('final.csv')
final.drop(['Unnamed: 0'], axis=1, inplace=True)
final = change_indexes_2(final)
final.drop(['index'], axis=1, inplace=True)
final['uniqueid'] = final.apply(lambda row: obtain_id(row), axis=1)

new_final = final_operations(final)

print(new_final)
import pandas as pd
import numpy as np
import datetime as dt

import os

from tsl.utils import download_url, extract_zip
import pathlib


class TemporalDataBuilder():

    def __init__(self, data_dir = 'data'):

        self.nodes_to_keep = []
        self.data_dir = pathlib.Path(data_dir)
        self.pk_dir = pathlib.Path(f'{self.data_dir}/final.pk')
        self.csv_dir = pathlib.Path(f'{self.data_dir}/final2.csv')

        if not os.path.exists(self.pk_dir):
            print("Starting build for temporal data... ")
            self.dataset = self.build()
            print("\tDONE!")


        else:

            print("Found temporal data pickle, loading...", end = '')
            self.dataset = pd.read_pickle(self.pk_dir)
            self.select_optimal_nodes(self.dataset)
            print("\tDONE!")
    
   

    def build(self):

       
        print("\tReading temporal data CSV... ", end='')

        final = pd.read_csv(self.csv_dir)

        print("\tDONE!")

        print("\tConstructing dataset multiindex... ", end='')

        #final.drop(['Unnamed: 0'], axis=1, inplace=True)
        final = self.change_indexes_2(final)
        final.drop(['index'], axis=1, inplace=True)
        final['uniqueid'] = final.apply(lambda row: self.obtain_id(row), axis=1)
        print("\tDONE!")


        print("\tPerforming final operations on the dataset... ", end='')

        new_final = self.final_operations(final)
        new_final.index = pd.DatetimeIndex(new_final.index, freq='H')
        print("\tDONE!")

        print("\tSelecting relevant data (PM25 NaNs < 90%)... ", end='')

        self.select_optimal_nodes(new_final)

        print("\tDONE!")



        new_final = new_final.loc[:, self.nodes_to_keep]
        print("\tSaving obtained data to pickle... ", end='')


        new_final.to_pickle(self.pk_dir)
        print("\tDONE!")


        return new_final

    def delete_columns(self, df, columns):
        df.drop(columns, axis = 1, inplace = True)


    def merge_date_time(self, df, date_column, time_column = None, set_index = False):
        if not time_column:
            df['DateTime'] = pd.to_datetime(df[date_column])
        else: 
            df['DateTime'] = pd.to_datetime(df[date_column] + ' ' + df[time_column])
        
        self.delete_columns(df, [date_column, time_column])
        
        if set_index:
            df.set_index('DateTime', drop=True, inplace=True, append=False)


    def data_asNodes(self,df):
        new_df = (df.assign(labels = df.groupby(level = 0).cumcount())
                .groupby([df.index,'labels']).first()
                .unstack('labels')
                .sort_index(axis =1,level = 1)  #qua level era 1
                .droplevel(1,axis = 1))
        
        return new_df
        
        
    def clean_dataset(self, df, measurement):
        self.delete_columns(df, ['Parameter Code', 
                            'POC', 'Datum', 'Date Local', 'Time Local', 'Units of Measure',
                            'MDL', 'Uncertainty', 'Qualifier', 'Method Type', 'Method Code', 
                            'Method Name', 'State Name', 'County Name', 'Date of Last Change'])

        self.merge_date_time(df, 'Date GMT', 'Time GMT', True)

        self.delete_columns(df, ['Latitude', 'Longitude', 'Parameter Name']) 

        df.rename(columns={'Sample Measurement': measurement}, inplace=True)

        return df


    def change_indexes_1(self, df):
        return  df.reset_index().set_index(['DateTime', 'State Code', 'County Code', 'Site Num'])


    def change_indexes_2(self, df):
        return df.reset_index().set_index(['DateTime'])


    def join_dfs(self, df1, df2):
    #return df1.join(df2, on=['DateTime', 'State Code', 'County Code', 'Site Num'])
        return pd.merge(df1, df2, how='outer', on=['DateTime', 'State Code', 'County Code', 'Site Num'])


    def obtain_id(self, row):
        sc = str(row['State Code'])
        cc = str(row['County Code'])
        sn = str(row['Site Num'])

        return sc + '-' + cc + '-' + sn

    def final_operations(self, final):
        final.drop(['State Code', 'County Code', 'Site Num'], axis = 1, inplace = True)
        new_final = final.groupby([final.index, 'uniqueid']).first().unstack(['uniqueid']).sort_index(axis=1, level=1)
        #new_final = new_final.columns.swaplevel(i = 0, j = -1)
        new_final = new_final.swaplevel(i = 0, j = -1, axis=1)

        return new_final

    def select_optimal_nodes(self, df):

        for node in df.columns.get_level_values(0).drop_duplicates():
            keep = self.keep_node(df, node)
            if keep:
                self.nodes_to_keep.append(node)

    def keep_node(self, df, node):
      
        pm25 = df.loc[:, node]['PM25']
        pm25_nans = pm25.isna().sum()
        pm25_perc = round((pm25_nans/len(pm25))*100, 2)

        if pm25_perc < 90:
            return True
        else: 
            return False
   

        # #OPEN DFS for all elements

        

        # #PM25
        # df_PM25_2018 = pd.read_csv(r'./data/raw/PM25/PM25_2018.csv')
        # df_PM25_2019 = pd.read_csv(r'./data/raw/PM25/PM25_2019.csv')
        # df_PM25_2020 = pd.read_csv(r'./data/raw/PM25/PM25_2020.csv')

        # #SO2
        # df_SO2_2018 = pd.read_csv(r'./data/raw/SO2/SO2_2018.csv')
        # df_SO2_2019 = pd.read_csv(r'./data/raw/SO2/SO2_2019.csv')
        # df_SO2_2020 = pd.read_csv(r'./data/raw/SO2/SO2_2020.csv')

        # #NO2
        # df_NO2_2018 = pd.read_csv(r'./data/raw/NO2/NO2_2018.csv')
        # df_NO2_2019 = pd.read_csv(r'./data/raw/NO2/NO2_2019.csv')
        # df_NO2_2020 = pd.read_csv(r'./data/raw/NO2/NO2_2020.csv')

        # # #CO
        # df_CO_2018 = pd.read_csv(r'./data/raw/CO/CO_2018.csv')
        # df_CO_2019 = pd.read_csv(r'./data/raw/CO/CO_2019.csv')
        # df_CO_2020 = pd.read_csv(r'./data/raw/CO/CO_2020.csv')

        # # #PM10
        # df_PM10_2018 = pd.read_csv(r'./data/raw/PM10/PM10_2018.csv')
        # df_PM10_2019 = pd.read_csv(r'./data/raw/PM10/PM10_2019.csv')
        # df_PM10_2020 = pd.read_csv(r'./data/raw/PM10/PM10_2020.csv')

        # # #WIND
        # df_WIND_2018 = pd.read_csv(r'./data/raw/WIND/WIND_2018.csv')
        # df_WIND_2019 = pd.read_csv(r'./data/raw/WIND/WIND_2019.csv')
        # df_WIND_2020 = pd.read_csv(r'./data/raw/WIND/WIND_2020.csv')

        # # #TEMPERATURE
        # df_TEMP_2018 = pd.read_csv(r'./data/raw/TEMPERATURE/TEMP_2018.csv')
        # df_TEMP_2019 = pd.read_csv(r'./data/raw/TEMPERATURE/TEMP_2019.csv')
        # df_TEMP_2020 = pd.read_csv(r'./data/raw/TEMPERATURE/TEMP_2020.csv')

        # # #PRESSURE
        # df_PRE_2018 = pd.read_csv(r'./data/raw/PRESSURE/PRESS_2018.csv')
        # df_PRE_2019 = pd.read_csv(r'./data/raw/PRESSURE/PRESS_2019.csv')
        # df_PRE_2020 = pd.read_csv(r'./data/raw/PRESSURE/PRESS_2020.csv')



        # df_PM25 = pd.concat([df_PM25_2018, df_PM25_2019, df_PM25_2020])
        # df_SO2 = pd.concat([df_SO2_2018, df_SO2_2019, df_SO2_2020])
        # df_NO2 = pd.concat([df_NO2_2018, df_NO2_2019, df_NO2_2020])
        # df_CO = pd.concat([df_CO_2018, df_CO_2019, df_CO_2020])
        # df_PM10 = pd.concat([df_PM10_2018, df_PM10_2019, df_PM10_2020])
        # df_WIND = pd.concat([df_WIND_2018, df_WIND_2019, df_WIND_2020])
        # df_TEMP = pd.concat([df_TEMP_2018, df_TEMP_2019, df_TEMP_2020])
        # df_PRE = pd.concat([df_PRE_2018, df_PRE_2019, df_PRE_2020]) 


        # df_PM25 = self.clean_dataset(df_PM25, 'PM25')
        # df_SO2 = self.clean_dataset(df_SO2, 'SO2')
        # df_NO2 = self.clean_dataset(df_NO2, 'NO2')
        # df_CO = self.clean_dataset(df_CO, 'CO')
        # df_PM10 = self.clean_dataset(df_PM10, 'PM10')
        # df_WIND = self.clean_dataset(df_WIND, 'WIND')
        # df_TEMP = self.clean_dataset(df_TEMP, 'TEMP')
        # df_PRE = self.clean_dataset(df_PRE, 'PRE')


        # df_PM25 = self.change_indexes_1(df_PM25)
        # df_SO2 = self.change_indexes_1(df_SO2)
        # df_NO2 = self.change_indexes_1(df_NO2)
        # df_CO = self.change_indexes_1(df_CO)
        # df_PM10 = self.change_indexes_1(df_PM10)
        # df_WIND = self.change_indexes_1(df_WIND)
        # df_TEMP = self.change_indexes_1(df_TEMP)
        # df_PRE = self.change_indexes_1(df_PRE)

        # df_PRE.to_csv('df_PRE.csv')

        # df_PM25 = pd.read_csv('df_PM25.csv') 
        # df_SO2 = pd.read_csv('df_SO2.csv') 
        # df_NO2 = pd.read_csv('df_NO2.csv') 
        # df_CO = pd.read_csv('df_CO.csv') 
        # df_PM10 = pd.read_csv('df_PM10.csv') 
        # df_WIND = pd.read_csv('df_WIND.csv') 
        # df_TEMP = pd.read_csv('df_TEMP.csv') 
        # df_PRE = pd.read_csv('df_PRE.csv') 


        # final = self.join_dfs(df_PM25, df_SO2)
        # final = self.join_dfs(final, df_NO2)
        # final = self.join_dfs(final, df_CO)
        # final = self.join_dfs(final, df_PM10)
        # final = self.join_dfs(final, df_WIND)
        # final = self.join_dfs(final, df_TEMP)
        # final = self.join_dfs(final, df_PRE)

        # final.to_csv('final.csv')

        # #to have the proper dataset: (~30 sec)



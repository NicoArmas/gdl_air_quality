import pandas as pd
import numpy as np
import datetime as dt

class TemporalDataBuilder():

    def __init__(self):

        self.dataset = self.build()
    
   

    def build(self):

        final = pd.read_csv('final.csv')
        final.drop(['Unnamed: 0'], axis=1, inplace=True)
        final = self.change_indexes_2(final)
        final.drop(['index'], axis=1, inplace=True)
        final['uniqueid'] = final.apply(lambda row: self.obtain_id(row), axis=1)

        new_final = self.final_operations(final)

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
        return pd.merge(df1, df2, on=['DateTime', 'State Code', 'County Code', 'Site Num'])


    def obtain_id(self, row):
        sc = str(row['State Code'])
        cc = str(row['County Code'])
        sn = str(row['Site Num'])

        return sc + '-' + cc + '-' + sn






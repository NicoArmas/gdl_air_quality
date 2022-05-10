#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import os
from re import sub
from tracemalloc import start
from idna import valid_contextj

import pandas as pd
import numpy as np
from tqdm import tqdm
from tsl.ops.similarities import geographical_distance
from tsl.utils import download_url, extract_zip

from tsl.datasets.prototypes import PandasDataset
from temporal_builder import TemporalDataBuilder


DATA_DIR_PATH = 'data'


def infer_mask(df, infer_from='next'):
    """Infer evaluation mask from DataFrame. In the evaluation mask a value is 1
    if it is present in the DataFrame and absent in the :obj:`infer_from` month.

    Args:
        df (pd.Dataframe): The DataFrame.
        infer_from (str): Denotes from which month the evaluation value must be
            inferred. Can be either :obj:`previous` or :obj:`next`.

    Returns:
        pd.DataFrame: The evaluation mask for the DataFrame.
    """
    mask = (~df.isna()).astype('uint8')
    eval_mask = pd.DataFrame(index=mask.index, columns=mask.columns,
                             data=0).astype('uint8')
    if infer_from == 'previous':
        offset = -1
    elif infer_from == 'next':
        offset = 1
    else:
        raise ValueError('`infer_from` can only be one of {}'
                         .format(['previous', 'next']))
    months = sorted(set(zip(mask.index.year, mask.index.month)))
    length = len(months)
    for i in range(length):
        j = (i + offset) % length
        year_i, month_i = months[i]
        year_j, month_j = months[j]
        cond_j = (mask.index.year == year_j) & (mask.index.month == month_j)
        mask_j = mask[cond_j]
        offset_i = 12 * (year_i - year_j) + (month_i - month_j)
        mask_i = mask_j.shift(1, pd.DateOffset(months=offset_i))
        mask_i = mask_i[~mask_i.index.duplicated(keep='first')]
        mask_i = mask_i[np.in1d(mask_i.index, mask.index)]
        i_idx = mask_i.index
        eval_mask.loc[i_idx] = ~mask_i.loc[i_idx] & mask.loc[i_idx]
    return eval_mask


class AirQuality(PandasDataset):

    similarity_options = {'distance'}
    temporal_aggregation_options = {'mean', 'nearest'}
    spatial_aggregation_options = {'mean'}

    def __init__(self,  overwrite_data = False, 
                        is_subgraph = False, 
                        sub_start = None, 
                        sub_size = 0):

        self.sites_url = "https://drive.switch.ch/index.php/s/hJN6mvHd13puOIa/download"

        self.sites_df = None
        self.red_sites_df = None
        self.dist_mat = None

        self.data_path = DATA_DIR_PATH
        self.sites_path = f'{self.data_path}\\aqs_sites.csv'
        self.red_sites_path = f'{self.data_path}\\aqs_sites_reduced.csv'
        self.dist_mat_path = f'{self.data_path}\\dist_matrix.npy'

        self.init_build(overwrite_data)

        self.temporal = TemporalDataBuilder()
        self.dataset = self.temporal.dataset
        self.is_subgraph = is_subgraph
        self.sub_nodes = None
        
        if self.is_subgraph:
            self.sub_nodes = self.get_closest_nodes(sub_start, sub_size)
            self.dataset = self.slice_data(self.sub_nodes)

        super().__init__(dataframe=self.dataset,
                         attributes=dict(dist=self.dist_mat),
                         similarity_score="distance",
                         spatial_aggregation="mean",
                         temporal_aggregation="nearest",
                         name="AQ")
  
    def init_build(self, overwrite):
        
        built_dist, built_sites = self.valid_build()

        if built_dist and built_sites and not overwrite:
            
            print("Found a valid build, loading...")

            self.load_datasets()
        
        else:
            if not built_sites:
                self.download_sites_data(self.sites_url)
                self.sites_df, self.red_sites_df = self.prepare_sites_data()
                if built_dist:
                    self.dist_mat = pd.DataFrame(np.load(self.dist_mat_path, allow_pickle=True))
            
            if not built_dist:
                self.sites_df = pd.read_csv(self.sites_path)
                self.red_sites_df = pd.read_csv(self.red_sites_path)
                self.dist_mat = self.gen_full_dist_matrix()

    def load_datasets(self):
        self.sites_df = pd.read_csv(self.sites_path)
        self.red_sites_df = pd.read_csv(self.red_sites_path)
        self.dist_mat = pd.DataFrame(np.load(self.dist_mat_path, allow_pickle=True))

    def valid_build(self):
        check1 = os.path.exists(self.dist_mat_path)
        check2 = os.path.exists(self.sites_path) 
        check3 = os.path.exists(self.red_sites_path)

        return check1, (check2 and check3)

    def download_sites_data(self, url):
        print("Downloading sites list... ", end = '')
        filename = 'aqs_sites.csv'
        if not os.path.exists(filename):
            download_url(url, self.data_path, filename)            
            print("Sites list download completed!")
        else: print("File already present, skipped sites list download")      

    def prepare_sites_data(self):
        print("Preparing sites data... ", end='')
        sites_full = pd.read_csv(self.sites_path)
        sites_red = sites_full.loc[:, "State Code":"Longitude"]
        sites_red.insert(0, 'ID', value= None)
        sites_red.insert(0, 'Index', value=[i for i in range(sites_red.shape[0])])

        sites_red = generate_id(sites_red)
        sites_red.drop(columns=['State Code', 'County Code', 'Site Number'], inplace=True)

        filename = self.red_sites_path

        if not os.path.exists(filename):
            sites_red.to_csv(filename, index = False)
            print("\tDONE!")
        else: print("File already present, skipped " + filename)

        return sites_full, sites_red

    def gen_full_dist_matrix(self):
        print("Generating distance matrix... ", end = '')
        
        dist = geographical_distance(self.red_sites_df.loc[:, ["Latitude","Longitude"]])

        filename = self.dist_mat_path

        if not os.path.exists(filename):
            print("(this may take a while, do not panic!)", end='')
            dist_npy = dist.to_numpy()
            np.save(filename, dist_npy)
            print("\tDONE!")
        else: print("File already present, skipped " + filename)

        return dist

    def get_closest_nodes(self, start_node, n_nodes):
        """ Get the N closest nodes from a starting node. 

            Arguments:
                start_node -- the node from which to get the N closest
                          nodes
                n_nodes -- the number of desired nodes to get by closeness          

            Returns:
                closest_nodes -- a list of N nodes which are the closest 
                      ones to the starting node       
        """
        
        closest_dist = self.dist_mat.loc[:, [start_node]]
        closest_dist = (closest_dist.sort_values(by=[start_node]))[:n_nodes]
        
        closest_nodes = list(closest_dist.index)
        closest_nodes.sort()
        
        return closest_nodes

    def slice_data(self, nodes):
        return self.dataset.loc[:, nodes]
    

    # def create_subgraph(self, start_node, size):

    #     nodes = self.get_closest_nodes(start_node, size)
    
    #     return AirQualityGraph(nodes, self.dist_mat, self.red_sites_df)

    def lookup_id(self, id):
        df = self.red_sites_df
        return (df.loc[lambda df: df['ID'] == id])['Index']

    def lookup_index(self, index):
        df = self.red_sites_df
        return (df.loc[lambda df: df['Index'] == index])['ID']
    


def generate_id(df):
    print("Generating IDs for nodes... ")
    for row in tqdm(range(df.shape[0])):
        sc = str(float(df.at[row, 'State Code']))
        cc = str(float(df.at[row, 'County Code']))
        sn = str(float(df.at[row, 'Site Number']))

        df.at[row, 'ID']=f'{sc}-{cc}-{sn}'
    print("DONE!")

    return df





# class AirQualityGraph():
#     def __init__(self, nodes, full_dm, full_sites):
        
#         self.nodes = nodes
#         self.size = len(nodes) 

#         self.full_sites = full_sites

#         self.sites_df = None
#         self.dist_mat = None
        
#         self.gen_graph_sites(full_sites)
#         self.gen_dist_mat(full_dm)
        

#     def gen_graph_sites(self, full_s):

#         self.sites_df = full_s.loc[self.nodes]
        

#     def gen_dist_mat(self, full_dm):
        
#         self.dist_mat = full_dm.loc[self.nodes, self.nodes]


#     def get_graph_size(self):
#         """ Return the number of nodes in the graph.
#         """
#         return self.size

#     def print_info(self):
#         print(f'Size: {self.size}')
#         print(f'Nodes: {self.nodes}')
#         print(f'Dist. matrix: \n {self.dist_mat}')

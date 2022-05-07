#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import os
from tracemalloc import start
from idna import valid_contextj

import pandas as pd
import numpy as np
from tqdm import tqdm
from tsl.ops.similarities import geographical_distance
from tsl.utils import download_url, extract_zip


DATA_DIR_PATH = '..\data'


#TODO AirQualityBuilder --> SpatialDataBuilder, in uno script separato insieme a TemporalDataBuilder
#TODO usare obtain_id e non il numero progressivo dei nodi
class AirQuality():

    def __init__(self, overwrite_data = False):
        self.sites_url = "https://drive.switch.ch/index.php/s/hJN6mvHd13puOIa/download"

        self.sites_df = None
        self.red_sites_df = None
        self.dist_mat = None


        self.data_path = DATA_DIR_PATH
        self.sites_path = f'{self.data_path}\\aqs_sites.csv'
        self.red_sites_path = f'{self.data_path}\\aqs_sites_reduced.csv'
        self.dist_mat_path = f'{self.data_path}\\dist_matrix.npy'

        self.init_build(overwrite_data)
  
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

    def create_subgraph(self, start_node, size):

        nodes = self.get_closest_nodes(start_node, size)
    
        return AirQualityGraph(nodes, self.dist_mat, self.red_sites_df)


    def lookup_id(self, node_ind):
        """ Get the ID of a node based on its index.

            Arguments:
                node_ind -- index of the node of interest

            Returns:
                id -- ID of the node
        """
        pass

    


class AirQualityGraph():
    def __init__(self, nodes, full_dm, full_sites):
        
        self.nodes = nodes
        self.size = len(nodes) 

        self.full_sites = full_sites

        self.sites_df = None
        self.dist_mat = None
        
        self.gen_graph_sites(full_sites)
        self.gen_dist_mat(full_dm)
        

    def gen_graph_sites(self, full_s):

        self.sites_df = full_s.loc[self.nodes]
        

    def gen_dist_mat(self, full_dm):
        
        self.dist_mat = full_dm.loc[self.nodes, self.nodes]


    def get_graph_size(self):
        """ Return the number of nodes in the graph.
        """
        return self.size

    def print_info(self):
        print(f'Size: {self.size}')
        print(f'Nodes: {self.nodes}')
        print(f'Dist. matrix: \n {self.dist_mat}')

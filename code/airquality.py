#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import os
from tracemalloc import start

import pandas as pd
from tqdm import tqdm
from tsl.ops.similarities import geographical_distance
from tsl.utils import download_url, extract_zip

DATA_DIR_PATH = "data"

class AirQuality():

    def __init__(self):
        self.sites_url = "https://drive.switch.ch/index.php/s/hJN6mvHd13puOIa/download"

        self.sites_df = None
        self.red_sites_df = None
        self.dist_mat = None
    
    def init_build(self):
        self.download_sites_data(self.sites_url)
        self.sites_df, self.red_sites_df = self.prepare_sites_data()

        self.dist_mat = self.generate_dist_matrix()

    def download_sites_data(self, url):
        print("Downloading sites list... ", end = '')
        filename = 'aqs_sites.csv'
        if not os.path.exists(filename):
            download_url(url, DATA_DIR_PATH, filename)            
            print("Sites list download completed!")
        else: print("File already present, skipped sites list download")      

    def prepare_sites_data(self):
        print("Preparing initial data... ", end='')
        sites_full = pd.read_csv(f'{DATA_DIR_PATH}\\aqs_sites.csv')
        sites_red = sites_full.loc[:, "State Code":"Longitude"]
        filename = f'{DATA_DIR_PATH}\\aqs_sites_reduced.csv'

        if not os.path.exists(filename):
            sites_red.to_csv(filename, index = False)
            print("\tDONE!")
        else: print("File already present, skipped " + filename)

        return sites_full, sites_red

    def generate_dist_matrix(self):
        print("Generating distance matrix... ", end = '')
        
        dist = geographical_distance(self.red_sites_df.loc[:, ["Latitude","Longitude"]])

        filename = f"{DATA_DIR_PATH}\\dist_matrix.csv"

        if not os.path.exists(filename):
            print("(this may take a while, do not panic!)", end='')
            dist.to_csv(filename)
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
        
        closest_dist = (self.dist_mat.loc[:, [start_node]].sort_values(by=[start_node]))[:n_nodes]
        closest_nodes = list(closest_dist.index)
        return closest_nodes

    def lookup_id(self, node_ind):
        """ Get the ID of a node based on its index.

            Arguments:
                node_ind -- index of the node of interest

            Returns:
                id -- ID of the node
        """
        pass


    
def main():
    aq = AirQuality()
    aq.init_build()


if __name__ == '__main__':
    main()
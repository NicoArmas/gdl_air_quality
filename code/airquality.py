#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import os
from re import sub
from tracemalloc import start

import numpy as np
import pandas as pd
from idna import valid_contextj
from matplotlib.pyplot import close
from tqdm import tqdm
from tsl.data.datamodule.splitters import Splitter, disjoint_months
from tsl.datasets.prototypes import PandasDataset
from tsl.ops.similarities import gaussian_kernel, geographical_distance
from tsl.utils import download_url, extract_zip

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

class AirQualitySplitter(Splitter):

    def __init__(self, val_len: int = None,
                 test_months = (3, 6, 9, 12)):
        super(AirQualitySplitter, self).__init__()
        self._val_len = val_len
        self.test_months = test_months

    def fit(self, dataset):
        nontest_idxs, test_idxs = disjoint_months(dataset,
                                                  months=self.test_months,
                                                  synch_mode=HORIZON)
        # take equal number of samples before each month of testing
        val_len = self._val_len
        if val_len < 1:
            val_len = int(val_len * len(nontest_idxs))
        val_len = val_len // len(self.test_months)
        # get indices of first day of each testing month
        delta = np.diff(test_idxs)
        delta_idxs = np.flatnonzero(delta > delta.min())
        end_month_idxs = test_idxs[1:][delta_idxs]
        if len(end_month_idxs) < len(self.test_months):
            end_month_idxs = np.insert(end_month_idxs, 0, test_idxs[0])
        # expand month indices
        month_val_idxs = [np.arange(v_idx - val_len, v_idx) - dataset.window
                          for v_idx in end_month_idxs]
        val_idxs = np.concatenate(month_val_idxs) % len(dataset)
        # remove overlapping indices from training set
        ovl_idxs, _ = dataset.overlapping_indices(nontest_idxs, val_idxs,
                                                  synch_mode=HORIZON,
                                                  as_mask=True)
        train_idxs = nontest_idxs[~ovl_idxs]
        self.set_indices(train_idxs, val_idxs, test_idxs)


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

        self.temporal = TemporalDataBuilder()
        self.dataset = self.temporal.dataset
        self.is_subgraph = is_subgraph
        self.sub_nodes = None

        self.data_path = DATA_DIR_PATH
        self.sites_path = f'{self.data_path}\\aqs_sites.csv'
        self.red_sites_path = f'{self.data_path}\\aqs_sites_reduced.csv'
        self.dist_mat_path = f'{self.data_path}\\dist_matrix.npy'


        self.infer_eval_from = 'next'


        self.init_build(overwrite_data, self.temporal.nodes_to_keep)


        
        if self.is_subgraph:
            self.sub_nodes = self.get_closest_nodes(sub_start, sub_size)
            self.dataset, self.dist_mat = self.slice_data(self.sub_nodes)

        mask, eval_mask = self.create_masks()

        super().__init__(dataframe=self.dataset,
                         attributes=dict(dist=self.dist_mat),
                         mask=mask,
                         similarity_score="distance",
                         spatial_aggregation="mean",
                         temporal_aggregation="mean",
                         default_splitting_method='air_quality',
                         name="AQ")
  
    def init_build(self, overwrite, nodes_to_keep):
        
        built_dist, built_sites = self.valid_build()

        if built_dist and built_sites and not overwrite:
            
            print("Found a valid build, loading... ", end = '')
            self.load_datasets()
            print("\tDONE!")

        
        else:
            if not built_sites:
                self.download_sites_data(self.sites_url)
                self.sites_df, self.red_sites_df = self.prepare_sites_data(nodes_to_keep)
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

    def prepare_sites_data(self, nodes_to_keep):
        print("Preparing sites data... ", end='')
        sites_full = pd.read_csv(self.sites_path)
        sites_red = sites_full.loc[:, "State Code":"Longitude"]
        sites_red.insert(0, 'ID', value= None)
        

        sites_red = generate_id(sites_red)
        sites_red.drop(columns=['State Code', 'County Code', 'Site Number'], inplace=True)
        sites_red = sites_red.set_index('ID')
        sites_red = sites_red.loc[nodes_to_keep]
        sites_red = sites_red.reset_index(level='ID')
        sites_red.insert(0, 'Index', value=[i for i in range(sites_red.shape[0])])


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
        start_node = self.lookup_index(start_node)
        closest_dist = self.dist_mat.iloc[:, start_node]
        closest_dist = (closest_dist.sort_values())[:n_nodes]
        
        closest_nodes = list(closest_dist.index)
        closest_nodes.sort()
        
        return closest_nodes

    def slice_data(self, nodes):

        nodes_ids = [self.lookup_id(n) for n in nodes]

        data = self.dataset.loc[: , (nodes_ids, 'PM25')]       
        dist = self.dist_mat.loc[nodes, nodes]
        return data, dist

    def lookup_index(self, id):
        df = self.red_sites_df
        return (df.loc[df['ID'] == id])['Index'].item()

    def lookup_id(self, index):
        df = self.red_sites_df
        return (df.loc[df['Index'] == index])['ID'].item()

    def create_masks(self):
        mask = (~np.isnan(self.dataset.values)).astype('uint8')  # 1 if value is valid
        
        eval_mask = infer_mask(self.dataset, infer_from=self.infer_eval_from)
        # 1 if value is ground-truth for imputation
        eval_mask = eval_mask.values.astype('uint8')
        # eventually replace nans with weekly mean by hour
        
        return mask, eval_mask

    def get_splitter(self, method = None, **kwargs):
        if method == 'air_quality':
            val_len = kwargs.get('val_len')
            return AirQualitySplitter(test_months=self.test_months,
                                      val_len=val_len)

    def compute_similarity(self, method, **kwargs):
        if method == "distance":
            finite_dist = self.dist_mat.to_numpy().reshape(-1)
            finite_dist = finite_dist[~np.isinf(finite_dist)]
            sigma = finite_dist.std()
            return gaussian_kernel(self.dist_mat.to_numpy(), sigma)



    


def generate_id(df):
    print("Generating IDs for nodes... ")
    for row in tqdm(range(df.shape[0])):
        sc = str(float(df.at[row, 'State Code']))
        cc = str(float(df.at[row, 'County Code']))
        sn = str(float(df.at[row, 'Site Number']))

        df.at[row, 'ID']=f'{sc}-{cc}-{sn}'
    print("DONE!")

    return df





def main():
    dataset = AirQuality(is_subgraph=True, sub_start='6.0-73.0-1201.0', sub_size=100)
    print(dataset)

    print(f"Sampling period: {dataset.freq}\n"
      f"Has missing values: {dataset.has_mask}\n"
      f"Percentage of missing values: {(1 - dataset.mask.mean()) * 100:.2f}%\n"
      f"Has dataset exogenous variables: {dataset.has_exogenous}\n"
      f"Relevant attributes: {', '.join(dataset.attributes.keys())}")

    print(f"Default similarity: {dataset.similarity_score}\n"
      f"Available similarity options: {dataset.similarity_options}\n")

    sim = dataset.get_similarity("distance")  # same as dataset.get_similarity()
    print(sim[:10, :10])  # just check first 10 nodes for readability


if __name__ == '__main__':
    main()

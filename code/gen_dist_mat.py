
from tsl.ops.similarities import geographical_distance
import pandas as pd
import csv
import os 
from tqdm import tqdm

df = pd.read_csv('data\\aqs_sites.csv')


df = df.loc[:, "State Code":"Longitude"]

filename = "data\\aqs_sites_reduced.csv"

if not os.path.exists(filename):
    df.to_csv(filename, index = False)
else: print("File already present, skipped " + filename)

dist = geographical_distance(df.loc[:, ["Latitude","Longitude"]])
codes = df.loc[:, "State Code":"Site Number"]



filename = "data\dist_matrix.csv"

if not os.path.exists(filename):
    dist.to_csv(filename)
else: print("File already present, skipped " + filename)

codes.head()

def generate_id(site):
    state_code, county_code, site_num = site
    return f'{state_code}-{county_code}-{site_num}'

codes_id = pd.DataFrame(columns=["Site ID"])

for index, row in codes.iterrows():
   

   codes_id = pd.concat([codes_id, pd.DataFrame({"Site ID" : [generate_id((codes.loc[index, "State Code"],
                                 codes.loc[index, "County Code"],
                                 codes.loc[index, "Site Number"]))]})], ignore_index= True, axis=0)



with open('data\sites_dist.csv', mode = "w") as distfile:
    t_writer = csv.writer(distfile, delimiter=',')
    t_writer.writerow(["Site1", "Site2", "Dist"])

with open('data\sites_dist.csv', mode = "a", newline='') as distfile:
    t_writer = csv.writer(distfile, delimiter=',')

    for row in tqdm(range(dist.shape[0])):
        site1 = codes_id.loc[row,["Site ID"]].item()
        for col in tqdm(range(dist.shape[1]), leave = False):
            site2 = codes_id.loc[col,["Site ID"]].item()
            t_writer.writerow([site1, site2, dist.loc[row,col]])
        



